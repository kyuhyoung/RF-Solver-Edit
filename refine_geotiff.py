"""
Refine a GeoTIFF satellite image using RF-Solver-Edit + FLUX.

Usage:
    python refine_geotiff.py --input /path/to/input.tif --output /path/to/output.tif
    python refine_geotiff.py --input /path/to/input.tif --output /path/to/output.tif \
        --tile_size 1024 --overlap 128 --inject 20
"""

import sys
sys.path.insert(0, "FLUX_Image_Edit/src")

import os
import argparse
import numpy as np
import torch
import rasterio
from PIL import Image
from tqdm import tqdm
from einops import rearrange

from flux.sampling import denoise, get_schedule, prepare, unpack
from flux.util import load_ae, load_clip, load_flow_model, load_t5
from flux.multi_gpu import distribute_model


def parse_args():
    parser = argparse.ArgumentParser(description="Refine GeoTIFF with RF-Solver-Edit + FLUX")
    parser.add_argument("--input", type=str, required=True, help="Input GeoTIFF path")
    parser.add_argument("--output", type=str, default=None, help="Output GeoTIFF path (default: input_rfsolver.tif)")
    parser.add_argument("--tile_size", type=int, default=1024, help="Tile size (must be divisible by 16)")
    parser.add_argument("--overlap", type=int, default=128, help="Overlap between tiles for blending")
    parser.add_argument("--num_steps", type=int, default=25, help="Number of ODE steps for inversion & denoising")
    parser.add_argument("--inject", type=int, default=20, help="Feature sharing steps (higher=more structure preservation)")
    parser.add_argument("--guidance", type=float, default=2.0, help="Guidance scale for denoising")
    parser.add_argument("--src_prompt", type=str,
                        default="Low resolution satellite image with noise, blurring, and artifacts",
                        help="Source prompt describing input image")
    parser.add_argument("--tar_prompt", type=str,
                        default="High resolution clean satellite image with sharp details and natural colors",
                        help="Target prompt describing desired output")
    parser.add_argument("--gamma", type=float, default=0.7,
                        help="Gamma correction before FLUX (0=disable, default: 0.7)")
    parser.add_argument("--save_preview", action="store_true",
                        help="Save pre-FLUX preview PNG (after stretch+gamma)")
    parser.add_argument("--offload", action="store_true",
                        help="CPU offload for low-memory GPUs")
    parser.add_argument("--name", type=str, default="flux-dev",
                        help="FLUX model name (flux-dev or flux-schnell)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_stretch", action="store_true",
                        help="Skip percentile stretch and gamma (for 2-pass: input is already processed)")
    return parser.parse_args()


def read_geotiff(path, no_stretch=False):
    """Read GeoTIFF, return (numpy HxWx3 float32 0-1, profile, stretch_params)."""
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        nodata = src.nodata
        data = src.read()  # (C, H, W)

    data = data.transpose(1, 2, 0).astype(np.float64)  # (H, W, C)

    # Use first 3 bands only
    if data.shape[2] > 3:
        data = data[:, :, :3]

    if no_stretch:
        # For 2-pass: input is already processed uint8, just normalize to 0-1
        data_f = np.clip(data / 255.0, 0, 1).astype(np.float32) if data.max() > 1 else data.astype(np.float32)
        stretch_params = {"p2": None, "p98": None, "mask": None}
        print(f"[read_geotiff] no_stretch mode: simple normalization to 0-1")
        return data_f, profile, stretch_params

    # Mask nodata and all-zero pixels
    zero_mask = np.all(data == 0, axis=-1)
    if nodata is not None:
        nodata_mask = np.any(data == nodata, axis=-1)
        mask = nodata_mask | zero_mask
    else:
        mask = zero_mask if zero_mask.any() else None

    if mask is not None:
        data[mask] = 0

    # Percentile stretch to 0-1
    valid = data[~mask] if mask is not None else data.reshape(-1, 3)
    p2 = np.percentile(valid, 2, axis=0)
    p98 = np.percentile(valid, 98, axis=0)
    print(f"[read_geotiff] percentile stretch: p2={p2}, p98={p98}")

    stretch_params = {"p2": p2, "p98": p98, "mask": mask}

    for c in range(data.shape[2]):
        rng = p98[c] - p2[c]
        if rng < 1:
            rng = 1
        data[:, :, c] = (data[:, :, c] - p2[c]) / rng

    data_f = np.clip(data, 0, 1).astype(np.float32)

    # Fill nodata regions with nearest valid pixel
    if mask is not None and mask.any():
        from scipy.ndimage import distance_transform_edt
        _, nearest_idx = distance_transform_edt(mask, return_distances=True, return_indices=True)
        for ch in range(data_f.shape[2]):
            data_f[:, :, ch][mask] = data_f[:, :, ch][nearest_idx[0][mask], nearest_idx[1][mask]]
        print(f"[read_geotiff] filled {mask.sum()} nodata pixels with nearest valid pixels")
        stretch_params["mask"] = None

    return data_f, profile, stretch_params


def save_geotiff(path, data_float, profile, stretch_params):
    """Save float32 0-1 image back to GeoTIFF as uint8."""
    # Data is already 0-1 from the generation model, just scale to 0-255
    data_out = np.clip(data_float * 255, 0, 255).astype(np.uint8)

    # (H, W, C) -> (C, H, W)
    data_out = data_out.transpose(2, 0, 1)

    out_profile = profile.copy()
    out_profile["dtype"] = "uint8"
    out_profile["count"] = 3
    out_profile["nodata"] = None
    out_profile["compress"] = "lzw"

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data_out)


def compute_tiles(h, w, tile_size, overlap):
    """Compute tile positions with overlap."""
    step = tile_size - overlap
    tiles = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            tiles.append((y_start, x_start, y_end, x_end))
    tiles = list(set(tiles))
    tiles.sort()
    return tiles


def make_blend_weight(tile_h, tile_w, overlap):
    """Create smooth blending weights that taper at edges."""
    weight = np.ones((tile_h, tile_w), dtype=np.float32)
    if overlap <= 0:
        return weight

    ramp = np.linspace(0, 1, overlap)
    for i in range(min(overlap, tile_h)):
        weight[i, :] *= ramp[i]
        weight[tile_h - 1 - i, :] *= ramp[i]
    for i in range(min(overlap, tile_w)):
        weight[:, i] *= ramp[i]
        weight[:, tile_w - 1 - i] *= ramp[i]

    return weight


@torch.inference_mode()
def encode_image(img_np, ae, device):
    """Encode numpy HxWx3 float32 0-1 image to FLUX latent."""
    ae.to(device)
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() * 2 - 1  # to [-1, 1]
    img_t = img_t.unsqueeze(0).to(device)
    result = ae.encode(img_t).to(torch.bfloat16)
    ae.cpu()
    torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def decode_latent(latent, ae, device):
    """Decode FLUX latent to numpy HxWx3 float32 0-1 image."""
    ae.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        x = ae.decode(latent)
    ae.cpu()
    torch.cuda.empty_cache()
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    return ((x + 1.0) / 2.0).cpu().float().numpy()


@torch.inference_mode()
def refine_tile(tile_img, prompt_cache, model, ae, args):
    """Run RF-Solver inversion + denoising on a single tile."""
    h, w = tile_img.shape[:2]

    # Pad to multiple of 16
    h_pad = (16 - h % 16) % 16
    w_pad = (16 - w % 16) % 16
    if h_pad > 0 or w_pad > 0:
        tile_img = np.pad(tile_img, ((0, h_pad), (0, w_pad), (0, 0)), mode="reflect")

    model_device = getattr(model, '_primary_device', torch.device(args.device))
    ae_device = next(ae.parameters()).device
    ph, pw = tile_img.shape[:2]

    # Encode (AE on ae_device)
    init_image = encode_image(tile_img, ae, ae_device)

    # Pack image into latent sequence
    from einops import rearrange, repeat
    img = rearrange(init_image, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    bs = img.shape[0]
    h_lat, w_lat = init_image.shape[2] // 2, init_image.shape[3] // 2
    img_ids = torch.zeros(h_lat, w_lat, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h_lat)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w_lat)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    # Build inputs from pre-encoded prompts
    inp_src = {
        "img": img.to(model_device),
        "img_ids": img_ids.to(model_device),
        "txt": prompt_cache["src_txt"].to(model_device),
        "txt_ids": prompt_cache["src_txt_ids"].to(model_device),
        "vec": prompt_cache["src_vec"].to(model_device),
    }
    inp_tar = {
        "img": img.to(model_device),  # will be replaced by z
        "img_ids": img_ids.to(model_device),
        "txt": prompt_cache["tar_txt"].to(model_device),
        "txt_ids": prompt_cache["tar_txt_ids"].to(model_device),
        "vec": prompt_cache["tar_vec"].to(model_device),
    }
    timesteps = get_schedule(args.num_steps, inp_src["img"].shape[1], shift=(args.name != "flux-schnell"))

    # Setup feature sharing info
    info = {
        "feature_path": "feature_tmp",
        "feature": {},
        "inject_step": args.inject,
    }
    os.makedirs("feature_tmp", exist_ok=True)

    # RF-Solver Inversion (image -> noise)
    z, info = denoise(model, **inp_src, timesteps=timesteps, guidance=1, inverse=True, info=info)
    torch.cuda.empty_cache()

    # RF-Solver Denoising (noise -> refined image, with feature sharing)
    inp_tar["img"] = z
    timesteps_denoise = get_schedule(args.num_steps, inp_tar["img"].shape[1], shift=(args.name != "flux-schnell"))
    x, _ = denoise(model, **inp_tar, timesteps=timesteps_denoise, guidance=args.guidance, inverse=False, info=info)

    # Decode (move back to AE device)
    batch_x = unpack(x.float(), ph, pw).to(ae_device)
    result = decode_latent(batch_x, ae, ae_device)

    # Remove padding
    if h_pad > 0 or w_pad > 0:
        result = result[:h, :w]

    return result


def main():
    args = parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_rfsolver{ext}"

    assert args.tile_size % 16 == 0, "tile_size must be divisible by 16"

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")

    # Read GeoTIFF
    print("Reading GeoTIFF...")
    img, profile, stretch_params = read_geotiff(args.input, no_stretch=args.no_stretch)
    h, w, c = img.shape
    print(f"Image size: {w}x{h}, {c} bands")

    # Gamma correction (skip for 2-pass)
    gamma = args.gamma if not args.no_stretch else 1.0
    if gamma > 0 and gamma != 1.0:
        print(f"Applying gamma correction: {gamma}")
        img = np.power(np.clip(img, 0, 1), gamma)

    # Save preview
    if args.save_preview:
        output_dir = os.path.dirname(args.output) or "."
        input_stem = os.path.splitext(os.path.basename(args.input))[0]
        preview_path = os.path.join(output_dir, input_stem + "_rfsolver_input.png")
        preview = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(preview).save(preview_path)
        print(f"Saved preview: {preview_path}")

    # Load FLUX model
    print("Loading FLUX model...")
    num_gpus = torch.cuda.device_count()

    # Encode prompts with T5/CLIP on CPU to save GPU memory
    print("Encoding prompts with T5/CLIP (on CPU)...")
    t5 = load_t5("cpu", max_length=512 if args.name == "flux-dev" else 256)
    clip = load_clip("cpu")

    # Pre-encode prompts once (they're the same for every tile)
    dummy_img = torch.zeros(1, 16, 16, 16)  # dummy latent for prepare()
    src_tokens = prepare(t5, clip, dummy_img, prompt=args.src_prompt)
    tar_tokens = prepare(t5, clip, dummy_img, prompt=args.tar_prompt)
    # Keep only txt, txt_ids, vec (not img)
    prompt_cache = {
        "src_txt": src_tokens["txt"],
        "src_txt_ids": src_tokens["txt_ids"],
        "src_vec": src_tokens["vec"],
        "tar_txt": tar_tokens["txt"],
        "tar_txt_ids": tar_tokens["txt_ids"],
        "tar_vec": tar_tokens["vec"],
    }
    del t5, clip
    print("Prompts encoded.")

    # Load AE (move to GPU only during encode/decode to save VRAM)
    ae = load_ae(args.name, device="cpu")

    # Load and distribute FLUX model
    model = load_flow_model(args.name, device="cpu")
    print(f"Distributing model across {num_gpus} GPUs...")
    model = distribute_model(model, num_gpus)
    print("FLUX model loaded.")

    # Compute tiles
    tiles = compute_tiles(h, w, args.tile_size, args.overlap)
    print(f"Processing {len(tiles)} tiles (tile_size={args.tile_size}, overlap={args.overlap})")
    print(f"RF-Solver params: num_steps={args.num_steps}, inject={args.inject}, guidance={args.guidance}")

    # Process tiles
    output = np.zeros((h, w, 3), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    for y_start, x_start, y_end, x_end in tqdm(tiles, desc="Refining tiles"):
        tile = img[y_start:y_end, x_start:x_end].copy()
        tile_h, tile_w = tile.shape[:2]

        refined = refine_tile(tile, prompt_cache, model, ae, args)

        blend_w = make_blend_weight(tile_h, tile_w, args.overlap)
        for ch in range(3):
            output[y_start:y_end, x_start:x_end, ch] += refined[:, :, ch] * blend_w
        weight_sum[y_start:y_end, x_start:x_end] += blend_w

    # Normalize by weight
    for ch in range(3):
        output[:, :, ch] /= np.maximum(weight_sum, 1e-8)

    # Reverse gamma
    if gamma > 0 and gamma != 1.0:
        output = np.power(np.clip(output, 0, 1), 1.0 / gamma)

    # Save output
    print(f"Saving output: {args.output}")
    save_geotiff(args.output, output, profile, stretch_params)
    print("Done!")


if __name__ == "__main__":
    main()
