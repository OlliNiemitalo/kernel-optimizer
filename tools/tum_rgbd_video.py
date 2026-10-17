"""Converts a TUM RGB-D sequence into the blur demo's RGB-D video format.

Each output frame has the color image on top and the depth below, as grayscale inverse depth
v = 0.5 m / z (white = 0.5 m, black = infinitely far), H.264 in MP4.

Usage:
    python tools/tum_rgbd_video.py SEQUENCE_DIR OUT.mp4 [--start SECONDS] [--seconds SECONDS]

SEQUENCE_DIR is an extracted TUM RGB-D sequence (rgb.txt, depth.txt, rgb/, depth/), e.g.
https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_static.tgz
Needs numpy, Pillow and imageio-ffmpeg.
"""

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

DEPTH_SCALE = 5000.0  # TUM depth PNG units per meter
WHITE_DEPTH = 0.5  # meters at v = 1
FPS = 30


def read_list(path):
    entries = []
    for line in Path(path).read_text().splitlines():
        if line and not line.startswith("#"):
            stamp, name = line.split()
            entries.append((float(stamp), name))
    return entries


def fill_holes(depth):
    """Fills missing depth (0) with the farther of the nearest valid values to the left and
    right: holes in sensor depth are mostly occlusion shadows on the background side of edges."""
    h, w = depth.shape
    valid = depth > 0
    idx = np.arange(w)[None, :].repeat(h, 0)
    left = np.where(valid, idx, -1)
    np.maximum.accumulate(left, axis=1, out=left)
    right = np.where(valid, idx, w)
    right = np.minimum.accumulate(right[:, ::-1], axis=1)[:, ::-1]
    rows = np.arange(h)[:, None]
    from_left = np.where(left >= 0, depth[rows, np.clip(left, 0, w - 1)], 0)
    from_right = np.where(right < w, depth[rows, np.clip(right, 0, w - 1)], 0)
    filled = np.where(valid, depth, np.maximum(from_left, from_right))
    # Rows with no valid depth at all: take the column's farthest value.
    empty = filled == 0
    if empty.any():
        filled[empty] = np.broadcast_to(filled.max(axis=0, keepdims=True), filled.shape)[empty]
    return filled


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sequence", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--crf", type=int, default=20)
    args = parser.parse_args()

    rgb = read_list(args.sequence / "rgb.txt")
    depth = read_list(args.sequence / "depth.txt")
    depth_stamps = np.array([s for s, _ in depth])
    t0 = rgb[0][0] + args.start
    frames = [(s, n) for s, n in rgb if t0 <= s < t0 + args.seconds][: int(args.seconds * FPS)]

    first = Image.open(args.sequence / frames[0][1])
    w, h = first.size
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{2 * h}", "-r", str(FPS), "-i", "-",
        "-vf", "scale=out_color_matrix=bt709:out_range=tv",
        "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
        # sRGB transfer tag, so browsers decode without a gamma conversion that would alter depth.
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "iec61966-2-1", "-color_range", "tv",
        "-movflags", "+faststart", str(args.out),
    ]
    encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for stamp, name in frames:
        color = np.asarray(Image.open(args.sequence / name).convert("RGB"))
        k = int(np.argmin(np.abs(depth_stamps - stamp)))
        z = fill_holes(np.asarray(Image.open(args.sequence / depth[k][1]), dtype=np.float32) / DEPTH_SCALE)
        v = np.clip(np.round(255.0 * WHITE_DEPTH / np.maximum(z, 1e-3)), 0, 255).astype(np.uint8)
        encoder.stdin.write(np.concatenate([color, np.repeat(v[:, :, None], 3, axis=2)], axis=0).tobytes())
    encoder.stdin.close()
    encoder.wait()
    print(f"{len(frames)} frames, {w}x{2 * h}, {args.out.stat().st_size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
