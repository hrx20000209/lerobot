#!/usr/bin/env python
"""Stitch the world model's predicted future frames into a video.

The model emits one predicted future frame per camera per inference, not a
sequence, so this is not a video the model generated -- it is the succession of
its one-step predictions over the run, which is the only way to watch what it
was anticipating.

Each output frame is a 2x3 panel: the three predicted views on top, and the
real camera frames closest in time underneath, so prediction and reality can be
compared as it plays.
"""

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PRED_RE = re.compile(r"t(\d+)_(\d+\.\d+)_(\w+)_pred\.jpg")
ACT_RE = re.compile(r"t(\d+)_(\d+\.\d+)_(\w+)\.jpg")
CAMS = ("front", "right", "wrist")
TILE = (320, 320)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--future-dir", required=True)
    ap.add_argument("--run-dir", required=True, help="For the real frames to compare against")
    ap.add_argument("--out", required=True, help="Output .mp4 (a .gif is written alongside)")
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--horizon", type=float, default=0.0,
                    help="Pair each prediction with the real frame this many seconds later")
    args = ap.parse_args()

    fut = Path(args.future_dir)
    shot = next(Path(args.run_dir).glob("screenshots_*"))

    preds = {}
    for p in fut.glob("*_pred.jpg"):
        m = PRED_RE.fullmatch(p.name)
        if m:
            preds.setdefault(float(m.group(2)), {})[m.group(3)] = p
    times = sorted(t for t, v in preds.items() if all(c in v for c in CAMS))
    if not times:
        raise SystemExit("no prediction triples found")

    actual = {c: [] for c in CAMS}
    for p in shot.glob("*.jpg"):
        m = ACT_RE.fullmatch(p.name)
        if m and m.group(3) in actual:
            actual[m.group(3)].append((float(m.group(2)), p))
    for c in actual:
        actual[c].sort()
    act_t = {c: np.array([t for t, _ in v]) for c, v in actual.items()}

    t0 = times[0]
    out_dir = Path(args.out).parent / "_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.png"):
        f.unlink()

    W, H = TILE
    for i, t in enumerate(times):
        canvas = Image.new("RGB", (W * 3, H * 2 + 46), "white")
        d = ImageDraw.Draw(canvas)
        d.text((8, 6), f"Cosmos world model — predicted future vs actual    t = {t - t0:6.1f} s", fill="black")
        for k, cam in enumerate(CAMS):
            canvas.paste(Image.open(preds[t][cam]).resize((W, H)), (k * W, 24))
            if len(act_t[cam]):
                j = int(np.argmin(np.abs(act_t[cam] - (t + args.horizon))))
                canvas.paste(Image.open(actual[cam][j][1]).resize((W, H)), (k * W, H + 30))
            d.text((k * W + 8, 26), f"pred {cam}", fill="yellow")
            d.text((k * W + 8, H + 32), f"real {cam}", fill="yellow")
        canvas.save(out_dir / f"f{i:05d}.png")

    out = Path(args.out)
    print(f"{len(times)} frames")
    # ffmpeg is not installed here; fall back to imageio's bundled encoder, and
    # to the GIF below if neither is available.
    try:
        cmd = ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", str(out_dir / "f%05d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"video -> {out}")
        else:
            raise RuntimeError(r.stderr[-300:])
    except (FileNotFoundError, RuntimeError) as exc:
        try:
            import imageio.v2 as imageio

            with imageio.get_writer(out, fps=args.fps, macro_block_size=1) as w:
                for p in sorted(out_dir.glob("*.png")):
                    w.append_data(imageio.imread(p))
            print(f"video -> {out}  (imageio)")
        except Exception as exc2:  # noqa: BLE001
            print(f"no mp4 encoder available ({type(exc).__name__} / {type(exc2).__name__});"
                  f" using the GIF below. PNG frames kept in {out_dir}")

    gif = out.with_suffix(".gif")
    frames = [Image.open(p) for p in sorted(out_dir.glob("*.png"))]
    if frames:
        step = max(1, len(frames) // 120)  # keep the gif manageable
        frames[0].save(gif, save_all=True, append_images=frames[::step][1:],
                       duration=int(1000 / args.fps), loop=0, optimize=True)
        print(f"gif   -> {gif}  ({len(frames[::step])} frames)")


if __name__ == "__main__":
    main()
