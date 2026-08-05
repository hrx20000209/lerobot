#!/usr/bin/env python
"""Does the world model's predicted future frame match the observation that arrived?

For every predicted frame the server saved at wall time W, this compares it
against the real camera frame at W + h for a sweep of horizons h, and against
two baselines:

  persistence  the frame at W itself -- "predict nothing changes". A world model
               only earns its keep by beating this.
  chance       a randomly chosen frame from the run, i.e. how much similarity
               comes from the scene being mostly a static desk.

The horizon that maximises similarity tells us what the model is actually
predicting, which is not documented anywhere and has to be recovered empirically.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from skimage.metrics import structural_similarity as ssim  # noqa: E402

for _f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
    if Path(_f).exists():
        try:
            fm.fontManager.addfont(_f)
        except Exception:  # noqa: BLE001
            pass
_have = {f.name for f in fm.fontManager.ttflist}
for _c in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"):
    if _c in _have:
        plt.rcParams["font.sans-serif"] = [_c, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

PRED_RE = re.compile(r"t(\d+)_(\d+\.\d+)_(\w+)_pred\.jpg")
ACT_RE = re.compile(r"t(\d+)_(\d+\.\d+)_(\w+)\.jpg")
SIZE = (224, 224)


def load(path):
    return np.asarray(Image.open(path).convert("RGB").resize(SIZE, Image.BILINEAR), dtype=np.float64) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--future-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--cameras", default="front,right,wrist")
    ap.add_argument("--max-preds", type=int, default=60, help="Subsample predictions to keep this tractable")
    args = ap.parse_args()

    fut_dir = Path(args.future_dir)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir or run_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    shot_dir = next(run_dir.glob("screenshots_*"))
    cameras = args.cameras.split(",")

    # index the real frames by camera and wall time
    actual = {c: [] for c in cameras}
    for p in shot_dir.glob("*.jpg"):
        m = ACT_RE.fullmatch(p.name)
        if m and m.group(3) in actual:
            actual[m.group(3)].append((float(m.group(2)), p))
    for c in actual:
        actual[c].sort()
    print({c: len(v) for c, v in actual.items()})

    horizons = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    results = {c: {h: [] for h in horizons} for c in cameras}
    persistence = {c: {h: [] for h in horizons} for c in cameras}
    chance = {c: [] for c in cameras}
    rng = np.random.default_rng(0)

    for cam in cameras:
        preds = []
        for p in fut_dir.glob(f"*_{cam}_pred.jpg"):
            m = PRED_RE.fullmatch(p.name)
            if m:
                preds.append((float(m.group(2)), p))
        preds.sort()
        step = max(1, len(preds) // args.max_preds)
        preds = preds[::step]
        times = np.array([t for t, _ in actual[cam]])
        if len(times) == 0 or not preds:
            continue
        print(f"{cam}: {len(preds)} 预测帧参与比较")

        for w, pp in preds:
            pred_img = load(pp)
            # frame at prediction time = the persistence baseline
            i0 = int(np.argmin(np.abs(times - w)))
            base_img = load(actual[cam][i0][1])
            for h in horizons:
                target = w + h
                if target > times[-1]:
                    continue
                i = int(np.argmin(np.abs(times - target)))
                if abs(times[i] - target) > 0.6:  # no real frame near that horizon
                    continue
                act_img = load(actual[cam][i][1])
                results[cam][h].append(ssim(pred_img, act_img, channel_axis=2, data_range=1.0))
                persistence[cam][h].append(ssim(base_img, act_img, channel_axis=2, data_range=1.0))
            j = int(rng.integers(0, len(actual[cam])))
            chance[cam].append(ssim(pred_img, load(actual[cam][j][1]), channel_axis=2, data_range=1.0))

    summary = {}
    print(f"\n{'camera':<8}{'horizon':>9}{'SSIM(预测)':>12}{'SSIM(持恒)':>12}{'差值':>9}{'n':>6}")
    for cam in cameras:
        best = None
        for h in horizons:
            v, b = results[cam][h], persistence[cam][h]
            if not v:
                continue
            mv, mb = float(np.mean(v)), float(np.mean(b))
            print(f"{cam:<8}{h:>9.2f}{mv:>12.4f}{mb:>12.4f}{mv - mb:>+9.4f}{len(v):>6}")
            if best is None or mv > best[1]:
                best = (h, mv, mb)
        if best:
            summary[cam] = {"best_horizon_s": best[0], "ssim_pred": best[1],
                            "ssim_persistence": best[2],
                            "ssim_chance": float(np.mean(chance[cam])) if chance[cam] else None}
            print(f"{cam:<8}  -> 最佳 horizon {best[0]:.2f}s, SSIM {best[1]:.4f}, "
                  f"持恒 {best[2]:.4f}, 随机 {summary[cam]['ssim_chance']:.4f}")

    # ---- figure ----
    fig, axes = plt.subplots(1, len(cameras), figsize=(5 * len(cameras), 4), squeeze=False)
    for k, cam in enumerate(cameras):
        ax = axes[0][k]
        hs = [h for h in horizons if results[cam][h]]
        if not hs:
            continue
        ax.plot(hs, [np.mean(results[cam][h]) for h in hs], "o-", label="世界模型预测", color="tab:blue")
        ax.plot(hs, [np.mean(persistence[cam][h]) for h in hs], "s--", label="持恒基线(当前帧)", color="tab:orange")
        if chance[cam]:
            ax.axhline(np.mean(chance[cam]), color="tab:gray", ls=":", label="随机帧基线")
        ax.set_title(f"{cam}")
        ax.set_xlabel("预测视界 h (秒)")
        ax.set_ylabel("SSIM vs 真实观测")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("future state 预测 与 真实观测 的相似度（越高越准）", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "future_state_similarity.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- side-by-side sample ----
    cam = cameras[0]
    preds = sorted((float(PRED_RE.fullmatch(p.name).group(2)), p)
                   for p in fut_dir.glob(f"*_{cam}_pred.jpg") if PRED_RE.fullmatch(p.name))
    if preds and summary.get(cam):
        h = summary[cam]["best_horizon_s"]
        times = np.array([t for t, _ in actual[cam]])
        picks = preds[len(preds) // 5 :: max(1, len(preds) // 4)][:4]
        fig, axes = plt.subplots(2, len(picks), figsize=(3.2 * len(picks), 6.6))
        for j, (w, pp) in enumerate(picks):
            i = int(np.argmin(np.abs(times - (w + h))))
            axes[0][j].imshow(Image.open(pp).resize(SIZE))
            axes[0][j].set_title(f"预测 t={w - preds[0][0]:.0f}s", fontsize=9)
            axes[1][j].imshow(Image.open(actual[cam][i][1]).resize(SIZE))
            axes[1][j].set_title(f"真实 t+{h:.1f}s", fontsize=9)
            for r in (0, 1):
                axes[r][j].axis("off")
        fig.suptitle(f"{cam}: 世界模型预测 vs 真实观测 (horizon={h:.2f}s)")
        fig.tight_layout()
        fig.savefig(out_dir / "future_state_examples.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    (out_dir / "future_state_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    main()
