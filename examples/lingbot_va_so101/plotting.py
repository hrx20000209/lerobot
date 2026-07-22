import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ACTION_NAMES


def plot_action_compare(pred_raw, gt_raw, out_path, title=""):
    """One subplot per action dim: x=timestep, y=action value, pred vs gt lines."""
    n_dims = pred_raw.shape[1]
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 2.2 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]
    t = range(pred_raw.shape[0])
    for i, ax in enumerate(axes):
        ax.plot(t, gt_raw[:, i].numpy(), label="ground truth", color="tab:blue", linewidth=1.5)
        ax.plot(t, pred_raw[:, i].numpy(), label="predicted", color="tab:orange", linewidth=1.2, linestyle="--")
        ax.set_ylabel(ACTION_NAMES[i] if i < len(ACTION_NAMES) else f"dim{i}")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("timestep")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
