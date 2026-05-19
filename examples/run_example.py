from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spatial_reg_ward import SpatialRegWard, make_half_moon_toy_data


def _fit_line(x: np.ndarray, y: np.ndarray):
    coef = np.polyfit(x, y, deg=1)
    return coef[1], coef[0]  # intercept, slope


def main() -> None:
    data = make_half_moon_toy_data(n=300)

    model = SpatialRegWard(
        data["x"],
        data["y"],
        n_clusters=3,
        coords=data["coords"],
        use_cluster_nn=True,
        min_cluster_size=10,
        pbar=False,
    )
    labels = model.fit()

    coords = data["coords"]
    v = data["v"]
    r = data["y"]
    regimes_true = data["regime_true"]

    regime_colors = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]
    v_grid = np.linspace(v.min(), v.max(), 200)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), dpi=150)
    ax_true_space, ax_est_space = axes[0]
    ax_true_reg, ax_est_reg = axes[1]

    # (A) True regimes in space
    for idx, c in enumerate(sorted(np.unique(regimes_true))):
        color = regime_colors[idx % len(regime_colors)]
        mask = regimes_true == c
        ax_true_space.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
            color=color,
            label=f"Regime {c}",
        )
    ax_true_space.set_title("(A) True regimes (space)")
    ax_true_space.set_xlabel("x")
    ax_true_space.set_ylabel("y")
    ax_true_space.set_aspect("equal", "box")
    ax_true_space.legend(frameon=True, fontsize=8)

    # (B) Estimated clusters in space
    for idx, c in enumerate(sorted(np.unique(labels))):
        color = regime_colors[idx % len(regime_colors)]
        mask = (labels == c)
        ax_est_space.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
            color=color,
            label=f"Cluster {c + 1}",
        )
    ax_est_space.set_title("(B) Estimated clusters (space)")
    ax_est_space.set_xlabel("x")
    ax_est_space.set_ylabel("y")
    ax_est_space.set_aspect("equal", "box")
    ax_est_space.legend(frameon=True, fontsize=8)

    # (C) True regimes regression
    for idx, c in enumerate(sorted(np.unique(regimes_true))):
        color = regime_colors[idx % len(regime_colors)]
        mask = regimes_true == c
        ax_true_reg.scatter(
            v[mask],
            r[mask],
            s=14,
            alpha=0.6,
            edgecolor="white",
            linewidth=0.4,
            color=color,
        )
        b0, b1 = _fit_line(v[mask], r[mask])
        ax_true_reg.plot(
            v_grid,
            b0 + b1 * v_grid,
            color=color,
            linewidth=1.5,
            label=f"Regime {c}",
        )
    ax_true_reg.set_title("(C) True regime regressions")
    ax_true_reg.set_xlabel("v")
    ax_true_reg.set_ylabel("r")
    ax_true_reg.legend(frameon=True, fontsize=8)

    # (D) Estimated regime regression
    for idx, c in enumerate(sorted(np.unique(labels))):
        color = regime_colors[idx % len(regime_colors)]
        mask = labels == c
        ax_est_reg.scatter(
            v[mask],
            r[mask],
            s=14,
            alpha=0.6,
            edgecolor="white",
            linewidth=0.4,
            color=color,
        )
        b0, b1 = _fit_line(v[mask], r[mask])
        ax_est_reg.plot(
            v_grid,
            b0 + b1 * v_grid,
            color=color,
            linewidth=1.5,
            label=f"Cluster {c + 1}",
        )
    ax_est_reg.set_title("(D) Estimated regime regressions")
    ax_est_reg.set_xlabel("v")
    ax_est_reg.set_ylabel("r")
    ax_est_reg.legend(frameon=True, fontsize=8)

    fig.tight_layout()

    out_path = Path("half_moon_clusters.png")
    fig.savefig("half_moon_clusters.png", bbox_inches="tight")
    print(f"Saved plot to: {out_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
