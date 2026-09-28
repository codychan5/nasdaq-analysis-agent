from pathlib import Path


def render_chart(dates: list[str], closes: list[float], path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 2.6), dpi=150)
    ax.plot(dates, closes, marker="o", linewidth=2)
    ax.set_title("Adjusted close, last six sessions")
    ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
