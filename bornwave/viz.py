"""Visualization: shot gathers and wavefield animations.

Torch-free (numpy + matplotlib only); matplotlib is imported lazily so the
core solver can be used without it. MP4 output uses the ffmpeg writer and
falls back to an animated GIF if ffmpeg is unavailable.
"""
from __future__ import annotations

import warnings

import numpy as np

__all__ = ["trace_norm", "plot_shot", "plot_wavefield_video"]


def trace_norm(data, axis: int = -2) -> np.ndarray:
    """Normalize each trace by its own max |amplitude|.

    `axis` is the TIME axis: -2 works for both (nt, nrec) single-shot and
    (nshot, nt, nrec) multi-shot records. Zero traces are left untouched.
    """
    d = np.asarray(data, dtype=np.float64)
    m = np.max(np.abs(d), axis=axis, keepdims=True)
    m = np.where(m == 0.0, 1.0, m)
    return d / m


def _lazy_plt():
    import matplotlib
    import matplotlib.pyplot as plt  # noqa: F401
    return matplotlib, plt


def plot_shot(
    data,
    fname: str | None = None,
    *,
    dt: float | None = None,
    offsets=None,
    title: str | None = None,
    cmap: str = "gray",
    perc: float = 98.0,
    figsize=(6.4, 8.0),
    dpi: int = 150,
):
    """Variable-density plot of a shot record `data` of shape (nt, nrec).

    dt      : sample interval — y axis in seconds instead of samples.
    offsets : per-trace horizontal coordinate (e.g. offset in m) — x axis.
    Returns fname if saving, else (fig, ax) for further tweaking.
    """
    _, plt = _lazy_plt()
    d = np.asarray(data, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError("plot_shot expects (nt, nrec); for multi-shot "
                         "records pass res.seis_p[b]")
    nt, nrec = d.shape

    x0, x1 = (float(offsets[0]), float(offsets[-1])) if offsets is not None \
        else (0.0, float(nrec - 1))
    t1 = (nt - 1) * dt if dt is not None else float(nt - 1)
    v = np.percentile(np.abs(d), perc)
    v = v if v > 0 else 1.0

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(d, aspect="auto", cmap=cmap, vmin=-v, vmax=v,
              extent=[x0, x1, t1, 0.0], interpolation="bilinear")
    ax.set_xlabel("Offset (m)" if offsets is not None else "Trace #")
    ax.set_ylabel("Time (s)" if dt is not None else "Sample #")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=dpi)
        plt.close(fig)
        return fname
    return fig, ax


def plot_wavefield_video(
    snaps,
    fname: str,
    *,
    fps: int = 10,
    dh: float | None = None,
    snap_times=None,
    adaptive_clims: bool = True,
    perc: float = 99.5,
    cmap: str = "seismic",
    model=None,
    model_cmap: str = "gray",
    wave_alpha: float | None = None,
    title: str | None = None,
    dpi: int = 110,
):
    """Animate pressure snapshots `snaps` of shape (nsnap, nz, nx) to
    MP4 (ffmpeg) or GIF (fallback / .gif extension).

    adaptive_clims : rescale color limits per frame (early frames are weak
        after geometric spreading — this keeps late reflections visible).
    model : optional (nz, nx) background (e.g. vp) drawn in grayscale under
        the semi-transparent wavefield.
    Returns the written filename.
    """
    _, plt = _lazy_plt()
    from matplotlib import animation

    s = np.asarray(snaps, dtype=np.float64)
    if s.ndim == 4 and s.shape[0] == 1:
        s = s[0]
    if s.ndim != 3:
        raise ValueError("snaps must be (nsnap, nz, nx); for multi-shot "
                         "results pass res.snaps[b]")
    nsnap, nz, nx = s.shape

    extent = [0.0, nx * dh, nz * dh, 0.0] if dh is not None else None
    vmax_g = np.percentile(np.abs(s), perc)
    vmax_g = vmax_g if vmax_g > 0 else 1.0

    fig, ax = plt.subplots(figsize=(7.2, 7.2 * nz / max(nx, 1) + 0.6))
    if model is not None:
        ax.imshow(np.asarray(model, dtype=np.float64), cmap=model_cmap,
                  extent=extent, interpolation="bilinear")
        alpha = 0.65 if wave_alpha is None else wave_alpha
    else:
        alpha = 1.0 if wave_alpha is None else wave_alpha

    im = ax.imshow(s[0], cmap=cmap, vmin=-vmax_g, vmax=vmax_g,
                   extent=extent, alpha=alpha, interpolation="bilinear")
    ax.set_xlabel("x (m)" if dh is not None else "x (cells)")
    ax.set_ylabel("z (m)" if dh is not None else "z (cells)")
    if title:
        ax.set_title(title)
    label = ax.text(0.02, 0.04, "", transform=ax.transAxes,
                    color="k", fontsize=10,
                    bbox=dict(fc="w", alpha=0.6, ec="none"))
    fig.tight_layout()

    def update(i):
        im.set_data(s[i])
        if adaptive_clims:
            v = np.percentile(np.abs(s[i]), perc)
            v = max(v, vmax_g * 1e-3)
            im.set_clim(-v, v)
        if snap_times is not None:
            label.set_text(f"t = {snap_times[i] * 1e3:.0f} ms")
        else:
            label.set_text(f"frame {i}")
        return im, label

    ani = animation.FuncAnimation(fig, update, frames=nsnap, blit=False)

    out = str(fname)
    if out.lower().endswith(".mp4"):
        if animation.writers.is_available("ffmpeg"):
            writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
        else:
            warnings.warn("ffmpeg not found — writing GIF instead")
            out = out[:-4] + ".gif"
            writer = animation.PillowWriter(fps=fps)
    elif out.lower().endswith(".gif"):
        writer = animation.PillowWriter(fps=fps)
    else:
        raise ValueError("fname must end with .mp4 or .gif")

    ani.save(out, writer=writer, dpi=dpi)
    plt.close(fig)
    return out
