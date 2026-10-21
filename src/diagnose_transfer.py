"""Fast-to-Gazebo transfer-gap diagnosis (TRANSFER_GAP_TEST.md). Reads the per-step recordings written by
`evaluate.py --record-steps`; `replay` needs a sim for --env gz.

    python diagnose_transfer.py analyse [--a fast --b gz]          # tables + plots -> results/transfer_diag/
    python diagnose_transfer.py analyse --b fast_noise2            # control: fast vs fast with other LiDAR noise
    python diagnose_transfer.py replay --src fast --env gz         # open-loop counterfactual (step 6)
    python diagnose_transfer.py replay --src gz --env fast
    python diagnose_transfer.py drift --set gz_stock               # odom vs scan-matched pose along each episode

A recording "set" is a directory under results/transfer_diag holding <run>__on_<tier>_steps.npz: `fast` and `gz`
are results/transfer_diag itself (tier fast / gz); any other name is a sub-directory (e.g. fast_noise2, which
holds a fast-tier recording made with a different noise seed).
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from envs.layout import Layout, eval_layouts, surface_dist
from utils import apply_overrides, load_config, results_dir

RUNS = ["fast_dqn_seed0", "fast_dqn_seed1", "fast_dqn_seed2"]
THRESH = (0.05, 0.10, 0.20)


# ---- loading ------------------------------------------------------------------------------
def set_path(diag: Path, name: str) -> tuple[Path, str]:
    """(directory, tier) of a recording set."""
    if name in ("fast", "gz"):
        return diag, name
    d = diag / name
    tiers = {Path(f).name.split("__on_")[1].split("_")[0] for f in glob.glob(str(d / "*_steps.npz"))}
    if len(tiers) != 1:
        raise SystemExit(f"cannot tell the tier of recording set {name!r} in {d}")
    return d, tiers.pop()


def load_steps(path: Path) -> list[dict[str, np.ndarray]]:
    with np.load(path) as z:
        n = 1 + max(int(k.split("_")[0][2:]) for k in z.files)
        return [{k.split("_", 1)[1]: z[k] for k in z.files if k.startswith(f"ep{i}_")} for i in range(n)]


# ---- geometry -----------------------------------------------------------------------------
def nearest_object(layout: Layout, x: float, y: float, cfg: dict[str, Any]) -> tuple[float, str]:
    """Clearance from point (x, y) to the nearest obstacle surface or wall inner face, and its name."""
    half = cfg["arena"]["size"] / 2.0
    best = [(half - x, "wall_e"), (half + x, "wall_w"), (half - y, "wall_n"), (half + y, "wall_s")]
    best += [(surface_dist(o, x, y, cfg), f"{o.kind}_{o.slot}") for o in layout.obstacles]
    return min(best)


def lidar_origin(pose: np.ndarray, cfg: dict[str, Any]) -> tuple[float, float]:
    off = cfg["robot"]["lidar_offset_x"]
    return pose[0] + off * np.cos(pose[2]), pose[1] + off * np.sin(pose[2])


def q_margin(q: np.ndarray) -> np.ndarray:
    s = np.sort(q, axis=-1)
    return s[..., -1] - s[..., -2]


# ---- analysis -----------------------------------------------------------------------------
def compare_pair(ea: dict, eb: dict, layout: Layout, cfg: dict, render: Any) -> dict[str, Any]:
    """Per-(checkpoint, layout) comparison of rollouts a and b."""
    pa, pb = ea["pose"], eb["pose"]
    n = min(len(pa), len(pb))
    dxy = np.hypot(pa[:n, 0] - pb[:n, 0], pa[:n, 1] - pb[:n, 1])
    row: dict[str, Any] = {"out_a": str(ea["outcome"]), "out_b": str(eb["outcome"]),
                           "steps_a": len(ea["action"]), "steps_b": len(eb["action"])}
    for t in THRESH:
        k = np.nonzero(dxy > t)[0]
        row[f"div{int(t * 100):02d}"] = int(k[0]) if k.size else -1
        if k.size:
            ox, oy = lidar_origin(pa[k[0]], cfg)
            row[f"div{int(t * 100):02d}_clear"] = nearest_object(layout, ox, oy, cfg)[0]
    na = min(len(ea["action"]), len(eb["action"]))
    dis = np.nonzero(ea["action"][:na] != eb["action"][:na])[0]
    row["first_dis"] = int(dis[0]) if dis.size else -1
    if dis.size:
        k = int(dis[0])
        row["margin_a"] = float(q_margin(ea["q"][k])) if len(ea["q"]) else np.nan
        row["margin_b"] = float(q_margin(eb["q"][k])) if len(eb["q"]) else np.nan
        row["dxy_at_dis"] = float(dxy[k]) if k < n else np.nan
        d = np.abs(ea["obs"][k] - eb["obs"][k])
        row["d_lidar_mean"], row["d_lidar_max"] = float(d[:24].mean()), float(d[:24].max())
        row["d_dist"], row["d_heading"], row["d_v"], row["d_w"] = map(float, d[24:28])
        ox, oy = lidar_origin(pa[k], cfg)
        row["dis_clear"] = nearest_object(layout, ox, oy, cfg)[0]
    # sensor consistency of rollout b: its LiDAR vs the fast-tier LiDAR rendered (noise-free) at b's own pose
    if render is not None:
        clip = cfg["observation"]["lidar_clip"]
        errs = [np.abs(eb["obs"][k][:24] * clip - np.minimum(render(layout, eb["pose"][k]), clip))
                for k in range(len(eb["pose"]))]
        e = np.stack(errs)
        row["b_sensor_err_mean"], row["b_sensor_err_p99"] = float(e.mean()), float(np.percentile(e, 99))
        row["b_sensor_err_final_min"] = float(eb["obs"][-1][:24].min() * clip - np.minimum(render(layout, eb["pose"][-1]), clip).min())
    if row["out_b"] == "collision":
        pf = pb[-1]
        ox, oy = lidar_origin(pf, cfg)
        clear, obj = nearest_object(layout, ox, oy, cfg)
        row["b_coll_obj"], row["b_coll_clear_geom"] = obj, clear
        row["b_coll_min_range"] = float(eb["min_range"][-1])
        row["b_coll_goal_dist"] = float(np.hypot(pf[0] - layout.goal[0], pf[1] - layout.goal[1]))
        bins = eb["obs"][-1][:24]
        row["b_coll_bin"] = int(np.argmin(bins))       # 0 = straight ahead, CCW, 15 deg per bin
    return row


def make_renderer(cfg: dict) -> Any:
    """Noise-free fast-tier LiDAR at an arbitrary pose (metres per bin)."""
    from envs.env_fast import FastEnv
    c = load_config(tier="fast")
    c["lidar"]["noise_std"] = 0.0
    env = FastEnv(c)
    env.reset(seed=0)
    cache: dict[int, Layout] = {}

    def render(layout: Layout, pose: np.ndarray) -> np.ndarray:
        if cache.get(0) is not layout:
            env._reset_world(layout)
            cache[0] = layout
        env._state = np.asarray(pose, dtype=np.float64).copy()
        return env._lidar()
    return render


def analyse(args: argparse.Namespace) -> None:
    cfg = load_config(tier="fast")
    layouts = eval_layouts(cfg)
    diag = results_dir(cfg) / "transfer_diag"
    (da, ta), (db, tb) = set_path(diag, args.a), set_path(diag, args.b)
    render = make_renderer(cfg)
    rows = []
    margins_all: list[np.ndarray] = []
    for run in RUNS:
        fa, fb = da / f"{run}__on_{ta}_steps.npz", db / f"{run}__on_{tb}_steps.npz"
        if not (fa.exists() and fb.exists()):
            print(f"skip {run}: missing {fa if not fa.exists() else fb}")
            continue
        A, B = load_steps(fa), load_steps(fb)
        for i, (ea, eb) in enumerate(zip(A, B)):
            r = compare_pair(ea, eb, layouts[i % len(layouts)], cfg, render)
            rows.append({"run": run, "layout": i % len(layouts), **r})
            if len(ea["q"]):
                margins_all.append(q_margin(ea["q"]))
    df = pd.DataFrame(rows)
    tag = f"{args.a}_vs_{args.b}"
    df.to_csv(diag / f"pairs_{tag}.csv", index=False)
    lines = report_tables(df, args.a, args.b, np.concatenate(margins_all) if margins_all else np.zeros(0))
    (diag / f"report_{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plots(df, A_dir=da, ta=ta, B_dir=db, tb=tb, layouts=layouts, cfg=cfg, out=diag, tag=tag,
          margins_all=np.concatenate(margins_all) if margins_all else np.zeros(0))


def report_tables(df: pd.DataFrame, a: str, b: str, margins_all: np.ndarray) -> list[str]:
    L = [f"# {a} vs {b}: per-layout comparison ({df['run'].nunique()} checkpoints x {df['layout'].nunique()} layouts)", ""]
    # 1. outcome table
    cells = {"both goal": (df.out_a == "goal") & (df.out_b == "goal"),
             "both collision": (df.out_a == "collision") & (df.out_b == "collision"),
             f"{a} goal, {b} collision": (df.out_a == "goal") & (df.out_b == "collision"),
             f"{a} collision, {b} goal": (df.out_a == "collision") & (df.out_b == "goal"),
             "other (timeouts)": ~((df.out_a.isin(["goal", "collision"])) & (df.out_b.isin(["goal", "collision"])))}
    L += ["## 1. Outcome table", "", "| cell | " + " | ".join(df.run.unique()) + " | total |", "|---" * (df.run.nunique() + 2) + "|"]
    for name, m in cells.items():
        L.append(f"| {name} | " + " | ".join(str(int(m[df.run == r].sum())) for r in df.run.unique()) + f" | {int(m.sum())} |")
    L.append(f"| success {a} / {b} | " + " | ".join(
        f"{100 * (df.out_a[df.run == r] == 'goal').mean():.0f} / {100 * (df.out_b[df.run == r] == 'goal').mean():.0f}%"
        for r in df.run.unique()) + f" | {100 * (df.out_a == 'goal').mean():.1f} / {100 * (df.out_b == 'goal').mean():.1f}% |")
    gap = cells[f"{a} goal, {b} collision"]
    flips = (df.out_a != df.out_b).sum()
    L += ["", f"Outcome flips: {int(flips)} / {len(df)} pairs ({100 * flips / len(df):.0f}%); net gap = "
          f"{int(gap.sum()) - int(cells[f'{a} collision, {b} goal'].sum())} episodes.", ""]
    # 2. divergence onset
    L += ["## 2. Divergence onset (first step with xy distance > threshold)", "",
          "| threshold | diverged pairs | median step | p25-p75 step | median clearance at onset (m) | gap pairs diverged |",
          "|---|---|---|---|---|---|"]
    for t in THRESH:
        c = f"div{int(t * 100):02d}"
        k = df[c] >= 0
        cl = df.loc[k, f"{c}_clear"] if f"{c}_clear" in df else pd.Series(dtype=float)
        L.append(f"| {t:.2f} m | {int(k.sum())}/{len(df)} | {df.loc[k, c].median():.0f} | "
                 f"{df.loc[k, c].quantile(0.25):.0f}-{df.loc[k, c].quantile(0.75):.0f} | {cl.median():.3f} | "
                 f"{int((k & gap).sum())}/{int(gap.sum())} |")
    # 3. first disagreement
    k = df.first_dis >= 0
    L += ["", "## 3. First action disagreement", "",
          f"Pairs with any disagreement: {int(k.sum())}/{len(df)}; median first step {df.loc[k, 'first_dis'].median():.0f}; "
          f"xy distance between the rollouts at that step: median {df.loc[k, 'dxy_at_dis'].median():.3f} m.", ""]
    if margins_all.size and k.any():
        m = df.loc[k, "margin_a"].dropna()
        L += [f"Q-margin (best - second best, {a}) at the first disagreement: median {m.median():.2f}, "
              f"p75 {m.quantile(0.75):.2f}; over all {a} steps: median {np.median(margins_all):.2f}, "
              f"p25 {np.percentile(margins_all, 25):.2f}. "
              f"Share of first disagreements with margin below the all-steps p25: "
              f"{100 * (m < np.percentile(margins_all, 25)).mean():.0f}% (25% if unrelated).", ""]
    # 4. observation difference at first disagreement (gap pairs, and all)
    feats = ["d_lidar_mean", "d_lidar_max", "d_dist", "d_heading", "d_v", "d_w"]
    L += ["## 4. |obs_a - obs_b| at the first disagreement step (normalised obs units; LiDAR x 3.5 = metres)", "",
          "| pairs | " + " | ".join(f.replace("d_", "") for f in feats) + " |", "|---" * (len(feats) + 1) + "|"]
    for name, m in (("all with a disagreement", k), (f"{a} goal & {b} collision", k & gap)):
        if m.any():
            L.append(f"| {name} (n={int(m.sum())}) | " + " | ".join(f"{df.loc[m, f].median():.4f}" for f in feats) + " |")
    L.append("")
    L.append("Values are medians. heading is in radians, dist in units of 5.66 m.")
    # sensor consistency
    if "b_sensor_err_mean" in df:
        L += ["", f"## Sensor consistency of {b}: its LiDAR vs the noise-free fast-tier LiDAR rendered at its own pose", "",
              f"mean |diff| over all steps and bins: {df.b_sensor_err_mean.mean():.4f} m; mean per-episode p99: "
              f"{df.b_sensor_err_p99.mean():.3f} m. At the final step, (recorded min bin) - (rendered min bin): median "
              f"{df.b_sensor_err_final_min.median():+.4f} m, collisions only "
              f"{df.loc[df.out_b == 'collision', 'b_sensor_err_final_min'].median():+.4f} m.", ""]
    # 5. collision geometry
    gc = df[gap]
    if len(gc):
        L += [f"## 5. Collision geometry of the {a}-goal & {b}-collision episodes (n={len(gc)})", ""]
        kind = gc.b_coll_obj.str.split("_").str[0].value_counts()
        L.append("Object hit: " + ", ".join(f"{k} {v}" for k, v in kind.items()))
        bins = gc.b_coll_bin
        side = np.where(bins.isin([0, 1, 23]), "front (+-22 deg)",
                        np.where(bins.isin([2, 3, 4, 5, 6]), "front-left", np.where(bins.isin([18, 19, 20, 21, 22]), "front-right", "side/rear")))
        L.append("Closest-bin direction at the collision: " + ", ".join(f"{k} {v}" for k, v in pd.Series(side).value_counts().items()))
        L.append(f"Distance to goal at collision: median {gc.b_coll_goal_dist.median():.2f} m (p25-p75 "
                 f"{gc.b_coll_goal_dist.quantile(.25):.2f}-{gc.b_coll_goal_dist.quantile(.75):.2f}).")
        L.append(f"Geometric clearance of the lidar origin at the collision step (from pose + layout): median "
                 f"{gc.b_coll_clear_geom.median():.3f} m; recorded min range {gc.b_coll_min_range.median():.3f} m.")
        per_layout = gc.groupby("layout").size().sort_values(ascending=False)
        L.append(f"Layouts: {len(per_layout)} distinct; hit by all 3 checkpoints: "
                 f"{list(per_layout[per_layout >= 3].index)}; by 2: {list(per_layout[per_layout == 2].index)}.")
        L.append("")
    return L


def plots(df: pd.DataFrame, A_dir: Path, ta: str, B_dir: Path, tb: str, layouts: list, cfg: dict, out: Path, tag: str,
          margins_all: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    a, b = tag.split("_vs_")
    gap = df[(df.out_a == "goal") & (df.out_b == "collision")]
    # (1) overlaid paths on up to 6 gap layouts
    sel = gap.drop_duplicates("layout").head(6)
    if len(sel):
        fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
        half, oc = cfg["arena"]["size"] / 2.0, cfg["obstacles"]
        for ax, (_, r) in zip(axes.flat, sel.iterrows()):
            lay = layouts[r.layout]
            ax.add_patch(Rectangle((-half, -half), 2 * half, 2 * half, fill=False, lw=2, ec="#444"))
            for o in lay.obstacles:
                if o.kind == "cylinder":
                    ax.add_patch(Circle((o.x, o.y), oc["cylinder_radius"], fc="#d9826b", ec="k"))
                else:
                    s = oc["box_size"]
                    ax.add_patch(Rectangle((-s / 2, -s / 2), s, s, fc="#6b8fd9", ec="k", transform=matplotlib.transforms.Affine2D()
                                           .rotate(o.yaw).translate(o.x, o.y) + ax.transData))
            ax.add_patch(Circle(lay.goal, cfg["task"]["goal_radius"], fc="#2ca02c", alpha=0.4))
            ea = load_steps(A_dir / f"{r.run}__on_{ta}_steps.npz")[r.layout]
            eb = load_steps(B_dir / f"{r.run}__on_{tb}_steps.npz")[r.layout]
            ax.plot(ea["pose"][:, 0], ea["pose"][:, 1], "-", color="#1f77b4", lw=1.6, label=f"{a} ({ea['outcome']})")
            ax.plot(eb["pose"][:, 0], eb["pose"][:, 1], "--", color="#9467bd", lw=1.6, label=f"{b} ({eb['outcome']})")
            ax.plot(*lay.start[:2], "ks")
            if r.div05 >= 0:
                ax.plot(*ea["pose"][int(r.div05), :2], "o", ms=9, mfc="none", mec="orange", mew=2, label="diverge > 5 cm")
            if r.first_dis >= 0:
                ax.plot(*ea["pose"][int(r.first_dis), :2], "x", ms=9, color="red", mew=2, label="first action disagreement")
            ax.set(xlim=(-half - .1, half + .1), ylim=(-half - .1, half + .1), aspect="equal",
                   title=f"{r.run.replace('fast_dqn_', '')}, layout {r.layout}")
            ax.legend(fontsize=7, loc="lower left")
            ax.grid(alpha=.3)
        for ax in list(axes.flat)[len(sel):]:
            ax.axis("off")
        fig.suptitle(f"Paths where {a} reaches the goal and {b} collides (same checkpoint, same layout)")
        fig.tight_layout()
        fig.savefig(out / f"paths_{tag}.png", dpi=130)
        plt.close(fig)
    # (2) divergence onset histogram, (3) Q-margin at first disagreement
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    for t, col in zip(THRESH, ["#1f77b4", "#ff7f0e", "#d62728"]):
        c = f"div{int(t * 100):02d}"
        v = df.loc[df[c] >= 0, c]
        axs[0].hist(v, bins=np.arange(0, 301, 10), histtype="step", lw=1.8, color=col, label=f"> {t:.2f} m (n={len(v)})")
    axs[0].set(xlabel="control step", ylabel="pairs", title=f"Divergence onset, {a} vs {b}")
    axs[0].legend()
    axs[0].grid(alpha=.3)
    m = df.loc[df.first_dis >= 0, "margin_a"].dropna()
    hi = max(np.percentile(margins_all, 95) if margins_all.size else 1, m.max() if len(m) else 1)
    bins = np.linspace(0, hi, 30)
    if margins_all.size:
        axs[1].hist(margins_all, bins=bins, density=True, alpha=.35, color="grey", label=f"all {a} steps")
    axs[1].hist(m, bins=bins, density=True, histtype="step", lw=2, color="#d62728", label=f"at first disagreement (n={len(m)})")
    axs[1].set(xlabel="Q margin (best - second best)", ylabel="density", title="Q-value margin")
    axs[1].legend()
    axs[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / f"divergence_{tag}.png", dpi=130)
    plt.close(fig)


# ---- counterfactual replay ----------------------------------------------------------------
def replay(args: argparse.Namespace) -> None:
    """Replay the recorded action sequences of --src open-loop in --env on the given pairs."""
    cfg = load_config(tier=args.env)
    apply_overrides(cfg, args.overrides)
    layouts = eval_layouts(cfg)
    diag = results_dir(cfg) / "transfer_diag"
    pairs = pd.read_csv(diag / args.pairs)
    sel = pairs[(pairs.out_a == "goal") & (pairs.out_b == "collision")]
    src_dir, src_tier = set_path(diag, args.src)
    from envs import make_env
    env = make_env(args.env, cfg)
    if args.env == "gz":
        env.check_connections(verbose=False)
    env.reset(seed=20_000)
    rows, trajs = [], {}
    for run, grp in sel.groupby("run"):
        rec = load_steps(src_dir / f"{run}__on_{src_tier}_steps.npz")
        for lay_i in grp.layout:
            e = rec[lay_i]
            _, info = env.reset(options={"layout": layouts[lay_i]})
            xy, out, k = [info["pose"].copy()], "sequence_end", 0
            for k, a in enumerate(e["action"]):
                _, _, term, trunc, info = env.step(int(a))
                xy.append(info["pose"].copy())
                if term or trunc:
                    out = info["outcome"]
                    break
            p = np.asarray(xy)
            n = min(len(p), len(e["pose"]))
            rows.append({"run": run, "layout": int(lay_i), "src": args.src, "env": args.env, "src_outcome": str(e["outcome"]),
                         "replay_outcome": out, "steps": k + 1, "src_steps": len(e["action"]),
                         "final_dxy_vs_src": float(np.hypot(*(p[n - 1, :2] - e["pose"][n - 1, :2]))),
                         "max_dxy_vs_src": float(np.max(np.hypot(p[:n, 0] - e["pose"][:n, 0], p[:n, 1] - e["pose"][:n, 1])))})
            trajs[f"{run}_L{lay_i}"] = p
            print(rows[-1], flush=True)
    tag = f"replay_{args.src}_in_{args.env}{('_' + args.tag) if args.tag else ''}"
    df = pd.DataFrame(rows)
    df.to_csv(diag / f"{tag}.csv", index=False)
    np.savez(diag / f"{tag}_traj.npz", **trajs)
    print(f"\n{tag}: replay outcomes:\n{df.replay_outcome.value_counts().to_string()}")
    if args.env == "gz":
        env.close()
        os._exit(0)


# ---- odometry drift by scan matching -------------------------------------------------------
def scan_match(render: Any, layout: Layout, pose: np.ndarray, scan: np.ndarray, start: np.ndarray) -> tuple[np.ndarray, float]:
    """Local coordinate search for the pose whose noise-free fast-tier scan best explains `scan` (bins, metres)."""
    clip = 3.5

    def err(p: np.ndarray) -> float:
        return float(np.abs(np.minimum(render(layout, p), clip) - scan).mean())
    best = np.asarray(start, dtype=np.float64).copy()
    be = err(best)
    for step in (0.08, 0.04, 0.02, 0.01, 0.005):
        improved = True
        while improved:
            improved = False
            for d in ((step, 0, 0), (-step, 0, 0), (0, step, 0), (0, -step, 0), (0, 0, step), (0, 0, -step)):
                c = best + np.asarray(d)
                e = err(c)
                if e < be - 1e-6:
                    best, be, improved = c, e, True
    return best, be


def drift(args: argparse.Namespace) -> None:
    """Track, for every episode of a recording set, the pose that explains the recorded LiDAR (scan matching against
    the known layout) and compare it with the pose the env reported (wheel odometry in Gazebo)."""
    from envs.common import wrap_angle
    cfg = load_config(tier="fast")
    layouts = eval_layouts(cfg)
    diag = results_dir(cfg) / "transfer_diag"
    d, tier = set_path(diag, args.set)
    render = make_renderer(cfg)
    rows = []
    for run in RUNS:
        f = d / f"{run}__on_{tier}_steps.npz"
        if not f.exists():
            continue
        for i, e in enumerate(load_steps(f)):
            lay = layouts[i % len(layouts)]
            off = np.zeros(3)
            yaw_err, xy_err, resid = [], [], []
            for k in range(0, len(e["pose"]), args.every):
                p, be = scan_match(render, lay, e["pose"][k], e["obs"][k][:24] * 3.5, e["pose"][k] + off)
                off = p - e["pose"][k]
                yaw_err.append(abs(wrap_angle(off[2])))
                xy_err.append(float(np.hypot(off[0], off[1])))
                resid.append(be)
            rows.append({"run": run, "layout": i % len(layouts), "outcome": str(e["outcome"]), "steps": len(e["action"]),
                         "final_yaw_err": yaw_err[-1], "max_yaw_err": max(yaw_err), "final_xy_err": xy_err[-1],
                         "max_xy_err": max(xy_err), "match_resid": float(np.median(resid))})
    df = pd.DataFrame(rows)
    df.to_csv(diag / f"drift_{args.set}.csv", index=False)
    print(f"odometry drift in {args.set} (scan-matched pose vs reported pose, every {args.every} steps, n={len(df)}):")
    for name, sub in (("all", df), ("goal", df[df.outcome == "goal"]), ("collision", df[df.outcome == "collision"])):
        if len(sub):
            print(f"  {name:<9} n={len(sub):>3}  max yaw err median {sub.max_yaw_err.median():.3f} rad (p90 "
                  f"{sub.max_yaw_err.quantile(.9):.3f}), share > 0.5 rad {100 * (sub.max_yaw_err > 0.5).mean():.0f}%; "
                  f"max xy err median {sub.max_xy_err.median():.3f} m; match residual {sub.match_resid.median():.4f} m")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    an = sub.add_parser("analyse")
    an.add_argument("--a", default="fast")
    an.add_argument("--b", default="gz")
    rp = sub.add_parser("replay")
    rp.add_argument("--src", required=True, help="recording set whose actions are replayed (fast, gz, ...)")
    rp.add_argument("--env", choices=["fast", "gz"], required=True)
    rp.add_argument("--pairs", default="pairs_fast_vs_gz.csv", help="pairs CSV from `analyse` (uses its a-goal & b-collision rows)")
    rp.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VAL")
    rp.add_argument("--tag", default="")
    dr = sub.add_parser("drift")
    dr.add_argument("--set", required=True)
    dr.add_argument("--every", type=int, default=5, help="scan-match every N steps")
    args = ap.parse_args()
    {"analyse": analyse, "replay": replay, "drift": drift}[args.cmd](args)


if __name__ == "__main__":
    main()
