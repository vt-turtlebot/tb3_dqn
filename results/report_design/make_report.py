"""Builds results/report_design/tb3_dqn_phase1_report.pdf from the project's real result files.

    ../../venv/bin/python make_report.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
from envs.layout import eval_layouts  # noqa: E402
from utils import load_config  # noqa: E402

FONTS = Path(__file__).parent / "fonts"
for f in FONTS.glob("*.ttf"):
    fm.fontManager.addfont(str(f))
SERIF = fm.FontProperties(fname=str(FONTS / "InstrumentSerif-Regular.ttf"))
SERIF_I = fm.FontProperties(fname=str(FONTS / "InstrumentSerif-Italic.ttf"))
SANS = fm.FontProperties(fname=str(FONTS / "InstrumentSans-Regular.ttf"))
SANS_B = fm.FontProperties(fname=str(FONTS / "InstrumentSans-Bold.ttf"))
MONO = fm.FontProperties(fname=str(FONTS / "JetBrainsMono-Regular.ttf"))

PAPER, INK, GREY, RULE = "#F4F1EA", "#1E1E1C", "#8A867C", "#C9C3B6"
FAIL, HOLD, FAST = "#B5462E", "#2F6B5A", "#2B4C7E"
W, H = 8.27, 11.69                      # A4 portrait, inches
ML, MR = 0.09, 0.91                     # margins (figure fraction)

plt.rcParams.update({"axes.edgecolor": INK, "axes.linewidth": 0.6, "xtick.color": GREY, "ytick.color": GREY,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 3, "ytick.major.size": 3})


def page(n: int, section: str):
    fig = plt.figure(figsize=(W, H), facecolor=PAPER)
    # frame: running head, hairline, folio, registration ticks
    fig.text(ML, 0.955, "TURTLEBOT3 · VANILLA DQN · PHASE 1", fontproperties=MONO, fontsize=6.5, color=GREY)
    fig.text(MR, 0.955, section.upper(), fontproperties=MONO, fontsize=6.5, color=GREY, ha="right")
    fig.add_artist(plt.Line2D([ML, MR], [0.948, 0.948], color=RULE, lw=0.5))
    fig.add_artist(plt.Line2D([ML, MR], [0.052, 0.052], color=RULE, lw=0.5))
    fig.text(ML, 0.038, "19 OCT 2026", fontproperties=MONO, fontsize=6.5, color=GREY)
    fig.text(MR, 0.038, f"{n:02d} / 05", fontproperties=MONO, fontsize=6.5, color=GREY, ha="right")
    for x, y in ((0.03, 0.97), (0.97, 0.97), (0.03, 0.03), (0.97, 0.03)):
        fig.add_artist(plt.Line2D([x - 0.008, x + 0.008], [y, y], color=RULE, lw=0.5))
        fig.add_artist(plt.Line2D([x, x], [y - 0.006, y + 0.006], color=RULE, lw=0.5))
    return fig


def text(fig, x, y, s, fp=SANS, size=9, color=INK, **kw):
    return fig.text(x, y, s, fontproperties=fp, fontsize=size, color=color, **kw)


def marker(fig, x, y, k):
    text(fig, x, y, f"[{k}]", MONO, 7, GREY)


def style(ax, grid_y=True):
    ax.set_facecolor(PAPER)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(MONO)
        lab.set_fontsize(6.5)
    if grid_y:
        ax.yaxis.grid(True, color=RULE, lw=0.4)
        ax.set_axisbelow(True)


# ---------------------------------------------------------------- data
rep = pd.read_csv(RES / "report/summary.csv")
fast = rep[(rep.eval_env == "fast") & rep.run.str.startswith("fast_dqn")].set_index("run").sort_index()
gzt = rep[(rep.eval_env == "gz") & rep.run.str.startswith("fast_dqn")].set_index("run").reindex(fast.index)
gzown = rep[(rep.eval_env == "gz") & rep.run.str.startswith("gz_dqn")]
oracle = rep[(rep.run == "oracle_dqn") & (rep.eval_env == "fast")].success_rate.iloc[0] * 100
rnd = rep[(rep.run == "random_dqn") & (rep.eval_env == "fast")].success_rate.iloc[0] * 100

def save(pdf, fig):
    pdf.savefig(fig, facecolor=PAPER)
    fig.savefig(RES / f"report_design/page_{len(fig.texts):03d}_{pdf.get_pagecount() + 1}.png", dpi=90, facecolor=PAPER)


with PdfPages(RES / "report_design/tb3_dqn_phase1_report.pdf") as pdf:
    # ================================================================ 01 context
    fig = page(1, "Context")
    text(fig, ML, 0.86, "Closing the gap", SERIF, 50)
    text(fig, ML, 0.825, "between a fast 2D simulator and Gazebo, for a robot learning to reach goals",
         SERIF_I, 15, GREY)

    # research line: four stations, this run marked
    y0 = 0.70
    stations = [("Phase 1", "Vanilla DQN,\n5 discrete actions"), ("Phase 2", "TD3,\ncontinuous actions"),
                ("Hardware", "Real TurtleBot3\nBurger"), ("Goal", "Human feedback\n(RLHF) paper")]
    xs = np.linspace(ML + 0.04, MR - 0.06, 4)
    fig.add_artist(plt.Line2D([xs[0], xs[-1]], [y0, y0], color=INK, lw=0.6))
    for i, (x, (a, b)) in enumerate(zip(xs, stations)):
        filled = i == 0
        fig.add_artist(Circle((x, y0), 0.009, transform=fig.transFigure, fc=INK if filled else PAPER, ec=INK, lw=0.8))
        text(fig, x, y0 + 0.022, a.upper(), MONO, 7, INK if filled else GREY, ha="center")
        text(fig, x, y0 - 0.045, b, SANS, 8.5, INK if filled else GREY, ha="center", linespacing=1.4)
    text(fig, xs[0], y0 + 0.042, "THIS RUN", MONO, 7, FAIL, ha="center")

    text(fig, ML, 0.565, "Where this run sits", SANS_B, 10)
    para = ("Phase 1 trains a standard DQN agent in two simulators that share one interface: a pure-Python 2D\n"
            "simulator that is fast enough for full training, and Gazebo, which is slower but closer to the real\n"
            "robot. A policy is trained in 2D and then tested in Gazebo. The previous report found that policies\n"
            "lost 23 points of success in that move, without knowing why. This run had three jobs:")
    text(fig, ML, 0.48, para, SANS, 9.5, linespacing=1.65)
    jobs = [("A", "Explain the 23-point drop, then fix it or show it can't be fixed."),
            ("B", "Report final numbers on layouts no decision ever used, with 10 seeds instead of 3."),
            ("C", "Show training is stable and that results reproduce exactly.")]
    for i, (k, s) in enumerate(jobs):
        yy = 0.425 - i * 0.042
        text(fig, ML + 0.005, yy, k, SERIF, 15, FAIL)
        text(fig, ML + 0.04, yy + 0.003, s, SANS, 9.5, linespacing=1.5)

    # three headline numbers
    fig.add_artist(plt.Line2D([ML, MR], [0.26, 0.26], color=RULE, lw=0.5))
    heads = [("23 to 1.6", "points of success lost\nmoving to Gazebo"), ("73.7%", "goal success, 10 seeds,\n200 held-out layouts"),
             ("bit-exact", "same seed reruns and\nresumed runs match")]
    for i, (big, small) in enumerate(heads):
        x = ML + i * (MR - ML) / 3
        text(fig, x, 0.175, big, SERIF, 30)
        text(fig, x, 0.125, small, SANS, 8.5, GREY, linespacing=1.45)
    text(fig, ML, 0.075, "Vanilla DQN only (no Double, Dueling or prioritised replay). Every number in this document comes "
         "from files in results/.", SANS, 7.5, GREY)
    save(pdf, fig)
    plt.close(fig)

    # ================================================================ 02 mechanism
    fig = page(2, "What was wrong")
    text(fig, ML, 0.875, "The robot believed it was elsewhere", SERIF, 30)
    text(fig, ML, 0.84, "Same policy, same layout, three simulators. The 2D path is the broad pale band.", SANS, 9.5, GREY)

    cfg = load_config(tier="fast")
    layouts = eval_layouts(cfg)
    pairs = pd.read_csv(RES / "transfer_diag/pairs_fast_vs_gz_stock.csv")
    gap = pairs[(pairs.out_a == "goal") & (pairs.out_b == "collision") & (pairs.run == "fast_dqn_seed0")]
    lay_i = int(gap.layout.iloc[0])
    lay = layouts[lay_i]

    def load_xy(sub, run):
        with np.load(RES / f"transfer_diag/{sub}{run}__on_{'fast' if sub == '' else 'gz'}_traj.npz") as z:
            return z[f"ep{lay_i}"]
    paths = [("", "2D simulator", FAST), ("gz_stock/", "Gazebo, before the fix", FAIL), ("gz_gt/", "Gazebo, after the fix", HOLD)]
    ax = fig.add_axes([ML, 0.40, 0.50, 0.40])
    half = cfg["arena"]["size"] / 2
    ax.add_patch(Rectangle((-half, -half), 2 * half, 2 * half, fill=False, ec=INK, lw=0.8))
    for o in lay.obstacles:
        if o.kind == "cylinder":
            ax.add_patch(Circle((o.x, o.y), cfg["obstacles"]["cylinder_radius"], fc=RULE, ec=INK, lw=0.5))
        else:
            s = cfg["obstacles"]["box_size"]
            ax.add_patch(Rectangle((-s / 2, -s / 2), s, s, fc=RULE, ec=INK, lw=0.5,
                                   transform=matplotlib.transforms.Affine2D().rotate(o.yaw).translate(o.x, o.y) + ax.transData))
    ax.add_patch(Circle(lay.goal, 0.2, fc="none", ec=HOLD, lw=0.8, ls=(0, (2, 2))))
    for sub, name, col in paths:
        xy = load_xy(sub, "fast_dqn_seed0")
        wide = sub == ""                                  # 2D path drawn as a broad band under the others
        ax.plot(xy[:, 0], xy[:, 1], color=col, lw=4.5 if wide else 1.2, alpha=0.28 if wide else 1.0,
                solid_capstyle="round")
        ax.plot(*xy[-1], "o" if sub != "gz_stock/" else "x", color=col, ms=5, mew=1.4)
    ax.plot(*lay.start[:2], "s", color=INK, ms=4)
    ax.set(xlim=(-half - 0.1, half + 0.1), ylim=(-half - 0.1, half + 0.1), aspect="equal")
    ax.axis("off")
    text(fig, ML, 0.385, f"layout {lay_i} · seed 0 · square start · ring goal · x where Gazebo's LiDAR read a collision\n"
         "while the env, trusting wheel odometry, placed the robot in open floor", MONO, 6.2, GREY, linespacing=1.6)

    lx = 0.63
    for i, (_, name, col) in enumerate(paths):
        yy = 0.77 - i * 0.05
        fig.add_artist(plt.Line2D([lx, lx + 0.04], [yy + 0.004, yy + 0.004], color=col,
                                  lw=4.5 if i == 0 else 1.6, alpha=0.28 if i == 0 else 1.0))
        text(fig, lx + 0.055, yy, name, SANS, 9)
    text(fig, lx, 0.585, "Two defects in the Gazebo tier", SANS_B, 9.5)
    notes = [("1", "Wheel grip", "After a held hard turn, Gazebo's\ndefault collision detector let the\n"
                                 "body keep spinning for 1–2 s while\nthe wheels drove straight."),
             ("2", "Wrong pose", "The env read its position from\nwheel odometry, which cannot see\n"
                                 "that spin. Heading to the goal was\noff by up to 2 rad.")]
    for i, (k, t, s) in enumerate(notes):
        yy = 0.545 - i * 0.105
        marker(fig, lx, yy, k)
        text(fig, lx + 0.04, yy, t, SANS_B, 8.5)
        text(fig, lx + 0.04, yy - 0.068, s, SANS, 8, GREY, linespacing=1.45)

    # coast table as a drawn ledger
    text(fig, ML, 0.31, "Uncommanded yaw after a hard turn, measured against the physics engine's true pose", SANS_B, 9.5)
    rows = [("default (FCL) collision detector", 2.22), ("+ base_footprint inertia fixed", 2.29), ("ODE detector", 2.21),
            ("bullet detector", 0.16), ("bullet + inertia fixed  (final)", 0.11)]
    for i, (lab, v) in enumerate(rows):
        yy = 0.27 - i * 0.032
        col = HOLD if v < 0.5 else FAIL
        text(fig, ML, yy, lab, SANS, 8.5)
        x0 = 0.47
        fig.add_artist(Rectangle((x0, yy - 0.002), v / 2.5 * 0.33, 0.012, transform=fig.transFigure, fc=col, ec="none"))
        text(fig, x0 + v / 2.5 * 0.33 + 0.012, yy, f"{v:.2f} rad", MONO, 7.5, col)
    text(fig, ML, 0.09, "Only 7% of the goals Gazebo recorded before the fix were real: scan-matching the LiDAR against the "
         "known layout\nput the robot within 0.2 m of the goal in 6 of 86 'successes'. Fix: bullet detector, true pose "
         "from the physics engine.", SANS, 8, GREY, linespacing=1.5)
    save(pdf, fig)
    plt.close(fig)

    # ================================================================ 03 decomposition
    fig = page(3, "Where the 23 points went")
    text(fig, ML, 0.875, "One fix at a time", SERIF, 30)
    text(fig, ML, 0.84, "Same three checkpoints, same 50 layouts. Only the Gazebo tier changes.", SANS, 9.5, GREY)
    steps = [("Gazebo as before", 57.3, 41), ("+ true pose", 74.0, 17), ("+ bullet physics", 80.0, 5)]
    ax = fig.add_axes([ML + 0.02, 0.47, 0.80, 0.32])
    for i, (lab, v, _) in enumerate(steps):
        col = FAIL if i == 0 else HOLD
        ax.bar(i, v, width=0.5, color=col, alpha=0.25 + 0.35 * i)
        ax.text(i, v - 9, f"{v:.1f}%", ha="center", fontproperties=SERIF, fontsize=20, color=PAPER if i == 2 else INK)
    ax.axhline(82.0, color=FAST, lw=0.9, ls=(0, (4, 3)))
    ax.text(-0.55, 84, "2D simulator  82.0%", ha="left", fontproperties=MONO, fontsize=7, color=FAST)
    ax.set_xticks(range(3), [s[0] for s in steps])
    for lab in ax.get_xticklabels():
        lab.set_fontproperties(SANS)
        lab.set_fontsize(9)
        lab.set_color(INK)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.6, 2.6)
    style(ax)
    for lab in ax.get_xticklabels():
        lab.set_fontproperties(SANS)
        lab.set_fontsize(9)
    ax.set_ylabel("goal success", fontproperties=MONO, fontsize=7, color=GREY)

    text(fig, ML, 0.37, "Outcome flips: layouts where the two simulators disagree on goal vs collision", SANS_B, 9.5)
    flips = [("before", 41, FAIL), ("true pose", 17, FAIL), ("both fixes", 5, HOLD), ("noise floor*", 3, GREY)]
    for i, (lab, v, col) in enumerate(flips):
        yy = 0.325 - i * 0.034
        text(fig, ML, yy, lab, SANS, 8.5)
        # 50 ticks = 100%, filled share shown as dense marks
        for k in range(50):
            x = 0.27 + k * 0.0108
            on = k < round(v / 2)
            fig.add_artist(plt.Line2D([x, x], [yy - 0.002, yy + 0.011], color=col if on else RULE, lw=1.6 if on else 0.6))
        text(fig, 0.83, yy, f"{v}%", MONO, 8, col)
    text(fig, ML, 0.17, "* the 2D simulator against itself with a different LiDAR-noise seed: the policy's own chance level.",
         SANS, 7.5, GREY)
    text(fig, ML, 0.12, "Ruled out: the 47–56 ms command delay Gazebo adds. Injecting 0–90 ms into the 2D simulator left "
         "success at 81–83%,\nand retraining with it moved Gazebo success by +3.3 points, which is noise. "
         "Collision rule and LiDAR geometry already match.", SANS, 8, GREY, linespacing=1.5)
    save(pdf, fig)
    plt.close(fig)

    # ================================================================ 04 final numbers
    fig = page(4, "Final results")
    text(fig, ML, 0.875, "Ten seeds, two simulators", SERIF, 30)
    text(fig, ML, 0.84, "Last checkpoint of every run, 200 layouts that no decision ever used.", SANS, 9.5, GREY)
    ax = fig.add_axes([ML + 0.02, 0.43, 0.82, 0.37])
    f_s, g_s = fast.success_rate.values * 100, gzt.success_rate.values * 100
    for i, (a, b) in enumerate(zip(f_s, g_s)):
        ax.plot([0, 1], [a, b], color=GREY, lw=0.6, alpha=0.7)
        ax.plot(0, a, "o", color=FAST, ms=5)
        ax.plot(1, b, "o", color=HOLD, ms=5)
    gs = gzown.success_rate.values * 100
    ax.plot(np.full(len(gs), 2), gs, "o", mfc=PAPER, mec=FAIL, ms=5, mew=1.2)
    for x, vals in ((0, f_s), (1, g_s), (2, gs)):
        ax.plot([x - 0.18, x + 0.18], [vals.mean()] * 2, color=INK, lw=1.4)
        ax.text(x + 0.24, vals.mean() - 1, f"{vals.mean():.1f}", fontproperties=SERIF, fontsize=16, color=INK)
    ax.axhline(oracle, color=GREY, lw=0.7, ls=(0, (1, 2)))
    ax.text(2.75, oracle + 1, f"ignores LiDAR {oracle:.0f}%", ha="right", fontproperties=MONO, fontsize=6.5, color=GREY)
    ax.axhline(rnd, color=GREY, lw=0.7, ls=(0, (1, 2)))
    ax.text(2.75, rnd + 1, f"random {rnd:.1f}%", ha="right", fontproperties=MONO, fontsize=6.5, color=GREY)
    ax.set_xticks([0, 1, 2], ["trained in 2D,\ntested in 2D", "trained in 2D,\ntested in Gazebo",
                              "trained in Gazebo\n(30k steps, 3 seeds)"])
    ax.set_ylim(0, 90)
    ax.set_xlim(-0.4, 2.8)
    style(ax)
    for lab in ax.get_xticklabels():
        lab.set_fontproperties(SANS)
        lab.set_fontsize(8.5)
        lab.set_color(INK)
    ax.set_ylabel("goal success (%)", fontproperties=MONO, fontsize=7, color=GREY)

    rows = [("Trained in 2D, tested in 2D", "10", "73.7 ± 2.4", "26.4"), ("Trained in 2D, tested in Gazebo", "10", "72.0 ± 2.4", "27.9"),
            ("Trained in Gazebo, 30k steps", "3", "60.2 ± 13.0", "39.8"), ("Heading only, ignores LiDAR", "–", "68.0", "32.0"),
            ("Random actions", "–", "4.5", "95.5")]
    cols = [ML, 0.52, 0.62, 0.80]
    for j, h in enumerate(["CONDITION", "SEEDS", "SUCCESS %", "COLLISION %"]):
        text(fig, cols[j], 0.335, h, MONO, 6.5, GREY)
    fig.add_artist(plt.Line2D([ML, MR], [0.328, 0.328], color=RULE, lw=0.5))
    for i, r in enumerate(rows):
        yy = 0.302 - i * 0.03
        for j, v in enumerate(r):
            text(fig, cols[j], yy, v, SANS if j == 0 else MONO, 8.5)
    text(fig, ML, 0.12, "Transfer gap 1.6 points (95% CI 0.0 to 3.2), paired over seeds; 5.9% of layout outcomes differ. Failures are "
         "collisions; timeouts are 0%.\nDQN beats a controller that ignores obstacles by about 6 points: a real but "
         "modest margin, and the headroom for phase 2.", SANS, 8, GREY, linespacing=1.5)
    save(pdf, fig)
    plt.close(fig)

    # ================================================================ 05 stability + ledger
    fig = page(5, "Stability and record")
    text(fig, ML, 0.875, "Stable, and exactly repeatable", SERIF, 30)
    text(fig, ML, 0.84, "Success on the tuning layouts during training, every seed drawn.", SANS, 9.5, GREY)
    ax = fig.add_axes([ML + 0.02, 0.55, 0.82, 0.26])
    for f in sorted(glob.glob(str(RES / "runs/fast_dqn_seed*/eval.csv"))):
        e = pd.read_csv(f)
        ax.plot(e.step / 1000, e.success_rate * 100, color=FAST, lw=0.7, alpha=0.35)
    allc = np.stack([pd.read_csv(f).success_rate.values for f in sorted(glob.glob(str(RES / "runs/fast_dqn_seed*/eval.csv")))])
    ax.plot(pd.read_csv(f).step / 1000, allc.mean(0) * 100, color=INK, lw=1.4)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 200)
    style(ax)
    ax.set_xlabel("training steps (thousands)", fontproperties=MONO, fontsize=7, color=GREY)

    ledger = [("No collapse", "16 shallow dips in 390 evaluations; deepest 20 points. The untuned run once fell to 3%."),
              ("No Q divergence", "Mean max-Q rises smoothly to ~70; loss falls slowly; Q at dips is no higher than elsewhere."),
              ("Mild overestimation", "Late in training Q(start) is about 53 vs a realised discounted return of 51."),
              ("Same seed, same run", "A rerun matches the original bit for bit, down to the final network weights."),
              ("Survives restarts", "Three runs cut by real WSL restarts resumed to bit-identical results."),
              ("Provenance", "Each run stores package versions and hashes of the config and every source file.")]
    text(fig, ML, 0.47, "Ledger", SANS_B, 9.5)
    for i, (k, s) in enumerate(ledger):
        yy = 0.435 - i * 0.04
        fig.add_artist(Circle((ML + 0.006, yy + 0.004), 0.0035, transform=fig.transFigure, fc=HOLD, ec="none"))
        text(fig, ML + 0.025, yy, k, SANS_B, 8.5)
        text(fig, 0.33, yy, s, SANS, 8.5, GREY)
    text(fig, ML, 0.128, "Caveats", SANS_B, 9.5)
    text(fig, ML, 0.075, "One task, one reward, one robot model. Gazebo now uses the true pose, which a real robot will not "
         "have.\nGazebo is not deterministic, so its numbers repeat statistically (about 3 points per checkpoint), not bit for bit.\n"
         "The Gazebo-trained baseline has 3 seeds and a 30k-step budget.", SANS, 8, GREY, linespacing=1.55)
    save(pdf, fig)
    plt.close(fig)
print("written", RES / "report_design/tb3_dqn_phase1_report.pdf")
