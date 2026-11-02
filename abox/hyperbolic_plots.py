"""Figures for Step 62 from data/hyperbolic_proof (written by abox/hyperbolic_proof.py).

    .venv-crystallm\\Scripts\\python abox/hyperbolic_plots.py

  fig_poincare_disk.png   base OnT vs K-Ont: concepts with their TBox edges, crystals coloured by structure
                          family, materials by battery role. Radius = exact hyperbolic distance from the origin.
  fig_depth.png           distance from the origin per ontology level, base vs K-Ont
  fig_family_confusion.png  held-out crystals: true family vs K-Ont's predicted family
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hyperbolic_proof")
FAM_COL = {"LayeredOxideStructure": "#1f77b4", "SpinelStructure": "#d62728", "RockSaltStructure": "#8c564b",
           "OlivineStructure": "#2ca02c", "NASICONStructure": "#ff7f0e", "GarnetStructure": "#9467bd",
           "PerovskiteStructure": "#e377c2", "LGPSTypeStructure": "#17becf", "ArgyroditeStructure": "#bcbd22"}
FAM_NAME = {"LayeredOxideStructure": "layered oxide", "SpinelStructure": "spinel", "RockSaltStructure": "rock salt",
            "OlivineStructure": "olivine", "NASICONStructure": "NASICON", "GarnetStructure": "garnet",
            "PerovskiteStructure": "perovskite", "LGPSTypeStructure": "LGPS-type", "ArgyroditeStructure": "argyrodite"}
FAM_CLASS = {"LayeredOxideStructure": "layered oxide structure", "SpinelStructure": "spinel structure",
             "RockSaltStructure": "rock salt structure", "OlivineStructure": "olivine structure",
             "NASICONStructure": "NASICON structure", "GarnetStructure": "garnet structure",
             "PerovskiteStructure": "perovskite structure", "LGPSTypeStructure": "LGPS type structure",
             "ArgyroditeStructure": "argyrodite structure"}
ROLE_COL = {"PositiveElectrode": "#a4123f", "NegativeElectrode": "#2e8b8b"}
LVL_COL = {0: "#1d2129", 1: "#6b7380", 2: "#b8891f"}


ZOOM = (10.0, 20.0)  # radial axis of the zoomed view: hyperbolic distance from the origin


def disk(ax, m, lab, title, c, zoom=False):
    """True Poincare disk, or (zoom) the same angles with radius = hyperbolic distance from the origin, axis 10-20."""
    def tf(xy):
        if not zoom:
            return xy
        r = np.linalg.norm(xy, axis=1, keepdims=True)
        d0 = 2 / np.sqrt(c) * np.arctanh(np.clip(r, 0, 1 - 1e-9))
        return xy / np.maximum(r, 1e-12) * np.clip((d0 - ZOOM[0]) / (ZOOM[1] - ZOOM[0]), 0, 1)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, lw=1, color="#888"))
    rings = [(d - ZOOM[0]) / (ZOOM[1] - ZOOM[0]) for d in (12, 14, 16, 18)] if zoom else (0.25, 0.5, 0.75)
    for d, r in zip((12, 14, 16, 18), rings):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, lw=0.4, ls=":", color="#bbb"))
        if zoom:
            ax.annotate(f"d = {d}", (0, -r), fontsize=6, color="#999", ha="center", va="top")
    if zoom:
        ax.annotate("d = 20", (0, -1), fontsize=6, color="#999", ha="center", va="top")
    Mx, Xx, Ex, Cx = tf(m["M"]), tf(m["X"]), tf(m["E"]), tf(m["C"])
    ax.scatter(Mx[:, 0], Mx[:, 1], s=1, c="#d5d8de", lw=0, label="materials (no label)", zorder=1)
    mrole = [lab["material_role"][i] for i in m["M_idx"]]
    xfam = [lab["crystal_family"][i] for i in m["X_idx"]]
    for role, col in ROLE_COL.items():
        i = [n for n, r in enumerate(mrole) if r == role]
        ax.scatter(Mx[i, 0], Mx[i, 1], s=2, c=col, lw=0, alpha=0.5, zorder=2,
                   label="materials: " + ("cathode" if role == "PositiveElectrode" else "anode"))
    ax.scatter(Ex[:, 0], Ex[:, 1], s=6, c="#3a3f4b", lw=0, zorder=2, label="elements")
    for f, col in FAM_COL.items():
        i = [n for n, x in enumerate(xfam) if x == f]
        ax.scatter(Xx[i, 0], Xx[i, 1], s=9, c=col, lw=0, zorder=3, label=f"crystals: {FAM_NAME[f]}")
    cls = lab["classes"]
    for ch, pa in lab["tbox"]:
        a, b = Cx[cls.index(ch)], Cx[cls.index(pa)]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#444", lw=0.7, zorder=4)
    for n, cl in enumerate(cls):
        lv = lab["class_level"][cl]
        ax.scatter(*Cx[n], s=36, marker="D", c=LVL_COL[lv], edgecolors="white", lw=0.6, zorder=5)
        ang = np.arctan2(Cx[n][1], Cx[n][0])
        ax.annotate(cl, Cx[n], fontsize=5.5, xytext=(-4 if np.cos(ang) < 0 else 4, 2 if n % 2 else -6),
                    textcoords="offset points", ha="right" if np.cos(ang) < 0 else "left", zorder=6, color="#1d2129")
    ax.scatter([0], [0], marker="+", c="k", s=30, zorder=6)
    ax.set_xlim(-1.03, 1.03), ax.set_ylim(-1.03, 1.03), ax.set_aspect("equal"), ax.axis("off")
    ax.set_title(title, fontsize=10)


def main():
    z, lab, rep = np.load(os.path.join(D, "map2d.npz")), json.load(open(os.path.join(D, "map_labels.json"), encoding="utf-8")), \
        json.load(open(os.path.join(D, "report.json"), encoding="utf-8"))
    m = {t: {k[len(t) + 1:]: z[k] for k in z.files if k.startswith(t + "_")} for t in ("base", "kont")}

    for zoom, fname, how in ((False, "fig_poincare_disk.png", "Poincaré disk (true scale). Radius = exact hyperbolic distance from the origin"),
                             (True, "fig_poincare_zoom.png", "Radially zoomed: radius = hyperbolic distance from the origin, drawn from 10 (centre) to 20 (rim)")):
        fig, axs = plt.subplots(1, 2, figsize=(13, 6.8))
        disk(axs[0], m["base"], lab, "Before: OnT-MiniLM-L12-galen (never saw the battery graph)", rep["curvature"], zoom)
        disk(axs[1], m["kont"], lab, "After: K-Ont (5,000-material graph)", rep["curvature"], zoom)
        h, l = axs[1].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=7, fontsize=7, markerscale=2.5, frameon=False)
        fig.text(0.5, 0.075, how + "; angles fitted so 2-D hyperbolic distances match the model's pairwise distances.\n"
                 "Diamonds: ontology classes (black root, grey level 1, gold leaf family) joined by their subclass axioms.",
                 ha="center", fontsize=7.5, color="#4a5160")
        fig.subplots_adjust(bottom=0.14, top=0.94, wspace=0.02)
        fig.savefig(os.path.join(D, fname), dpi=220)

    # depth per level
    fig, ax = plt.subplots(figsize=(9, 4.2))
    # tree roots = classes with subclasses and no parent; type classes = the other parentless classes, whose
    # children are individuals (substance <- materials, crystal structure <- crystals, ...)
    has_child = {pa for _, pa in lab["tbox"]}
    group = {c: ("tree roots" if c in has_child else "type classes") if lab["class_level"][c] == 0 else
             ("families / roles" if lab["class_level"][c] == 1 else "leaf families") for c in lab["classes"]}
    levels = ["tree roots", "families / roles", "leaf families", "type classes", "individuals: elements",
              "individuals: materials", "individuals: crystals"]
    names = ["tree roots\n(structure family,\nbattery role)", "families /\nroles", "leaf\nfamilies",
             "type classes\n(substance, crystal\nstructure, element…)", "elements", "materials", "crystals"]
    for off, t, col in ((-0.17, "base", "#9aa0aa"), (0.17, "kont", "#a4123f")):
        cls = lab["classes"]
        per = rep[t]["depth_per_class"]
        for x, lvname in enumerate(levels):
            if not lvname.startswith("individuals"):
                v = [per[c] for c in cls if group[c] == lvname]
            else:
                key = {"individuals: elements": "E", "individuals: materials": "M", "individuals: crystals": "X"}[lvname]
                v = m[t]["d0_" + key].tolist()
            ax.boxplot([v], positions=[x + off], widths=0.28, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=col, color=col), medianprops=dict(color="white"),
                       whiskerprops=dict(color=col), capprops=dict(color=col))
    ax.set_xticks(range(len(names)), names, fontsize=7)
    ax.set_ylabel("hyperbolic distance from the origin")
    ax.plot([], [], color="#9aa0aa", lw=6, label="before (OnT-galen)")
    ax.plot([], [], color="#a4123f", lw=6, label="after (K-Ont)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("General classes sit near the centre, specific ones further out, individuals at the edge", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "fig_depth.png"), dpi=220)

    # confusion
    cf = rep["kont"]["branch_heldout"]["family_confusion"]
    keep = [i for i, row in enumerate(cf["matrix"]) if sum(row)]
    M = np.array(cf["matrix"])[np.ix_(keep, keep)]
    names = [FAM_NAME[cf["labels"][i]] for i in keep]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(M, cmap="Reds")
    for i in range(len(keep)):
        for j in range(len(keep)):
            if M[i, j]:
                ax.text(j, i, M[i, j], ha="center", va="center", fontsize=8, color="white" if M[i, j] > 15 else "#1d2129")
    ax.set_xticks(range(len(keep)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(keep)), names, fontsize=8)
    ax.set_xlabel("predicted family (K-Ont)"), ax.set_ylabel("true family")
    acc = rep["kont"]["branch_heldout"]["family_via_relation_rotation"]["acc"]
    ax.set_title(f"86 held-out crystals, no family line in any sentence: {acc:.1%}", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "fig_family_confusion.png"), dpi=220)
    print("wrote figures to", D)


if __name__ == "__main__":
    main()
