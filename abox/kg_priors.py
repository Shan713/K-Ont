"""KG-steered generation, step 1: chemistry priors taken from the knowledge graph.

From the 5,000-material KG (data/battgpt_abox_grow_5000/entities.json):
  bonds          per element pair: median, robust spread (MAD), 1st/99th percentile and count of the
                 CrystalNN bond lengths the KG stores for every site
  coordination   per element: how often each coordination number occurs (from each site's
                 coordination geometry: Octahedral = 6, Tetrahedral = 4, ...)
  volume         per element: a volume contribution fitted by least squares so that a cell's volume
                 is predicted from its composition (sum of n_element * v_element)
Output: data/crystallm/kg_priors.json (small; committed). Run with the CrystaLLM venv (pymatgen).

    .venv-crystallm/Scripts/python.exe abox/kg_priors.py
"""
import json
import os
import re
import statistics
import warnings
from collections import Counter, defaultdict

import numpy as np
from pymatgen.core import Structure

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEOMETRY_CN = {"Linear": 2, "TrigonalPlanar": 3, "Tetrahedral": 4, "SquarePyramidal": 5, "Octahedral": 6,
               "PentagonalBipyramidal": 7, "SquareAntiprismatic": 8, "Cuboctahedral": 12}


def element_of(species_id):
    return re.match(r"species/([A-Z][a-z]?)", species_id).group(1)


def cn_of(geometry):
    if not geometry:
        return None
    if geometry in GEOMETRY_CN:
        return GEOMETRY_CN[geometry]
    m = re.search(r"(\d+)$", geometry)
    return int(m.group(1)) if m else None


def main():
    e = json.load(open(os.path.join(HERE, "data", "battgpt_abox_grow_5000", "entities.json"), encoding="utf-8"))
    sites = e["sites"]
    el = {sid: element_of(s["species"]) for sid, s in sites.items()}
    dist = defaultdict(list)
    cn = defaultdict(Counter)
    for sid, s in sites.items():
        a = el[sid]
        c = cn_of(s.get("coordination_geometry"))
        if c:
            cn[a][c] += 1
        for b in s.get("bonds", []):
            t = b.get("target_site")
            if t in el and b.get("distance_angstrom"):
                dist[tuple(sorted((a, el[t])))].append(float(b["distance_angstrom"]))
    bonds = {}
    for pair, d in dist.items():
        if len(d) < 5:
            continue
        med = statistics.median(d)
        mad = statistics.median(abs(x - med) for x in d)
        q = np.percentile(d, [1, 99])
        bonds["-".join(pair)] = {"median": round(med, 4), "mad": round(mad, 4), "p01": round(float(q[0]), 4),
                                 "p99": round(float(q[1]), 4), "n": len(d)}
    coord = {a: {str(k): v for k, v in sorted(c.items())} for a, c in cn.items()}

    rows, vols, elements = [], [], set()
    for c in e["crystals"].values():
        try:
            s = Structure.from_str(c["cif"], fmt="cif")
        except Exception:
            continue
        comp = s.composition.as_dict()
        rows.append(comp); vols.append(s.volume); elements |= set(comp)
    order = sorted(elements)
    A = np.array([[r.get(x, 0.0) for x in order] for r in rows])
    v, *_ = np.linalg.lstsq(A, np.array(vols), rcond=None)
    pred = A @ v
    rel = np.abs(pred - np.array(vols)) / np.array(vols)
    volume = {x: round(float(val), 4) for x, val in zip(order, v)}
    out = {"source": "data/battgpt_abox_grow_5000/entities.json", "n_sites": len(sites),
           "n_bonds": sum(len(d) for d in dist.values()), "bonds": bonds, "coordination": coord,
           "volume_per_atom": volume, "volume_fit_median_rel_error": round(float(np.median(rel)), 4)}
    json.dump(out, open(os.path.join(HERE, "data", "crystallm", "kg_priors.json"), "w"), indent=1)
    print(f"{out['n_bonds']} bonds -> {len(bonds)} element pairs; coordination for {len(coord)} elements; "
          f"volume model over {len(rows)} cells, median relative error {out['volume_fit_median_rel_error']}")
    for p in ("Li-O", "Mn-O", "Fe-O", "Co-O", "P-O"):
        print(" ", p, bonds.get(p))
    print("  Mn coordination:", coord.get("Mn"))


if __name__ == "__main__":
    main()
