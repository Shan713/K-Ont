"""DFT hand-off: the novel compositions where a K-Ont template beat CrystaLLM-large (Step 44), as CIFs.

For each of the 34 compositions whose best K-Ont template (top 3, CHGNet-relaxed) is > 5 meV/atom lower than
CrystaLLM-large's best of 10: the winning template is re-built and re-relaxed exactly as in
abox/novel_template.py (same substitution, fmax 0.1, 300 steps) and written as a CIF, next to CrystaLLM-large's
best relaxed structure for the same composition (data/crystallm/novel_best.tar.gz, Step 28). A CSV lists
formula, parent, framework source, CHGNet energies (eV/atom), space groups and the energy difference; a
README says what the files are and what the DFT check should answer.

    .venv-crystallm/Scripts/python.exe abox/novel_dft_export.py  ->  ~/Downloads/K-Ont_novel_cathodes_for_DFT.zip
"""
import csv
import io
import json
import os
import sys
import tarfile
import warnings
import zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")
DATA = os.path.join(HERE, "data", "crystallm")
OUT = os.path.join(os.path.expanduser("~"), "Downloads", "K-Ont_novel_cathodes_for_DFT.zip")

README = """K-Ont novel cathode candidates for DFT verification
====================================================

What this is
------------
34 new cathode compositions (not in Materials Project) for which a structure built by the K-Ont-guided
template generator reached a LOWER CHGNet energy (by > 5 meV/atom) than the best of 10 structures generated
by CrystaLLM-large for the same composition. Energies so far are from the CHGNet machine-learned potential
(v0.3.0) only, not DFT.

Files
-----
kont_template/<ID>_<formula>.cif   structure from the K-Ont template generator, CHGNet-relaxed
                                   (fmax 0.1 eV/A, 300 steps, cell free)
crystallm/<ID>_<formula>.cif       CrystaLLM-large's lowest-energy structure for the same composition,
                                   CHGNet-relaxed with the same settings
candidates.csv                     per composition: formula, cell, parent material, the KG framework the
                                   template came from, CHGNet energies (eV/atom) of both structures, their
                                   difference (meV/atom, negative = template lower), space groups after
                                   relaxation, theoretical capacity (upper bound)

What the DFT check should answer
--------------------------------
1. Relax both structures of each composition with DFT (same functional and settings, e.g. PBE(+U) as in
   Materials Project) and compare total energies per atom: is the K-Ont template structure still lower than
   the CrystaLLM structure? (the key question for the paper)
2. If possible, energy above the convex hull of the template structures (MP-compatible settings), to see
   which candidates are plausibly stable (<= ~50 meV/atom).
Even a subset (e.g. the 10 with the largest CHGNet gap, first rows of candidates.csv) is useful.

Provenance
----------
K-Ont repository, BUILD_LOG Step 44; abox/novel_template.py and abox/novel_dft_export.py.
"""


def main():
    from pymatgen.core import Composition
    from pymatgen.io.cif import CifWriter
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    import novel_template as nt
    import kg_pool
    _, rows = kg_pool.load()
    nv = json.load(open(os.path.join(DATA, "novel_template.json")))
    novel = json.load(open(nt.POOL))
    cands = {n["id"]: n for n in json.load(open(os.path.join(DATA, "novel_candidates.json")))}
    best, cl, plan, en = nv["best"], nv["crystallm"], nv["plan"], nv["energies"]
    wins = sorted((best["kont3"][i] - cl[i], i) for i in best["kont3"]
                  if best["kont3"][i] is not None and i in cl and best["kont3"][i] - cl[i] < -nt.TIE)
    print(len(wins), "compositions", flush=True)
    nt._init(1)
    tar = tarfile.open(os.path.join(DATA, "novel_best.tar.gz"))
    zf = zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED)
    out_rows = []
    for d, nid in wins:
        f = novel[nid]["formula"]
        c = min((c for c in plan[nid]["kont3"] if en.get(f"{nid}|{c}") is not None), key=lambda c: en[f"{nid}|{c}"])
        s, e = rebuild(nt, f, rows[c])
        name = f"{nid}_{Composition(f).reduced_formula}.cif"
        zf.writestr(f"kont_template/{name}", str(CifWriter(s, symprec=0.1)))
        cm = tar.extractfile(f"novel_best/{nid}.cif").read().decode()
        zf.writestr(f"crystallm/{name}", cm)
        sg = lambda st: SpacegroupAnalyzer(st, symprec=0.1).get_space_group_symbol()
        from pymatgen.core import Structure
        out_rows.append([nid, f, novel[nid]["cell"], cands[nid]["parent_formula"], cands[nid]["parent_id"].split("/")[-1],
                         f'{rows[c]["formula"]} ({c})', round(e, 5), round(cl[nid], 5), round(1000 * (e - cl[nid]), 1),
                         round(1000 * d, 1), sg(s), sg(Structure.from_str(cm, fmt="cif")),
                         cands[nid].get("theoretical_capacity_mAh_g")])
        print(f"  {nid} {f}: template {e:.4f} vs CrystaLLM {cl[nid]:.4f} eV/atom", flush=True)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "formula", "cell", "parent_formula", "parent_mp_id", "template_framework_from",
                "chgnet_e_template_eV_atom", "chgnet_e_crystallm_eV_atom", "delta_meV_atom_rerelaxed",
                "delta_meV_atom_step44", "spacegroup_template", "spacegroup_crystallm", "theoretical_capacity_mAh_g"])
    w.writerows(out_rows)
    zf.writestr("candidates.csv", buf.getvalue())
    zf.writestr("README.txt", README)
    zf.close()
    print("wrote", OUT, os.path.getsize(OUT), "bytes")


def rebuild(nt, target_formula, c):
    """the same substitution + relaxation as novel_template._relax, returning the structure too"""
    import kg_condition2 as k2
    import phaseC_generate as gen
    from pymatgen.core import Element
    s = gen.reference_from_text(c["cif"])
    old = sum(n * float(Element(e.symbol).atomic_radius or 1.5) ** 3 for e, n in s.composition.items())
    s.replace_species({x.symbol: y.symbol for x, y in zip(k2._ordered(c["formula"]), k2._ordered(target_formula))})
    new = sum(n * float(Element(e.symbol).atomic_radius or 1.5) ** 3 for e, n in s.composition.items())
    s.scale_lattice(s.volume * new / old)
    r = nt._OPT.relax(s, fmax=0.1, steps=300, verbose=False)
    fs = r["final_structure"]
    return fs, float(nt._M.predict_structure(fs)["e"])


if __name__ == "__main__":
    main()
