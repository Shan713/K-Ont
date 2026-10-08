"""Task B1: the KG's materials as a CrystaLLM-format training corpus, with KG labels.

From data/crystallm/kg_training_source.json.gz (5,000 KG materials with a formula-level split): each
structure is put in its conventional standard cell, written as a symmetrized CIF by pymatgen
(CifWriter, symprec 0.1 -- how CrystaLLM's own corpus was made) and passed through CrystaLLM's own
pre-processing (bin/preprocess.py's steps: non-reduced formula in data_, semi-symmetrized operators,
atomic-properties block, numbers rounded to 4 decimals). Each result must tokenize with CrystaLLM's
CIFTokenizer and fit the model's context (block_size of the checkpoint, 1024). Labels kept per entry:
battery_role, structure_family, space group, e_above_hull. "excluded" materials are never written.

Output: data/crystallm/kg_corpus_{train,val,test}.json.gz and kg_corpus_stats.json (counts, drop reasons).

    python abox/kg_corpus.py --workers 12
"""
import argparse
import gzip
import json
import os
import sys
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
warnings.filterwarnings("ignore")

import phaseC_generate as gen  # noqa: E402  (imports crystallm + the pymatgen alias)

DATA = gen.DATA


def convert(item):
    from crystallm import (CIFTokenizer, add_atomic_props_block, extract_formula_units,
                           replace_data_formula_with_nonreduced_formula, round_numbers, semisymmetrize_cif)
    from pymatgen.core import Structure
    from pymatgen.io.cif import CifWriter
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    x, block_size = item
    try:
        s = Structure.from_str(x["cif"], fmt="cif")
        conv = SpacegroupAnalyzer(s, symprec=0.1).get_conventional_standard_structure()
        cif = str(CifWriter(conv, symprec=0.1))
    except Exception as e:
        return x["id"], None, f"structure/symmetry: {e.__class__.__name__}"
    try:
        if extract_formula_units(cif) == 0:
            return x["id"], None, "formula units = 0"
        cif = replace_data_formula_with_nonreduced_formula(cif)
        cif = semisymmetrize_cif(cif)
        cif = add_atomic_props_block(cif, False)
        cif = round_numbers(cif, decimal_places=4)
    except Exception as e:
        return x["id"], None, f"preprocess: {e.__class__.__name__}"
    try:
        n = len(CIFTokenizer().tokenize_cif(cif))
    except Exception as e:
        return x["id"], None, f"tokenize: {e.__class__.__name__}"
    if n > block_size:
        return x["id"], None, "too long for the context"
    # the CIF must rebuild (as a generated one would) to its own cell: for ~1.5% the symmetry expansion
    # duplicates sites (mp-1296443 Li4Fe3CoO8 -> Li12Fe9Co3O32), which would teach the model wrong CIFs
    try:
        if gen.reference_from_text(cif).composition != conv.composition:
            return x["id"], None, "does not rebuild to its own cell"
    except Exception as e:
        return x["id"], None, f"rebuild: {e.__class__.__name__}"
    keep = {k: x[k] for k in ("id", "formula", "split", "space_group", "crystal_system", "structure_family",
                               "battery_role", "e_above_hull", "band_gap")}
    return x["id"], {**keep, "n_atoms": len(conv), "n_tokens": n, "cif": cif}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--model", default="crystallm_v1_small", help="checkpoint whose block_size limits length")
    a = ap.parse_args()
    import torch
    block_size = torch.load(os.path.join(DATA, a.model, "ckpt.pt"), map_location="cpu")["model_args"]["block_size"]
    src = json.load(gzip.open(os.path.join(DATA, "kg_training_source.json.gz"), "rt", encoding="utf-8"))
    src = [x for x in src if x["split"] != "excluded"]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(convert, [(x, block_size) for x in src], chunksize=8))
    by_split, drops = {"train": [], "val": [], "test": []}, Counter()
    for _, entry, reason in res:
        if entry:
            by_split[entry["split"]].append(entry)
        else:
            drops[reason] += 1
    for split, rows in by_split.items():
        with gzip.open(os.path.join(DATA, f"kg_corpus_{split}.json.gz"), "wt", encoding="utf-8") as f:
            json.dump(rows, f)
    stats = {"block_size": block_size, "input": len(src), "kept": {k: len(v) for k, v in by_split.items()},
             "dropped": dict(drops),
             "labels_in_train": {"battery_role": dict(Counter(r["battery_role"] for r in by_split["train"])),
                                 "structure_family": dict(Counter(r["structure_family"] for r in by_split["train"]))},
             "tokens_median": sorted(r["n_tokens"] for v in by_split.values() for r in v)[len([1 for v in by_split.values() for _ in v]) // 2]}
    json.dump(stats, open(os.path.join(DATA, "kg_corpus_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
