"""Phase 6: export K-Ont embeddings for crystal generation (ONT_ABOX_EXTENSION.md §10).

Encodes, with a trained K-Ont model, every material, crystal, element and ontology class of a KG, and
writes them as plain arrays for the generator side (which runs in another venv).

Generation conditioning must not hand the generator its answer, so the material vector used for it is
the encoding of the material sentence WITHOUT its structure lines (space group, crystal system, family,
lattice, volume, density, sites -- phase5_checks.STRUCT_LINES): formula, chemical system, flags and
properties only. The full sentence is exported too, for analysis only.

Vectors live in the Poincaré ball; for a Euclidean consumer they are also given in the tangent space at
the origin (logmap0), which is what a linear projection should read.

Output (data/kont_export/<name>/):
  embeddings.npz   material_cond, material_full, crystal, element, concept  (ball points)
                   + *_tan (logmap0 of each)
  id_maps.json     row order per type: material ids, formulas, crystal ids, element symbols, concept names
  meta.json        run, abox dir, dims, curvature, which lines were stripped

    .venv/Scripts/python abox/export_embeddings.py --run data/runs/full_grow5000_v2 \
        --abox-dir data/battgpt_abox_grow_5000_v2 --name grow5000_v2
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "OnT"))
sys.path.insert(0, os.path.join(HERE, "abox"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from ont.model import OntologyTransformer  # noqa: E402

from phase5_checks import STRUCT_LINES  # noqa: E402


def anonymise_sites(sentence):
    """'Sites: 28 (Li:4, P:4, O:16, Fe:4)' -> 'Sites: 28 (A:4, B:4, C:4, D:16)': the site pattern of the
    framework without its chemistry, so a vector of an analogue crystal cannot pass on its elements."""
    out = []
    for line in sentence.split("\n"):
        if line.startswith("Sites:") and "(" in line:
            head, body = line.split("(", 1)
            counts = sorted(int(x.split(":")[1]) for x in body.rstrip(")").split(", "))
            line = head + "(" + ", ".join(f"{chr(65 + i)}:{n}" for i, n in enumerate(counts)) + ")"
        out.append(line)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--abox-dir", required=True)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    out = os.path.join(HERE, "data", "kont_export", a.name)
    os.makedirs(out, exist_ok=True)
    load = lambda f: json.load(open(os.path.join(a.abox_dir, f), encoding="utf-8"))
    ents = load("entities.json")
    vm, vc, ve = load(f"verbalizations_materials_{a.variant}.json"), load(f"verbalizations_crystals_{a.variant}.json"), \
        load("verbalizations_elements.json")
    concepts = [v for _, v in sorted(json.load(open(os.path.join(a.run, "data", "concept_names.json"))).items(),
                                     key=lambda kv: int(kv[0]))]
    model = OntologyTransformer.from_pretrained(os.path.join(a.run, "final"))
    model.hit_model.eval()
    man = model.manifold

    def enc(sents):
        with torch.no_grad():
            return model.encode(sents, batch_size=64, convert_to_tensor=True, show_progress_bar=False).float()

    mats = sorted(vm)
    cond_sents = ["\n".join(l for l in vm[m].split("\n") if not l.startswith(STRUCT_LINES)) for m in mats]
    crys, els = sorted(vc), sorted(ve)
    arrays = {"material_cond": enc(cond_sents), "material_full": enc([vm[m] for m in mats]),
              "crystal": enc([vc[c] for c in crys]), "element": enc([ve[e] for e in els]), "concept": enc(concepts),
              # framework only: the Sites line with element symbols replaced by A, B, C... (Step 38)
              "crystal_anon": enc([anonymise_sites(vc[c]) for c in crys])}
    out_arr = {}
    for k, v in arrays.items():
        out_arr[k] = v.cpu().numpy()
        out_arr[k + "_tan"] = man.logmap0(v).cpu().numpy()
    np.savez_compressed(os.path.join(out, "embeddings.npz"), **out_arr)
    id_maps = {"material": [m.split("/")[-1] for m in mats],
               "material_formula": [ents["materials"][m]["formula"] for m in mats],
               "crystal": [c.split("/")[-1] for c in crys],
               "element": [ents["elements"][e]["symbol"] for e in els],
               "concept": concepts}
    json.dump(id_maps, open(os.path.join(out, "id_maps.json"), "w", encoding="utf-8"), indent=1)
    meta = {"run": a.run, "abox_dir": a.abox_dir, "variant": a.variant, "dim": int(out_arr["concept"].shape[1]),
            "curvature": float(man.c), "material_cond_stripped_lines": list(STRUCT_LINES),
            "counts": {k: int(v.shape[0]) for k, v in out_arr.items() if not k.endswith("_tan")},
            "example_cond_sentence": cond_sents[0],
            "tangent_norm_median": {k: float(np.median(np.linalg.norm(out_arr[k + "_tan"], axis=1))) for k in arrays}}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(meta, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
