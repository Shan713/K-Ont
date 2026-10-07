"""Phase C, step 2: does a predicted space group help CrystaLLM find the right structure?

For CrystaLLM test compositions it never trained on (data/crystallm/phaseC_test_with_pred.json),
generate CIFs under four prompt conditions, the same number of samples each:

  composition   data_<cell composition>                      (vanilla CrystaLLM)
  rf_top1       + the random forest's top-1 space group
  rf_top5       + its top-5 space groups, samples split evenly
  oracle        + the true space group                       (upper bound)

Each sample is post-processed as CrystaLLM's own bin/postprocess.py does, parsed with pymatgen and
compared with the true structure by StructureMatcher (CrystaLLM benchmark tolerances: ltol 0.2,
stol 0.3, angle_tol 5). Run with the CrystaLLM venv; the crystallm package is imported from the
authors' repo cloned at K-Ont/CrystaLLM (gitignored; docs/WORKSTATION_SETUP.md).

    .venv-crystallm/Scripts/python.exe abox/phaseC_generate.py --n-materials 100 --samples 10
"""
import argparse
import json
import os
import random
import re
import sys
import time
import warnings

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRYSTALLM = os.path.join(HERE, "CrystaLLM")  # github.com/lantunes/CrystaLLM, sparse clone of crystallm/
sys.path.insert(0, CRYSTALLM)
warnings.filterwarnings("ignore")

from pymatgen.analysis.structure_matcher import StructureMatcher  # noqa: E402
from pymatgen.core import Composition, Structure  # noqa: E402
from pymatgen.symmetry.groups import SpaceGroup  # noqa: E402
from pymatgen.core.operations import SymmOp  # noqa: E402

# CrystaLLM's replace_symmetry_operators() calls SymmOp.as_xyz_string(), which current pymatgen
# renamed to as_xyz_str(); without this alias every generated CIF fails post-processing (0 valid).
if not hasattr(SymmOp, "as_xyz_string"):
    SymmOp.as_xyz_string = SymmOp.as_xyz_str

from crystallm import (CIFTokenizer, GPT, GPTConfig, extract_space_group_symbol,  # noqa: E402
                       get_atomic_props_block_for_formula, remove_atom_props_block,
                       replace_symmetry_operators)

DATA = os.path.join(HERE, "data", "crystallm")
ALLOWED_SG = set(open(os.path.join(CRYSTALLM, "crystallm", "spacegroups.txt")).read().split())


def sg_symbol(number):
    sym = SpaceGroup.from_int_number(number).symbol
    # current pymatgen writes monoclinic groups in full form (P12_1/c1, C12/m1, P121); CrystaLLM's
    # vocabulary has the short form (P2_1/c, C2/m, P2), as in its training CIFs
    m = re.fullmatch(r"([PC])1(.+)1", sym)
    if sym not in ALLOWED_SG and m and m.group(1) + m.group(2) in ALLOWED_SG:
        sym = m.group(1) + m.group(2)
    return sym if sym in ALLOWED_SG else None


def prompt_for(cell, sg=None):
    comp_str = Composition(cell).formula.replace(" ", "")
    if sg is None:
        return f"data_{comp_str}\n"
    block = get_atomic_props_block_for_formula(comp_str)
    s = f"data_{comp_str}\n{block}\n_symmetry_space_group_name_H-M {sg}\n"
    return "\n".join(l.strip() for l in s.split("\n"))


def load_model(device, name="crystallm_v1_small"):
    ckpt = torch.load(os.path.join(DATA, name, "ckpt.pt"), map_location=device)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd)
    return model.eval().to(device)


@torch.no_grad()
def sample_batch(model, tok, prompt, n, device, max_new_tokens=1000, temperature=0.8, top_k=10):
    """Same sampling as GPT.generate (temperature, top-k, stop at a blank line), for n sequences at once."""
    ids = torch.tensor(tok.encode(tok.tokenize_cif(prompt)), dtype=torch.long, device=device)
    x = ids[None, :].repeat(n, 1)
    nl = tok.token_to_id["\n"]
    done = torch.zeros(n, dtype=torch.bool, device=device)
    prev = torch.full((n,), -1, device=device)
    bs = model.config.block_size
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(max_new_tokens):
            logits, _ = model(x if x.size(1) <= bs else x[:, -bs:])
            logits = logits[:, -1, :].float() / temperature
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(1)
            nxt = torch.where(done, torch.full_like(nxt, nl), nxt)
            x = torch.cat([x, nxt[:, None]], dim=1)
            done |= (prev == nl) & (nxt == nl)
            prev = nxt
            if done.all():
                break
    return [tok.decode(row.tolist()).split("\n\n")[0] + "\n" for row in x]


def load_truth(t):
    """The reference structure. CrystaLLM's test-set CIFs are stored like its training data: the
    asymmetric unit only, with a placeholder 'x, y, z' symmetry operator. They must go through the
    same postprocess() as generated CIFs, or the reference is missing most of its atoms (Li2ZnSiO4:
    8 sites instead of 32) and almost nothing can match -- the bug behind the first Phase C runs.
    (Fixed 2026-10-07; every earlier match count compared against incomplete references.)
    Sanity-set entries carry a full Materials Project CIF and are used as they are."""
    if t.get("cif"):
        return Structure.from_str(t["cif"], fmt="cif")
    with open(os.path.join(DATA, "test_cifs", t["id"] + ".cif"), encoding="utf-8") as f:
        # pymatgen-written test CIFs indent their lines ("  1  'x, y, z'"); CrystaLLM's
        # replace_symmetry_operators() only recognises the compact layout the model itself writes,
        # so collapse whitespace first (as CrystaLLM's own prompt builder does)
        text = "\n".join(re.sub(r"[ \t]+", " ", line.strip()) for line in f.read().splitlines())
    return Structure.from_str(postprocess(text), fmt="cif")


def postprocess(cif):
    sg = extract_space_group_symbol(cif)
    if sg is not None and sg != "P 1":
        cif = replace_symmetry_operators(cif, sg)
    return remove_atom_props_block(cif)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-materials", type=int, default=100, help="all cathode-like + random others up to this")
    ap.add_argument("--samples", type=int, default=10, help="samples per condition per material")
    ap.add_argument("--out", default=os.path.join(DATA, "phaseC_results.json"))
    ap.add_argument("--model", default="crystallm_v1_small", help="checkpoint folder under data/crystallm")
    ap.add_argument("--conditions", default="composition,rf_top1,rf_top5,oracle", help="comma-separated subset")
    ap.add_argument("--gen-dir", default="phaseC_gen", help="folder under data/crystallm for the generated CIFs")
    ap.add_argument("--seed", type=int, default=1337, help="base seed; each material x prompt gets its own fixed seed")
    ap.add_argument("--candidates", default="phaseC_test_with_pred.json",
                    help="candidate file under data/crystallm; an entry with a 'cif' field is its own reference")
    ap.add_argument("--start", type=int, default=0, help="skip the first N chosen materials (resume / extend a run)")
    a = ap.parse_args()
    device = "cuda"
    test = json.load(open(os.path.join(DATA, a.candidates), encoding="utf-8"))
    cath = [t for t in test if t.get("cathode_like")]
    rest = [t for t in test if not t.get("cathode_like")]
    random.Random(0).shuffle(rest)
    chosen = (cath + rest)[a.start: a.n_materials]
    model, tok = load_model(device, a.model), CIFTokenizer()
    keep = a.conditions.split(",")
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    results, t0 = [], time.time()
    for i, t in enumerate(chosen):
        truth = load_truth(t)
        top5 = [s for s in (sg_symbol(n) for n in t.get("sg_pred_top5", [])) if s]
        conds = {"composition": [(None, a.samples)],
                 "rf_top1": [(top5[0], a.samples)] if top5 else [],
                 "rf_top5": [(s, a.samples // len(top5) + (k < a.samples % len(top5))) for k, s in enumerate(top5)] if top5 else [],
                 "oracle": [(sg_symbol(t["sg"]), a.samples)] if sg_symbol(t["sg"]) else []}
        rec = {"id": t["id"], "cell": t["cell"], "formula": t["formula"], "sg": t["sg"],
               "cathode_like": t.get("cathode_like", False), "sg_pred_top5": t.get("sg_pred_top5", []), "conditions": {}}
        conds = {c: plan for c, plan in conds.items() if c in keep}
        os.makedirs(os.path.join(DATA, a.gen_dir, t["id"]), exist_ok=True)
        for cond, plan in conds.items():
            # fixed per material and prompt, so a run is reproducible and independent of which other
            # prompts or materials are included
            torch.manual_seed(a.seed + sum(map(ord, t["id"] + cond)))
            valid = match = 0
            cifs = []
            for sg, n in plan:
                if n:
                    cifs += sample_batch(model, tok, prompt_for(t["cell"], sg), n, device)
            for k, raw in enumerate(cifs):
                with open(os.path.join(DATA, a.gen_dir, t["id"], f"{cond}_{k}.cif"), "w") as f:
                    f.write(raw)
                try:
                    s = Structure.from_str(postprocess(raw), fmt="cif")
                    valid += 1
                    match += bool(matcher.fit(s, truth))
                except Exception:
                    pass
            rec["conditions"][cond] = {"n": len(cifs), "valid": valid, "matches": match}
        results.append(rec)
        el = time.time() - t0
        print(f"{i + 1}/{len(chosen)} {t['formula']} sg {t['sg']}: " +
              " | ".join(f"{c} {v['matches']}/{v['n']}" for c, v in rec["conditions"].items()) +
              f"  ({el / 60:.1f} min)", flush=True)
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
