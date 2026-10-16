"""Step 40 control: framework (crystal_anon) vectors from the PLAIN encoder K-Ont started from
(Hui97/OnT-MiniLM-L12-galen), same sentences, same row order and same tangent map as the K-Ont export,
so a generator trained on them differs from the K-Ont one only in where the vectors come from.

    .venv/Scripts/python.exe abox/export_base_crystal.py   ->  data/kont_export/grow5000_v2_base/
"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "OnT"))
sys.path.insert(0, os.path.join(HERE, "abox"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from ont.hit import HierarchyTransformer  # noqa: E402
from export_embeddings import anonymise_sites  # noqa: E402

SRC = os.path.join(HERE, "data", "kont_export", "grow5000_v2")
OUT = SRC + "_base"


def main():
    os.makedirs(OUT, exist_ok=True)
    shutil.copy(os.path.join(SRC, "id_maps.json"), OUT)
    vc = json.load(open(os.path.join(HERE, "data", "battgpt_abox_grow_5000_v2", "verbalizations_crystals_full.json"), encoding="utf-8"))
    crys = sorted(vc)
    ids = json.load(open(os.path.join(OUT, "id_maps.json")))["crystal"]
    assert [c.split("/")[-1] for c in crys] == ids
    m = HierarchyTransformer.from_pretrained("Hui97/OnT-MiniLM-L12-galen")
    m.eval()
    with torch.no_grad():
        v = m.encode([anonymise_sites(vc[c]) for c in crys], batch_size=64, convert_to_tensor=True, show_progress_bar=False).float()
        np.savez_compressed(os.path.join(OUT, "embeddings.npz"), crystal_anon=v.cpu().numpy(),
                            crystal_anon_tan=m.manifold.logmap0(v).cpu().numpy())
    print("base crystal_anon:", v.shape)


if __name__ == "__main__":
    main()
