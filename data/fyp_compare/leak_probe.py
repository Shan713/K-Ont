import sys, json
sys.path.insert(0, r"C:\Users\Shantharam P\Documents\GitHub\FYP_K-OnT\OnT")
from ont.data.train_abox import hasstructure_ranking, electrode_role_report, type_report, BASE_MODEL
from ont.hit import HierarchyTransformer
from ont.model import OntologyTransformer
D = r"C:\Users\Shantharam P\Documents\GitHub\K-Ont\data\fyp_compare\abox"
rep = json.load(open(D + r"\phase5b_report.json"))
ents = json.load(open(D + r"\entities.json", encoding="utf-8"))["materials"]
ids = [r["mp_id"] for r in ents]
verbal = json.load(open(D + r"\verbalizations.json", encoding="utf-8"))
crystals = json.load(open(D + r"\crystal_verbalizations.json", encoding="utf-8"))
role = {r["mp_id"]: r.get("battery_role") for r in ents}
counts = json.load(open(D + r"\merge_counts.json"))
import random
held = sorted(random.Random(8888).sample(ids, k=51))
assert len(held) == rep["n_heldout"] and held[:20] == counts["heldout_ids_sample"], "held-out mismatch"
print("held-out n =", len(held), "| example crystal sentence:", repr(crystals[held[0]]))
strip = lambda d, pre: {k: "\n".join(l for l in v.split("\n") if not l.startswith(pre)) for k, v in d.items()}
trained = OntologyTransformer.from_pretrained(D + r"\train_out\final")
base = OntologyTransformer(HierarchyTransformer.from_pretrained(BASE_MODEL, device="cpu"))
for name, m in (("UNTRAINED MiniLM", base), ("his TRAINED model", trained)):
    hs = hasstructure_ranking(m, verbal, crystals, held, ids)
    hs_s = hasstructure_ranking(m, strip(verbal, ("Structure:",)), crystals, held, ids)
    er = electrode_role_report(m, verbal, role, held)
    er_s = electrode_role_report(m, strip(verbal, ("Battery role:",)), role, held)
    print(f"\n{name}")
    print(f"  hasStructure MRR (pool 50, random {hs['random_mrr_baseline']:.3f}):  as-is {hs['mrr']:.3f}  |  'Structure:' line removed {hs_s['mrr']:.3f}")
    print(f"  electrode role acc (n={er['n']}, random 0.25):          as-is {er['accuracy']:.3f}  |  'Battery role:' line removed {er_s['accuracy']:.3f}")
