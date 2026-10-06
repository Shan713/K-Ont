import sys, json, random, torch
sys.path.insert(0, "OnT")
from ont.model import OntologyTransformer
e = json.load(open("data/battgpt_abox/entities.json", encoding="utf-8"))
vm = json.load(open("data/battgpt_abox/verbalizations_materials_full.json", encoding="utf-8"))
vc = json.load(open("data/battgpt_abox/verbalizations_crystals_full.json", encoding="utf-8"))
sp = json.load(open("data/split.json")); mats, crys = sp["test_materials"], sp["test_crystals"]
true = torch.tensor([crys.index(e["materials"][m]["structure"]) for m in mats])
perm = list(range(len(crys))); random.Random(0).shuffle(perm)
def swap(pre):
    out = []
    for i, c in enumerate(crys):
        donor = [l for l in vc[crys[perm[i]]].split("\n") if l.startswith(pre)]
        out.append("\n".join([l for l in vc[c].split("\n") if not l.startswith(pre)] + donor))
    return out
def mrr(m, M_txt, C_txt, how):
    with torch.no_grad():
        M = m.encode(M_txt, convert_to_tensor=True).float().cpu()
        if how == "rotation": C = torch.tensor(m.encode_existence(["has structure"] * len(C_txt), C_txt))
        else: C = m.encode(["has structure some " + c for c in C_txt], convert_to_tensor=True).float().cpu()
        s = -m.manifold.dist(M.unsqueeze(1), C.unsqueeze(0))
    t = s.gather(1, true.view(-1, 1)); r = (s > t).sum(1) + 1 + ((s == t).sum(1) - 1) / 2
    return round((1 / r.float()).mean().item(), 3)
M_txt = [vm[m] for m in mats]
M_nogeo = ["\n".join(l for l in vm[m].split("\n") if not l.startswith(("Lattice:", "Volume:", "Density:", "Sites:"))) for m in mats]
print(f"{'model':22}{'scoring':10}{'as is':>7}{'Sites shuf':>12}{'Latt+Vol shuf':>15}{'mat geo removed':>17}")
for name, run in (("M1 ours", "full_split_inbatch"), ("M2 his", "full_split_isa"), ("M3 both", "full_split_both")):
    m = OntologyTransformer.from_pretrained(f"data/runs/{run}/final"); m.hit_model.eval()
    for how in ("rotation", "text"):
        C = [vc[c] for c in crys]
        print(f"{name:22}{how:10}{mrr(m, M_txt, C, how):>7}{mrr(m, M_txt, swap(('Sites:',)), how):>12}{mrr(m, M_txt, swap(('Lattice:','Volume:')), how):>15}{mrr(m, M_nogeo, C, how):>17}")
