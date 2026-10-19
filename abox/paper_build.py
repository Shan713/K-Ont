"""IEEE-format paper draft of the K-Ont work: LaTeX (IEEEtran, conference) + BibTeX + a two-column HTML/PDF preview,
all generated from ONE text (so the numbers in both are identical). Numbers are the BUILD_LOG Steps 36-55 results.

    .venv-crystallm/Scripts/python.exe abox/paper_build.py
    -> docs/paper/kont_ieee.tex, docs/paper/refs.bib, docs/paper/figs/*.png   (compile on Overleaf: pdflatex + bibtex)
    -> ~/Downloads/K-Ont IEEE paper draft (preview).pdf                        (read-now preview, not the camera-ready)
"""
import html
import os
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "abox"))
OUT = os.path.join(HERE, "docs", "paper")
FIGS = os.path.join(OUT, "figs")
PREVIEW = os.path.join(os.path.expanduser("~"), "Downloads", "K-Ont IEEE paper draft (preview).pdf")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

TITLE = ("K-Ont: Hyperbolic Embeddings of Ontologies and Their Knowledge-Graph Individuals, "
         "with Application to Ontology-Guided Crystal Structure Generation")
AUTHORS = ["Dr. Gowtham Ramesh", "Shantharam P", "Sarvesha Nandhakumar", "Akshay KS", "RD Tarun", "S Ajeth"]
AFFIL = "[Department], [Institution], [City], [Country]"  # TODO: fill in before submission
KEYWORDS = ["ontology embedding", "knowledge graph", "hyperbolic embedding", "crystal structure generation",
            "battery materials", "retrieval-augmented generation"]

ABSTRACT = (
    "Ontology embedding models place the concepts of an ontology in a geometric space where subsumption becomes "
    "measurable, but they do not represent the individuals of a knowledge graph (KG) that instantiate those "
    "concepts. We present **K-Ont**, an extension of the hyperbolic ontology encoder OnT to KG individuals: every "
    "material, crystal and element of a 5,000-material battery KG is verbalised and trained jointly with the "
    "ontology's axioms, typing facts and relation facts in a Poincar\u00e9 ball. On 1,018 held-out materials K-Ont "
    "types materials and crystals at 99.7-100%, predicts battery role (100% vs 84% majority) and structure family "
    "(98.8% vs 47%), and keeps OnT's quality on the GALEN benchmark; an ablation shows the ontology's axioms are "
    "what give type reasoning (64% without them). With a retention objective that protects KG sentences, original "
    "ontology concepts and relation phrases, a single K-Ont model additionally retrieves crystal frameworks for "
    "unseen compositions, stably across seeds. Used as a template retriever for crystal structure generation, "
    "K-Ont recovers the reference structure at the first attempt for 60.5% of 902 held-out materials versus 48.3% "
    "for a fine-tuned CrystaLLM and 52.1% for data-mined substitution; combined with CrystaLLM and a CHGNet "
    "energy check it commits to a single answer that is right 69.3% vs 63.7% of the time. On 200 novel cathode "
    "compositions, an independent energy-above-hull calculation ranks the K-Ont template below the CrystaLLM "
    "structure in 32 of the 34 cases tested. In contrast, injecting K-Ont knowledge into the language model -- as "
    "a soft prompt or a retrieved hard prompt -- adds at most 1.5 points, with a measured ceiling below 3 points.")

# ---------------------------------------------------------------- the paper body (one source for LaTeX and HTML)
# block kinds: ("sec", title) ("sub", title) ("p", text) ("ul", [items]) ("eq", latex, plain) ("table", dict) ("fig", dict)
# inline: **bold**, *italic*, [@key] or [@a,@b] citations, $...$ inline math (kept simple)
B = []
S = lambda t: B.append(("sec", t))
SS = lambda t: B.append(("sub", t))
P = lambda t: B.append(("p", t))
UL = lambda items: B.append(("ul", items))

S("Introduction")
P("Materials knowledge is increasingly curated as ontologies and knowledge graphs (KGs): battery ontologies such as "
  "BattINFO [@clark2022battinfo] fix a vocabulary of materials, structures and electrode roles, and pipelines "
  "populate KGs with facts drawn from databases such as the Materials Project [@jain2013mp]. Generative models for "
  "crystals [@xie2022cdvae, @jiao2023diffcsp, @antunes2024crystallm, @zeni2025mattergen], on the other hand, learn "
  "from structure files alone and do not use this curated knowledge.")
P("Ontology embeddings offer a bridge. Hyperbolic models embed hierarchies with low distortion "
  "[@nickel2017poincare]; language-model encoders such as HiT [@he2024hit] and OnT [@yang2025ont] embed the *names* "
  "of ontology concepts in a Poincar\u00e9 ball so that subsumption and existential restrictions become geometric. "
  "These encoders, however, represent concepts (the TBox) only; the individuals of a KG (the ABox) -- the actual "
  "materials and crystals one wants to reason about and generate -- are not embedded.")
P("This paper makes four contributions:")
UL(["**K-Ont**, an extension of OnT to KG individuals: individuals are verbalised and trained jointly with the "
    "ontology's axioms, typing facts and relation facts in one hyperbolic space (Sec. III).",
    "An evaluation on a 5,000-material battery KG with held-out materials, on OnT's GALEN benchmark, and on a "
    "second (medical) domain, including an ontology ablation and a retention objective that lets **one** K-Ont "
    "model hold the original ontology, a new domain ontology and KG, and a crystal-retrieval skill (Sec. V-A, V-B).",
    "**K-Ont-guided template generation**: K-Ont retrieves known crystal frameworks for a target composition, which "
    "are built directly by element substitution; it outperforms a fine-tuned crystal language model and the "
    "standard data-mined substitution method, and complements the language model on novel cathodes (Sec. V-C, V-E).",
    "A controlled negative result: injecting the same knowledge *into* a crystal language model, as a soft prompt "
    "or as a retrieved hard prompt, yields at most a small gain, and we measure why (Sec. V-D)."])

S("Related Work")
P("**Ontology and KG embeddings.** Translational KG embeddings [@bordes2013transe] model relations between "
  "entities but not the logical structure of an ontology. Poincar\u00e9 embeddings [@nickel2017poincare] showed "
  "that hyperbolic space suits hierarchies; HiT [@he2024hit] re-trains a sentence encoder [@reimers2019sbert, "
  "@wang2020minilm] so that its outputs live in a Poincar\u00e9 ball and encode subsumption, and OnT [@yang2025ont] "
  "extends this to EL ontologies with role rotations for existential restrictions, evaluated on GO, GALEN and "
  "ANATOMY. Ontology preprocessing commonly relies on DeepOnto [@he2024deeponto]. None of these embed ABox "
  "individuals, which is the gap K-Ont addresses.")
P("**Crystal structure generation.** Diffusion and variational models [@xie2022cdvae, @jiao2023diffcsp, "
  "@zeni2025mattergen] and autoregressive language models over CIF text [@antunes2024crystallm] generate "
  "structures from compositions or properties. Substitution-based prediction reuses known prototypes; Hautier et "
  "al. [@hautier2011] learn ion-substitution probabilities from data and remain a strong, widely used baseline. "
  "Retrieval-augmented generation [@lewis2020rag] and prompt or prefix tuning [@lester2021prompt, @li2021prefix] "
  "are the two standard ways to condition a language model on external knowledge; we test both.")

S("Method")
SS("Battery knowledge graph")
P("The KG was built with the battGPT pipeline from Materials Project data [@jain2013mp] and pymatgen analysis "
  "[@ong2013pymatgen]: 5,000 Li/Na materials ($\\le$0.15 eV/atom above hull, $\\le$40 sites, 2-5 elements, 527 "
  "polymorph groups), 5.6 M triples, linked to a battery ontology with structure families (e.g. olivine, NASICON, "
  "layered oxide) and electrode roles. Each material has a crystal, its elements, properties (band gap, formation "
  "energy, hull distance, moduli, magnetisation), and relations *has structure*, *has structure family* and "
  "*belongs to electrode*. Structure-family and role labels are assigned by rules over composition and space group.")
SS("K-Ont")
P("OnT maps the name $x$ of a concept to a point $f(x)$ in a Poincar\u00e9 ball $\\mathbb{B}_c$ and scores a "
  "subsumption $C \\sqsubseteq D$ with")
B.append(("eq", r"s(C,D) = -\big(d_c(f(C),f(D)) + \lambda\,(\lVert f(D)\rVert_c - \lVert f(C)\rVert_c)\big),",
          "s(C,D) = \u2212( d_c(f(C), f(D)) + \u03bb ( \u2016f(D)\u2016_c \u2212 \u2016f(C)\u2016_c ) ),"))
P("where $d_c$ is the hyperbolic distance and $\\lVert\\cdot\\rVert_c$ the distance to the origin; an "
  "existential restriction $\\exists r.D$ is embedded by a learned rotation of $f(D)$ conditioned on the role "
  "$r$ [@yang2025ont]. **K-Ont** keeps this model and objective and adds the ABox: each individual is verbalised "
  "as a short multi-line sentence (name, formula, chemical system, properties, space group, lattice, sites; label "
  "lines omitted), and three kinds of training rows are merged -- the ontology's TBox axioms (oversampled to 15% "
  "of type rows and 5% of relation rows), ABox typing rows (individual $\\sqsubseteq$ class, e.g. a material "
  "$\\sqsubseteq$ *substance*) and ABox relation rows (material $\\sqsubseteq \\exists$*has structure*.crystal). For "
  "relation rows we add an in-batch contrastive term so that an individual must prefer its own filler over "
  "another row's. Training starts from OnT-MiniLM-L12-galen (384-d), batch 32, 3 epochs.")
SS("Framework relation and retention")
P("To retrieve crystal templates, K-Ont is taught one more KG relation, *has the framework of*: for a training "
  "material, its structure-free sentence (no space group, lattice or volume; only what is known for a new "
  "composition) should score highest against pool materials of the same anonymous formula and space group, with "
  "candidates of the same formula type as negatives (listwise softmax). Learning this alone erases K-Ont's typing "
  "(catastrophic forgetting [@kirkpatrick2017ewc]), so we add a **retention** term: the mean-squared distance "
  "between the current and the original embeddings of (i) all 10,108 KG sentences, (ii) 20,000 GALEN concept names "
  "held at the original OnT-galen embeddings, and (iii) the 23 relation names and role/family label phrases that "
  "drive the role rotations, each as its own term in every step. We call the resulting single model K-Ont-52c.")
SS("Template generation and selection")
P("For a target composition, candidates are KG materials with the same anonymous formula and a different formula. "
  "A gradient-boosted ranker combines candidate popularity and cell size, the KG's property facts, and the K-Ont "
  "retrieval score (out-of-fold). The target's elements are placed on the top-$k$ candidates' sites (element order "
  "by amount, then electronegativity; lattice scaled by atomic radii) -- a template. For a single answer, templates "
  "and language-model samples are relaxed briefly with CHGNet [@deng2023chgnet] and the lowest energy per atom wins.")
SS("Injection baselines")
P("For comparison we condition CrystaLLM-small [@antunes2024crystallm] on the same knowledge: (a) a **soft prompt** "
  "-- the K-Ont embedding of the retrieved template's framework as 8 prefix tokens plus gated per-layer adapters "
  "[@li2021prefix], trained with 6,453 counterfactual structures (a training composition placed in a different "
  "framework) so that the vector must be read; and (b) a **hard prompt** -- the retrieved template's CIF header "
  "(space group, cell, volume, $Z$) as the prompt, from which the model writes the atom sites [@lewis2020rag].")

S("Experimental Setup")
P("K-Ont is trained on 3,982 KG materials and evaluated on 1,018 held-out materials (polymorph groups kept on one "
  "side). All crystal experiments use the 902 held-out materials that have at least one same-formula-type "
  "candidate; candidates, retrievers, rankers and generators see only K-Ont's training materials. A material's "
  "structure is *recovered* when a candidate matches the reference under pymatgen's StructureMatcher (length tol. "
  "0.2, site tol. 0.3, angle 5\u00b0); we report found@$k$. Novel compositions (200 cathode substitution children not "
  "in Materials Project, charge-balanced with SMACT [@davies2019smact]) are compared by the lowest CHGNet energy "
  "after relaxation. Differences are paired over materials with 95% bootstrap intervals [@efron1994bootstrap]; "
  "pass criteria for the injection experiments were fixed before the runs. Everything ran on one laptop RTX 4060 "
  "(8 GB).")

S("Results")
SS("K-Ont learns the ontology and the KG")
B.append(("table", {"label": "tab:checks", "caption": "K-Ont on 1,018 held-out materials (H@1; MRR for the hierarchy). "
          "Role/family: label lines removed from the sentence; majority-class baselines 84% / 47%.",
          "head": ["Model", "Mat. typing", "Cryst. typing", "Role", "Family", "TBox MRR"],
          "rows": [["OnT-galen (untrained on KG)", "0.000", "1.000", "0.67", "0.30", "0.28"],
                   ["**K-Ont**", "**1.000**", "**0.997**", "**1.000**", "**0.988**", "**0.969**"],
                   ["K-Ont without TBox axioms", "0.641", "0.639", "1.000", "0.977", "0.313"],
                   ["K-Ont-52c (one model, 3 seeds)", "0.92-0.95", "1.000", "0.99-1.00", "0.988", "0.969"]]}))
P("K-Ont generalises its typing and both ontology relations to unseen materials (Table~\\ref{tab:checks}). Removing "
  "the ontology's TBox axioms (keeping all KG facts) drops material and crystal typing to 64% and the hierarchy "
  "MRR from 0.97 to 0.31, while roles and families, which come from relation facts, survive: the ontology is what "
  "gives K-Ont its type reasoning. Because the KG's role and family labels are rule-based, near-perfect scores "
  "partly reflect learning those rules. On a second domain (303 patients of the UCI heart-disease data "
  "[@detrano1989heart] verbalised as KG individuals), K-Ont patient embeddings outperformed OnT's by +2.2 AUC on "
  "average over five classifiers (four significant) and by +4.3 when three columns were missing.")
SS("One model for GALEN, the battery KG and crystal retrieval")
B.append(("table", {"label": "tab:galen", "caption": "GALEN (OnT benchmark) and crystal retrieval. Pred./inf. = "
          "prediction / inference test sets, MRR and AUC; retrieval = template found@1 on 902 held-out materials.",
          "head": ["Model", "Pred. MRR / AUC", "Inf. MRR / AUC", "Retr. @1"],
          "rows": [["OnT-galen [@yang2025ont]", "0.339 / 0.978", "0.022 / 0.973", "--"],
                   ["K-Ont", "0.335 / 0.952", "0.011 / 0.918", "--"],
                   ["+ framework, KG retention", "0.333 / 0.955", "0.010 / 0.934", "55.0"],
                   ["+ GALEN retention", "0.341 / 0.976", "0.016 / 0.961", "55.5$^\\dagger$"],
                   ["**K-Ont-52c** (+ anchors)", "**0.334-0.339 / 0.975**", "**0.014 / 0.95**", "**54.2-54.8**"],
                   ["native relation, hard neg.", "0.324 / 0.960", "0.008 / 0.896", "54.4$^\\ddagger$"]]}))
P("Learning the battery KG costs part of OnT's GALEN inference ranking (Table~\\ref{tab:galen}). Retaining KG "
  "sentences alone keeps the battery checks; adding GALEN concepts restores GALEN but ($\\dagger$) collapses the "
  "role/family checks (0.025 / 0.000) until the relation phrases are retained as well. K-Ont-52c keeps every "
  "battery check of Table~\\ref{tab:checks}, matches OnT on GALEN prediction, recovers a third of the inference "
  "loss, and retrieves frameworks, on three seeds. Training the relation natively inside K-Ont's objective with "
  "hard negatives ($\\ddagger$) matches the retrieval but collapses typing (0.09 / 0.001), so a separate fine-tune "
  "with retention is the better recipe.")
SS("Template generation versus language-model generation")
B.append(("fig", {"label": "fig:foundk", "file": "foundk.png", "caption": "Structures recovered within $k$ "
          "attempts on 902 held-out materials: K-Ont-guided templates vs a fine-tuned CrystaLLM, a KG ranker without "
          "K-Ont, Hautier substitution and a composition rule."}))
B.append(("table", {"label": "tab:found", "caption": "Found@k (%) on 902 held-out materials, and single-answer "
          "accuracy (CHGNet lowest energy among 8 candidates, random 300 materials).",
          "head": ["Method", "@1", "@3", "@8"],
          "rows": [["CrystaLLM, fine-tuned [@antunes2024crystallm]", "48.3", "61.2", "68.3"],
                   ["Template, composition rule", "51.3", "55.8", "60.8"],
                   ["Template, Hautier [@hautier2011]", "52.1", "64.5", "70.1"],
                   ["Template, KG ranker w/o K-Ont", "56.3", "63.2", "68.4"],
                   ["**Template, K-Ont ranker**", "**60.5**", "**66.9**", "**71.4**"],
                   ["Template, plain-encoder ranker", "60.5", "67.6", "72.7"],
                   ["3 templates + 5 CrystaLLM", "--", "--", "**77.3**"],
                   ["Single answer: CrystaLLM x8", "63.7", "", ""],
                   ["Single answer: 3 templ. + 5 CrystaLLM", "**69.3**", "", ""]]}))
P("K-Ont-guided templates recover the reference structure at the first attempt +12.2 points [9.0, 15.4] more "
  "often than the fine-tuned CrystaLLM, +8.4 [5.5, 11.2] more than Hautier substitution and +9.2 more than the "
  "composition rule (Fig.~\\ref{fig:foundk}, Table~\\ref{tab:found}); CHGNet relaxation of the templates changes this by "
  "+0.3. Hautier's method catches up from five attempts. The two generators get different materials right: "
  "three templates plus five CrystaLLM samples reach 77.3% versus 68.3% for eight samples [+6.9, +11.1]. When one "
  "answer must be chosen by CHGNet energy, the combined pool is right 69.3% versus 63.7% for CrystaLLM choosing "
  "among its own eight samples (+5.7 [1.3, 9.7]); with this physics selection the template alone (61.7%) no longer "
  "beats CrystaLLM. For template choice itself, a plain encoder fine-tuned on the same KG relation is as good as "
  "K-Ont (60.5 vs 60.5 at $k=1$); K-Ont's starting weights help only with 5% of the training data (+3.4 on the "
  "space group). The distinct value of K-Ont is doing this within the same model as the ontology reasoning.")
SS("Injecting K-Ont knowledge into the language model")
B.append(("table", {"label": "tab:inj", "caption": "True-structure share (%) of generated samples, 902 held-out "
          "materials x 8 samples. Ceiling: the analogue with the true framework.",
          "head": ["Arm", "CrystaLLM", "From-scratch GPT"],
          "rows": [["Plain fine-tune (no K-Ont)", "**47.5**", "20.6"],
                   ["Soft prompt, K-Ont-ranked analogue", "44.1", "**22.1**"],
                   ["Soft prompt, random analogue", "22.7", "11.1"],
                   ["Soft prompt, true analogue (ceiling)", "49.6", "23.3"],
                   ["Hard prompt (template CIF header)", "45.9$^*$", "--"]]}))
P("A K-Ont soft prompt clearly steers the generated framework (random vs retrieved analogue: -21.4 points), but on "
  "CrystaLLM it never beats plain fine-tuning, and even the true analogue adds only +2.1: CrystaLLM was "
  "pre-trained on Materials Project and recalls these structures (Table~\\ref{tab:inj}). We therefore pre-trained a "
  "25 M-parameter GPT with CrystaLLM's architecture from scratch on 300,000 CrystaLLM training structures with every "
  "evaluation composition removed. Without memorisation the soft prompt helps (+1.5 [0.5, 2.5]), but the ceiling "
  "stays at +2.7: the vector identifies the framework while the model still writes every coordinate. A hard prompt "
  "with the retrieved template's header ($^*$found@1) is worse than using the template itself (60.5%). Knowledge "
  "is most useful outside the generator.")
SS("Novel cathode compositions")
B.append(("table", {"label": "tab:novel", "caption": "Novel cathodes (not in Materials Project). Top: lowest CHGNet "
          "energy per composition vs CrystaLLM-large's best of 10 (199 compositions, 5 meV/atom tie band). Bottom: "
          "independent energy-above-hull check of the 34 compositions where the template was lower.",
          "head": ["", "Lower", "Tie", "Higher"],
          "rows": [["3 K-Ont templates", "34", "126", "39"],
                   ["K-Ont templates + CrystaLLM-large", "**34**", "165", "**0**"],
                   ["Independent $E_{hull}$, K-Ont template", "**32**", "1", "1"]]}))
P("On compositions with no reference structure, three K-Ont templates are statistically level with CrystaLLM-large's "
  "best of ten (+1.3 meV/atom [-4.1, 6.4]); adding them to CrystaLLM finds a lower-energy structure for 34 of 199 "
  "compositions and a higher one for none (-7.5 meV/atom [-11.8, -4.0]) (Table~\\ref{tab:novel}). An independent "
  "energy-above-hull calculation [TODO: method, e.g. DFT functional and settings] of those 34 places the template "
  "lower in 32; median 0.025 vs 0.088 eV/atom, with 23 vs 13 within 50 meV/atom of the hull. These 34 were selected "
  "by CHGNet, so this confirms the ranking rather than an unbiased rate.")

S("Discussion and Limitations")
UL(["**Rule-based labels.** The KG's roles and families are rule-assigned; very high scores partly reflect the rules.",
    "**K-Ont versus a plain encoder.** For template choice alone, a plain encoder trained on the same KG relation "
    "is as good; K-Ont's contribution is a single model that also reasons over the ontology (whose axioms the "
    "ablation shows are necessary) and keeps the original ontology.",
    "**Memorisation.** Known-material benchmarks favour generators pre-trained on the same database; the from-"
    "scratch control and the novel-composition test address this, the latter only energetically.",
    "**Template coverage.** Templates cannot produce frameworks absent from the KG. Extending the KG with 5,267 "
    "further Materials Project materials (built in memory-safe chunks) raises the share of targets with a correct "
    "candidate from 85.6% to 90.9% and found@3 / found@8 by +3.4 [1.7, 5.2] / +3.4 [1.9, 5.1], with no retraining; "
    "the first pick does not improve.",
    "**Scope.** One crystal KG; costs of the one-model recipe are material typing (0.92-0.95) and part of GALEN "
    "inference."])

S("Conclusion")
P("K-Ont extends hyperbolic ontology embeddings from concepts to knowledge-graph individuals. It learns a battery "
  "ontology and KG, keeps its original ontology, and -- with a retention objective covering KG sentences, original "
  "concepts and relation phrases -- holds a crystal-retrieval skill in the same model. As a template retriever it "
  "outperforms a fine-tuned crystal language model and data-mined substitution, and it complements the language "
  "model on novel cathodes, whereas injecting the same knowledge into the language model helps little. Future work: "
  "larger and expert-curated KGs, DFT validation at scale, and generators that edit retrieved templates rather "
  "than write structures from scratch.")

BIB = r"""
@inproceedings{nickel2017poincare, author={Nickel, Maximilian and Kiela, Douwe}, title={Poincar{\'e} Embeddings for Learning Hierarchical Representations}, booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2017}}
@inproceedings{he2024hit, author={He, Yuan and Yuan, Zhangdie and Chen, Jiaoyan and Horrocks, Ian}, title={Language Models as Hierarchy Encoders}, booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2024}}
@misc{yang2025ont, author={Yang, Hui and Chen, Jiaoyan and others}, title={Language Models as Ontology Encoders}, howpublished={arXiv preprint}, year={2025}, note={TODO: verify full author list and arXiv id}}
@inproceedings{bordes2013transe, author={Bordes, Antoine and Usunier, Nicolas and Garcia-Duran, Alberto and Weston, Jason and Yakhnenko, Oksana}, title={Translating Embeddings for Modeling Multi-relational Data}, booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2013}}
@article{antunes2024crystallm, author={Antunes, Luis M. and Butler, Keith T. and Grau-Crespo, Ricardo}, title={Crystal Structure Generation with Autoregressive Large Language Modeling}, journal={Nature Communications}, volume={15}, pages={10570}, year={2024}}
@inproceedings{xie2022cdvae, author={Xie, Tian and Fu, Xiang and Ganea, Octavian-Eugen and Barzilay, Regina and Jaakkola, Tommi}, title={Crystal Diffusion Variational Autoencoder for Periodic Material Generation}, booktitle={International Conference on Learning Representations (ICLR)}, year={2022}}
@inproceedings{jiao2023diffcsp, author={Jiao, Rui and Huang, Wenbing and Lin, Peijia and Han, Jiaqi and Chen, Pin and Lu, Yutong and Liu, Yang}, title={Crystal Structure Prediction by Joint Equivariant Diffusion}, booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2023}}
@article{zeni2025mattergen, author={Zeni, Claudio and others}, title={A Generative Model for Inorganic Materials Design}, journal={Nature}, volume={639}, pages={624--632}, year={2025}}
@article{hautier2011, author={Hautier, Geoffroy and Fischer, Chris and Ehrlacher, Virginie and Jain, Anubhav and Ceder, Gerbrand}, title={Data Mined Ionic Substitutions for the Discovery of New Compounds}, journal={Inorganic Chemistry}, volume={50}, number={2}, pages={656--663}, year={2011}}
@article{jain2013mp, author={Jain, Anubhav and others}, title={Commentary: The {Materials Project}: A Materials Genome Approach to Accelerating Materials Innovation}, journal={APL Materials}, volume={1}, pages={011002}, year={2013}}
@article{ong2013pymatgen, author={Ong, Shyue Ping and others}, title={Python Materials Genomics (pymatgen): A Robust, Open-Source Python Library for Materials Analysis}, journal={Computational Materials Science}, volume={68}, pages={314--319}, year={2013}}
@article{deng2023chgnet, author={Deng, Bowen and Zhong, Peichen and Jun, KyuJung and Riebesell, Janosh and Han, Kevin and Bartel, Christopher J. and Ceder, Gerbrand}, title={{CHGNet} as a Pretrained Universal Neural Network Potential for Charge-Informed Atomistic Modelling}, journal={Nature Machine Intelligence}, volume={5}, pages={1031--1041}, year={2023}}
@article{davies2019smact, author={Davies, Daniel W. and others}, title={{SMACT}: Semiconducting Materials by Analogy and Chemical Theory}, journal={Journal of Open Source Software}, volume={4}, number={38}, pages={1361}, year={2019}}
@inproceedings{lewis2020rag, author={Lewis, Patrick and others}, title={Retrieval-Augmented Generation for Knowledge-Intensive {NLP} Tasks}, booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2020}}
@inproceedings{wang2020minilm, author={Wang, Wenhui and Wei, Furu and Dong, Li and Bao, Hangbo and Yang, Nan and Zhou, Ming}, title={{MiniLM}: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers}, booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2020}}
@inproceedings{reimers2019sbert, author={Reimers, Nils and Gurevych, Iryna}, title={Sentence-{BERT}: Sentence Embeddings using Siamese {BERT}-Networks}, booktitle={Proc. EMNLP-IJCNLP}, year={2019}}
@inproceedings{lester2021prompt, author={Lester, Brian and Al-Rfou, Rami and Constant, Noah}, title={The Power of Scale for Parameter-Efficient Prompt Tuning}, booktitle={Proc. EMNLP}, year={2021}}
@inproceedings{li2021prefix, author={Li, Xiang Lisa and Liang, Percy}, title={Prefix-Tuning: Optimizing Continuous Prompts for Generation}, booktitle={Proc. ACL-IJCNLP}, year={2021}}
@article{kirkpatrick2017ewc, author={Kirkpatrick, James and others}, title={Overcoming Catastrophic Forgetting in Neural Networks}, journal={Proceedings of the National Academy of Sciences}, volume={114}, number={13}, pages={3521--3526}, year={2017}}
@article{clark2022battinfo, author={Clark, Simon and others}, title={Toward a Unified Description of Battery Data}, journal={Advanced Energy Materials}, volume={12}, number={17}, pages={2102702}, year={2022}}
@article{he2024deeponto, author={He, Yuan and Chen, Jiaoyan and Dong, Hang and Horrocks, Ian and Allocca, Carlo and Kim, Taehun and Sapkota, Brahmananda}, title={{DeepOnto}: A Python Package for Ontology Engineering with Deep Learning}, journal={Semantic Web}, volume={15}, number={5}, year={2024}}
@article{detrano1989heart, author={Detrano, Robert and others}, title={International Application of a New Probability Algorithm for the Diagnosis of Coronary Artery Disease}, journal={American Journal of Cardiology}, volume={64}, number={5}, pages={304--310}, year={1989}}
@book{efron1994bootstrap, author={Efron, Bradley and Tibshirani, Robert J.}, title={An Introduction to the Bootstrap}, publisher={Chapman and Hall/CRC}, year={1994}}
"""


# ---------------------------------------------------------------- figures
def figures():
    import json
    os.makedirs(FIGS, exist_ok=True)
    tg = json.load(open(os.path.join(HERE, "data", "crystallm", "kg_template_gen.json")))["report"]
    hz = json.load(open(os.path.join(HERE, "data", "crystallm", "kg_substitution.json")))["report"]["found"]
    haut = [v for k, v in hz.items() if k.startswith("Hautier") and "order" in k][0]
    ks = [1, 2, 3, 5, 8]
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for lab, ys, c, mk in (("CrystaLLM (fine-tuned)", [tg["CrystaLLM fine-tuned"][f"found@{k}"] for k in ks], "#6c5ba7", "o"),
                           ("Hautier substitution", [haut[f"@{k}"] for k in ks], "#3b6fb6", "^"),
                           ("composition rule", [tg["composition rule"][f"found@{k}"] for k in ks], "#8a8f98", "s"),
                           ("K-Ont templates", [tg["prompt+kg+ft_kont"][f"found@{k}"] for k in ks], "#d9762b", "o")):
        ax.plot(ks, ys, color=c, marker=mk, ms=3.5, lw=1.8 if "K-Ont" in lab else 1.2, label=lab)
    ax.set_xticks(ks)
    ax.set_xlabel("attempts per material $k$")
    ax.set_ylabel("structures recovered (%)")
    ax.grid(axis="y", color="#e3e6ea", lw=0.6)
    ax.legend(frameon=False, fontsize=6.6, loc="lower right")
    fig.savefig(os.path.join(FIGS, "foundk.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- renderers
def cite_order():
    order = []
    for kind, *rest in B:
        txt = " ".join(x if isinstance(x, str) else " ".join(x) if isinstance(x, list) else
                       " ".join([str(v) for v in x.values()]) for x in rest)
        for grp in re.findall(r"\[(@[^\]]+)\]", txt):
            for k in re.findall(r"@([\w]+)", grp):
                if k not in order:
                    order.append(k)
    return order


def tex_inline(t):
    t = re.sub(r"\s*\[(@[^\]]+)\]", lambda m: "~\\cite{" + ",".join(re.findall(r"@([\w]+)", m.group(1))) + "}", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)
    t = re.sub(r"(?<![\\\w])\*(?!\s)(.+?)\*", r"\\emph{\1}", t)
    t = t.replace("%", "\\%").replace("\\\\%", "\\%")
    t = t.replace("\u00e9", "\\'e").replace("\u00b0", "$^\\circ$").replace("\u2264", "$\\le$")
    t = re.sub(r"(?<=\d)-(?=\d)", "--", t)
    t = t.replace(" -- ", " --- ")
    return t


def html_inline(t, cites):
    t = html.escape(t, quote=False)
    t = re.sub(r"\[(@[^\]]+)\]", lambda m: "[" + ", ".join(str(cites.index(k) + 1) for k in re.findall(r"@([\w]+)", m.group(1))) + "]", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\\\w])\*(?!\s)(.+?)\*", r"<i>\1</i>", t)
    rep = {r"$\le$": "\u2264", r"$^\dagger$": "\u2020", r"$^\ddagger$": "\u2021", r"$^*$": "*", r"$\sqsubseteq \exists$": "\u2291 \u2203",
           r"$\sqsubseteq$": "\u2291", r"$\exists r.D$": "\u2203r.D", r"$\mathbb{B}_c$": "\U0001D539<sub>c</sub>", r"$k=1$": "k = 1",
           r"$k$": "<i>k</i>", r"$Z$": "<i>Z</i>", r"$E_{hull}$": "E<sub>hull</sub>", r"$d_c$": "d<sub>c</sub>",
           r"$\lVert\cdot\rVert_c$": "\u2016\u00b7\u2016<sub>c</sub>", r"$x$": "<i>x</i>", r"$f(x)$": "<i>f</i>(<i>x</i>)",
           r"$C \sqsubseteq D$": "C \u2291 D", r"$r$": "<i>r</i>", r"$f(D)$": "<i>f</i>(D)", r"$\exists$": "\u2203"}
    for a, b in rep.items():
        t = t.replace(html.escape(a, quote=False), b)
    labels = [x[1]["label"] for x in B if x[0] == "table"]
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
    t = re.sub(r"Table~\\ref\{([^}]+)\}", lambda m: "Table " + roman[labels.index(m.group(1))], t)
    t = t.replace("Fig.~\\ref{fig:foundk}", "Fig. 1")
    t = t.replace("($\\dagger$)", "(†)").replace("($\\ddagger$)", "(‡)")
    t = t.replace(" -- ", " — ").replace("--", "–")
    t = t.replace("~", "\u00a0")
    return t


def to_tex():
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
    out = [r"\documentclass[conference]{IEEEtran}", r"\IEEEoverridecommandlockouts",
           r"\usepackage{cite}", r"\usepackage{amsmath,amssymb,amsfonts}", r"\usepackage{graphicx}",
           r"\usepackage{textcomp}", r"\usepackage{xcolor}", r"\usepackage{booktabs}",
           r"% Target: IEEE ICMLA (IEEE conference template, IEEEtran). Check the current CFP for page limit and anonymity.",
           r"\begin{document}", r"\title{" + TITLE + "}", r"\author{"]
    for i, a in enumerate(AUTHORS):
        out.append(r"\IEEEauthorblockN{" + a + "}" + r"\IEEEauthorblockA{" + AFFIL + r"\\ [email]}" + (r" \and" if i < len(AUTHORS) - 1 else ""))
    out += ["}", r"\maketitle", r"\begin{abstract}", tex_inline(ABSTRACT), r"\end{abstract}",
            r"\begin{IEEEkeywords}", ", ".join(KEYWORDS), r"\end{IEEEkeywords}"]
    for kind, *rest in B:
        if kind == "sec":
            out.append(r"\section{" + rest[0] + "}")
        elif kind == "sub":
            out.append(r"\subsection{" + rest[0] + "}")
        elif kind == "p":
            out.append(tex_inline(rest[0]) + "\n")
        elif kind == "ul":
            out.append(r"\begin{itemize}")
            out += [r"\item " + tex_inline(x) for x in rest[0]]
            out.append(r"\end{itemize}")
        elif kind == "eq":
            out.append(r"\begin{equation}" + rest[0] + r"\end{equation}")
        elif kind == "table":
            t = rest[0]
            cols = "l" + "c" * (len(t["head"]) - 1)
            out += [r"\begin{table}[t]", r"\centering", r"\caption{" + tex_inline(t["caption"]) + "}", r"\label{" + t["label"] + "}",
                    r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{" + cols + "}", r"\toprule",
                    " & ".join(tex_inline(h) for h in t["head"]) + r" \\", r"\midrule"]
            out += [" & ".join(tex_inline(c) for c in r) + r" \\" for r in t["rows"]]
            out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        elif kind == "fig":
            f = rest[0]
            out += [r"\begin{figure}[t]", r"\centering", r"\includegraphics[width=\columnwidth]{figs/" + f["file"] + "}",
                    r"\caption{" + tex_inline(f["caption"]) + "}", r"\label{" + f["label"] + "}", r"\end{figure}"]
    out += [r"\bibliographystyle{IEEEtran}", r"\bibliography{refs}", r"\end{document}"]
    return "\n".join(out)


def to_html():
    import base64
    cites = cite_order()
    secn, subn, tabn = 0, 0, 0
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
    parts = [f'<h1>{html.escape(TITLE)}</h1><div class="authors">' +
             "".join(f'<div class="au"><b>{html.escape(a)}</b><br><span>{html.escape(AFFIL)}</span></div>' for a in AUTHORS) + "</div>",
             '<div class="cols">', f'<p class="abs"><b><i>Abstract</i>\u2014</b>{html_inline(ABSTRACT, cites)}</p>',
             f'<p class="abs"><b><i>Index Terms</i>\u2014</b>{html.escape(", ".join(KEYWORDS))}</p>']
    for kind, *rest in B:
        if kind == "sec":
            secn += 1
            subn = 0
            parts.append(f'<h2>{roman[secn - 1]}. {html.escape(rest[0]).upper()}</h2>')
        elif kind == "sub":
            subn += 1
            parts.append(f'<h3>{chr(64 + subn)}. {html.escape(rest[0])}</h3>')
        elif kind == "p":
            parts.append(f"<p>{html_inline(rest[0], cites)}</p>")
        elif kind == "ul":
            parts.append("<ul>" + "".join(f"<li>{html_inline(x, cites)}</li>" for x in rest[0]) + "</ul>")
        elif kind == "eq":
            parts.append(f'<div class="eq">{html.escape(rest[1]).replace("_c", "<sub>c</sub>")}<span>({sum(1 for k, *_ in B[:B.index((kind, *rest)) + 1] if k == "eq")})</span></div>')
        elif kind == "table":
            tabn += 1
            t = rest[0]
            rows = "".join("<tr>" + "".join(f"<td>{html_inline(c, cites)}</td>" for c in r) + "</tr>" for r in t["rows"])
            parts.append(f'<div class="tab"><div class="cap">TABLE {roman[tabn - 1]}<br>{html_inline(t["caption"], cites)}</div>'
                         f'<table><tr>{"".join(f"<th>{html_inline(h, cites)}</th>" for h in t["head"])}</tr>{rows}</table></div>')
        elif kind == "fig":
            f = rest[0]
            b64 = base64.b64encode(open(os.path.join(FIGS, f["file"]), "rb").read()).decode()
            parts.append(f'<div class="fig"><img src="data:image/png;base64,{b64}"><div class="fcap">Fig. 1. {html_inline(f["caption"], cites)}</div></div>')
    bib = {m.group(1): m.group(0) for m in re.finditer(r"@\w+\{(\w+),.*?\}\s*$", BIB, re.M)}

    def fmt(k):
        e = bib.get(k, "")
        g = lambda f: (re.search(f + r"=\{(.+?)\}(?:,|\}$)", e) or [None, ""])[1]
        au = g("author").replace(" and ", ", ").replace("{", "").replace("}", "")
        venue = g("journal") or g("booktitle") or g("howpublished") or g("publisher")
        return f'{html.escape(au)}, \u201c{html.escape(g("title").replace("{", "").replace("}", "").replace(chr(92) + chr(39) + "e", chr(233)))},\u201d <i>{html.escape(venue)}</i>, {g("year")}.'
    parts.append("<h2>REFERENCES</h2><ol class='refs'>" + "".join(f"<li>{fmt(k)}</li>" for k in cites) + "</ol>")
    parts.append("</div>")
    css = """
@page { size: letter; margin: 0.75in 0.62in 1in 0.62in; }
body { font-family: "Times New Roman", Times, serif; font-size: 10pt; line-height: 1.18; color: #000; margin: 0; }
h1 { font-size: 22pt; font-weight: normal; text-align: center; margin: 0 0 10pt; line-height: 1.15; }
.authors { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6pt 10pt; text-align: center; font-size: 9.5pt; margin-bottom: 14pt; }
.au span { font-style: italic; font-size: 8.5pt; }
.cols { column-count: 2; column-gap: 0.25in; text-align: justify; hyphens: auto; }
.abs { font-size: 9pt; font-weight: bold; } .abs b { font-weight: bold; }
.abs { font-weight: normal; }
h2 { font-size: 10pt; font-weight: normal; text-align: center; margin: 9pt 0 4pt; letter-spacing: .03em; break-after: avoid; }
h3 { font-size: 10pt; font-weight: normal; font-style: italic; margin: 6pt 0 3pt; break-after: avoid; }
p { margin: 0 0 4pt; text-indent: 1em; }
ul { margin: 0 0 4pt 12pt; padding: 0; } li { margin-bottom: 2pt; }
.eq { text-align: center; margin: 4pt 0; position: relative; font-style: italic; } .eq span { position: absolute; right: 0; font-style: normal; }
.tab { break-inside: avoid; margin: 6pt 0; } .cap { text-align: center; font-size: 8pt; margin-bottom: 3pt; letter-spacing: .02em; }
table { border-collapse: collapse; width: 100%; font-size: 7.6pt; border-top: 1px solid #000; border-bottom: 1px solid #000; }
th { border-bottom: 0.5px solid #000; font-weight: normal; padding: 1.5pt 2pt; } td { padding: 1.2pt 2pt; text-align: center; }
td:first-child, th:first-child { text-align: left; }
.fig { break-inside: avoid; margin: 6pt 0; } .fig img { width: 100%; } .fcap { font-size: 8pt; }
.refs { font-size: 8pt; padding-left: 14pt; } .refs li { margin-bottom: 2pt; }
"""
    return f'<!doctype html><html><head><meta charset="utf-8"><title>K-Ont IEEE draft</title><style>{css}</style></head><body>{"".join(parts)}</body></html>'


def main():
    os.makedirs(OUT, exist_ok=True)
    figures()
    open(os.path.join(OUT, "kont_ieee.tex"), "w", encoding="utf-8").write(to_tex())
    open(os.path.join(OUT, "refs.bib"), "w", encoding="utf-8").write(BIB.strip() + "\n")
    hp = os.path.join(OUT, "kont_ieee_preview.html")
    open(hp, "w", encoding="utf-8").write(to_html())
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={PREVIEW}",
                    "file:///" + hp.replace("\\", "/")], check=True, capture_output=True, timeout=180)
    print("wrote", os.path.join(OUT, "kont_ieee.tex"), "and", PREVIEW)


if __name__ == "__main__":
    main()
