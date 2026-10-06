# Explainable Agentic AI for Intent-Based Network Slice Selection

Reproduction package for the ICIITCEE 2027 submission *"Explainable Agentic AI for
Intent-Based Network Slice Selection"*.

A four-agent pipeline (Intent, Candidate, Selection, Validation) plus a cross-cutting
Explanation Agent converts a free-form operator request into a 3GPP slice/service type,
gates the result through a feasibility contract and a confidence threshold, and justifies
every decision with feature attributions.

## Architecture at a glance

![Pipeline and ablation chain](architecture.png)

One diagram covering everything: the six pipeline roles, the shared blackboard, the typed
tools the E5 agent calls, the RETRY edge from the critic back to the Intent Agent, and the
badge on each component showing which experiment substitutes it. Source:
[`architecture.puml`](architecture.puml); also rendered as `architecture.pdf` / `.svg`.

Regenerate after editing the source:

```bash
java -jar plantuml.jar -tpng -tsvg -charset UTF-8 architecture.puml
```

---

## The five experiments

All five share one corpus, one seed (`42`) and one train/test split, so the cross-experiment
comparison is **controlled rather than merely co-reported**. Each experiment changes exactly
one thing.

| | Experiment | What changes | Research gap addressed |
|---|---|---|---|
| **E1** | Post-hoc explanation | — (baseline) | Is a composed agentic slice selector accurate, and are post-hoc SHAP attributions faithful enough to drive an escalation policy? |
| **E2** | In-hoc vs post-hoc | the **explainer** | Post-hoc SHAP costs ~3,100× the inference it explains — fine for admission-time decisions, fatal for near-real-time RIC loops. Can in-hoc explanation match its faithfulness at negligible cost? |
| **E3** | Semantic Intent Agent | the **lexical prior** | The design-time lexicon is perfect in-vocabulary but loses ~11 points out-of-vocabulary. Does semantic coverage close that gap? |

### E2 — explainers compared

**In-hoc** (attribution emitted by the same forward pass as the prediction, no second computation):

- **Linear-Attribution** — logistic regression, `a_j = w_jc · x_j`
- **Decision-Path** — depth-8 tree, class-probability shift at each split along the decision path
- **SENN** — self-explaining network, `logit_c = Σ_j Θ_jc(x) · x_j`, attribution `a_j = Θ_jc(x) · x_j`
  (implemented in pure NumPy with manual backprop and Adam — no deep-learning dependency)

**Post-hoc** (attribution requires a separate computation):

- **SHAP** — `TreeExplainer`, falling back to `PermutationExplainer`
- **LIME** — local surrogate, subsampled (optional; install `lime`)

All are scored on accuracy, macro F1, explanation latency, and top-*k* ablation faithfulness
against a random-feature control.

### E3 — Intent Agent variants compared

| Variant | Lexical prior | Dependency |
|---|---|---|
| **A. Lexicon** (E1 baseline) | binary exact substring match | none |
| **B. TF-IDF n-gram** | char 3–5-gram cosine similarity to prototypes | scikit-learn |
| **C. Embeddings** | MiniLM sentence-embedding cosine similarity | `sentence-transformers` |
| **D. LLM zero-shot** | Together API slice-affinity scoring | `requests` + `TOGETHER_API_KEY` |

Prototypes are drawn **only** from the design-time known use cases, so unseen and generic
phrasings stay genuinely out-of-vocabulary for every variant. Scoring is cached per distinct
use-case phrase, so the LLM variant costs ~35 API calls, not 3,000.

**Headline metric:** `oov_gap = accuracy(in-lexicon) − mean(accuracy(unseen), accuracy(generic))`

### E4 — do the E2 and E3 gains compose?

E2 and E3 each change one component and each reports a gain on a *different* axis, but
neither varies the other factor — so neither can establish that the two improvements
compose. E4 tests that directly with a full 2×2 factorial on the same corpus, split and seed:

| | lexical prior = exact | lexical prior = TF-IDF |
|---|---|---|
| **explainer = SHAP** | E1 baseline | E3 best |
| **explainer = Decision-Path** | E2 best | **E4 combined** |

**Headline metric:** the *interaction*, `Δ(combined) − [Δ(explainer only) + Δ(prior only)]`.
An interaction near zero means the axes are orthogonal and the two substitutions can be
adopted independently.

### E5 — does a *real* agent beat the deterministic Intent Agent?

E1–E4 all keep the same Intent Agent: regex slot-filling plus a brittle exact-substring
lexicon. E5 substitutes that one component for an **LLM agent that plans, calls typed tools
and is audited by a critic that can send its work back**. Everything downstream — candidate
screening, the Selection Agent, the Explanation Agent, the feasibility contract — is
byte-identical to E1, so E5 extends the nested ablation chain instead of confounding it.

| Property | E1–E4 Intent Agent | E5 Intent Agent |
|---|---|---|
| Goal-directed reasoning | fixed regex | LLM decides *how* to extract |
| Tool use | none | `get_kpi_envelope`, `convert_units`, `query_lexicon`, `check_feasibility` |
| Planning / iteration | single pass | multi-turn until the frame closes (≤ 4 turns) |
| Inter-agent messaging | dict passing | typed append-only blackboard |
| Negotiation / critique | none | Validation Agent may return **RETRY** (≤ 1 retry) |
| Autonomy with deferral | confidence τ | Validation Agent may return **ESCALATE** |

**Pre-registered hypothesis (H5).** Replacing the deterministic Intent Agent with an agent
that reasons semantically about novel phrasings will reduce the open-vocabulary gap
`Δ_OOV` by more than E3's TF-IDF prior did, at materially higher per-intent cost.

Both outcomes are reportable. If E5 wins, the agentic framing is earned and the paper gains
a cost/benefit curve across E1 → E3 → E5. If E5 ties or loses, that is a second negative
composition result consistent with E4, and the thesis *"we measure which substitutions
pay"* is reinforced.

**Determinism.** Every LLM call runs at `temperature=0` with a fixed `seed` and is cached on
disk in `results/llm_cache.json`. Commit that file and anyone can replay E5 bit-exactly
**with no API key** by setting `XAGENTIC_LLM_OFFLINE=1`.

**Cost.** ≈ 900 test intents × ~1.8 calls ≈ **$1–2 total** on Llama-3.3-70B-Turbo. Use
`--agent-limit 60` for a ~$0.10 smoke run first.

---

## How to run all five experiments

> **TL;DR for Colab:** paste the block in Step 2 into a fresh notebook cell, run it, then
> download `xagentic_results.zip` (Step 4) and hand that file back for the paper update.

The experiments are **not run separately** — `run_all.py` generates one shared corpus
and feeds the same `DataFrame` to E1–E5, which is what makes the cross-experiment
comparison controlled. Running the scripts individually would give each a *different*
corpus instance and invalidate the comparison.

### Step 1 — Open Colab

Go to <https://colab.research.google.com> → **New notebook**. CPU runtime is sufficient;
no GPU needed. End-to-end runtime is roughly **8–15 minutes** for E1–E4 (3–5 minutes with
`--quick`), plus **~25–40 minutes** for E5 on the full test split.

### Step 2 — Paste and run this single cell

```python
# ── 1. Dependencies ───────────────────────────────────────────────────────
!pip -q install shap lime sentence-transformers scikit-learn pandas numpy matplotlib requests

# ── 2. Get the code ───────────────────────────────────────────────────────
# Option A: from GitHub, once the repo is published
# !git clone https://github.com/<YOUR-GITHUB-USERNAME>/xagentic-slice.git
# %cd xagentic-slice/experiment

# Option B: upload experiment/*.py manually (no repo needed)
import os
os.makedirs('/content/experiment', exist_ok=True)
%cd /content/experiment
from google.colab import files
files.upload()   # select: xagentic_common.py, exp1_posthoc.py, exp2_inhoc.py,
                 #         exp3_semantic.py, exp4_combined.py, exp5_agentic.py,
                 #         agents.py, run_all.py, run_seeds.py

# ── 3. Together API key (E3 variant D, and all of E5) ────────────────────
import getpass
os.environ['TOGETHER_API_KEY'] = getpass.getpass('TOGETHER_API_KEY (Enter to skip): ')
os.environ['TOGETHER_MODEL'] = 'meta-llama/Llama-3.3-70B-Instruct-Turbo'

# ── 4. Smoke-test E5 cheaply BEFORE paying for the full run (~$0.10) ─────
!python exp5_agentic.py --limit 40

# ── 5. Run all five experiments on one shared corpus ─────────────────────
!python run_all.py --with-agent
```

If you have no Together API key, press Enter at the prompt and drop `--with-agent` — E1–E4
still run and the paper compiles; only Table V's "LLM zero-shot" row and the whole of E5
will be absent.

### Step 3 — Confirm the run succeeded

The last lines of output must list every artefact with `OK`:

```
[run_all] done. Copy these back into the paper directory:
  OK /content/experiment/results/results_macros.tex
  OK /content/experiment/results/fig_shap_global.pdf
  OK /content/experiment/results/fig_escalation.pdf
  OK /content/experiment/results/fig_e2_latency.pdf
  OK /content/experiment/results/fig_e3_oov.pdf
  OK /content/experiment/results/fig_e4_combined.pdf
  OK /content/experiment/results/fig_e5_agentic.pdf
  OK /content/experiment/results/comparison.csv
  OK /content/experiment/results/e4_factorial.csv
  OK /content/experiment/results/e5_agentic.csv
  OK /content/experiment/results/e5_summary.json
  OK /content/experiment/results/llm_cache.json
  OK /content/experiment/results/all_summary.json
```

Any line starting `--` means that artefact was not produced — see Troubleshooting. The
four `e5_*` / `fig_e5_*` / `llm_cache.json` lines appear only when you passed
`--with-agent`.

Just before that, E4 prints the factorial it just measured — this is the
"do the gains compose?" result:

```
[E4] accuracy effect  explainer only : -1.4 pts
[E4] accuracy effect  prior only     : +1.9 pts
[E4] accuracy effect  both combined  : +0.X pts
[E4] interaction (combined - sum)    : +0.X pts
```

### Step 4 — Download the results

```python
import shutil
from google.colab import files
shutil.make_archive('/content/xagentic_results', 'zip', '/content/experiment/results')
files.download('/content/xagentic_results.zip')
```

### Step 5 — Hand the results back

**Send back `xagentic_results.zip`.** That single file contains everything needed to update
the manuscript. If you prefer to paste text instead of uploading, the two essential items are:

1. the full contents of **`results_macros.tex`** (this alone fills every number in the paper), and
2. the printed **`CROSS-EXPERIMENT COMPARISON`** table from the run log.

The figures (`fig_*.pdf`) must be sent as files — they cannot be pasted as text.

### Step 6 — Rebuild the paper

Unpack the archive into `experiment/results/`, then:

```bash
pdflatex paper.tex && pdflatex paper.tex && pdflatex paper.tex
```

`paper.tex` hard-codes **no** experimental number, so every `XX.X` placeholder resolves
automatically and Fig. 4's placeholder boxes are replaced by the real plots. No manual
transcription, and therefore no transcription errors.

Verify afterwards:

```bash
pdffonts paper.pdf | grep "Type 3"     # must return nothing (IEEE PDF eXpress)
pdftotext paper.pdf - | grep -c "XX"   # must return 0
pdfinfo paper.pdf | grep Pages         # page count may shift as floats reflow
```

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `fig_e2_latency.pdf` missing | E2 crashed before plotting | Re-run with `--no-llm`; check the E2 traceback |
| `-- ` beside `fig_shap_global.pdf` | `e1_shap_global.csv` absent — E1 SHAP step failed | Usually a `shap` version clash: `!pip install -q "shap>=0.44"` |
| E3 variant D absent from Table V | no `TOGETHER_API_KEY` | expected; set the key to enable |
| E3 variant C absent | `sentence-transformers` failed to install | re-run the pip cell, or pass `--no-embeddings` |
| `Session crashed` / OOM | default Colab RAM exceeded | use `--quick` (1,200-intent corpus) |
| Run exceeds ~20 min | LIME on the full corpus | `!python run_all.py --quick` |
| E5 absent / `E5 skipped (RuntimeError: TOGETHER_API_KEY is not set …)` | no key and no cache | set the key, or replay a committed `llm_cache.json` |
| E5 costs more than expected | full 900-intent split | smoke-test with `--agent-limit 60` first |
| E5 `429` warnings | Together rate limit | already retried with backoff; re-run — cached calls are not repaid |
| E5 numbers changed between runs | cache missed (different corpus/seed/prompt) | keep the seed fixed; set `XAGENTIC_LLM_OFFLINE=1` to make a miss fail loudly |

### Command-line options

```bash
python run_all.py                       # E1-E4, all variants
python run_all.py --quick               # 1,200-intent corpus, fast smoke run
python run_all.py --no-llm              # skip the Together API variant (E3-D)
python run_all.py --no-embeddings       # skip the sentence-transformer variant (E3-C)
python run_all.py --with-agent          # add E5 (needs TOGETHER_API_KEY)
python run_all.py --with-agent --agent-limit 60   # cheap E5 smoke run

python exp5_agentic.py --limit 40       # E5 alone, ~$0.10
XAGENTIC_LLM_OFFLINE=1 python run_all.py --with-agent   # replay from cache, no key
```

### Running locally instead

Only if you have a working Python toolchain — the Colab path above is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd experiment && python run_all.py
```

> Running `exp1_posthoc.py`, `exp2_inhoc.py`, `exp3_semantic.py` or `exp4_combined.py`
> directly is supported for
> debugging a single experiment, but each then synthesises its **own** corpus. Only
> `run_all.py` produces a valid cross-experiment comparison and the LaTeX macros.

---

## Colab notebook (alternative to Step 2)

`notebooks/ICIITCEE2027_experiments.ipynb` wraps the same workflow with narrative cells.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<YOUR-GITHUB-USERNAME>/xagentic-slice/blob/main/notebooks/ICIITCEE2027_experiments.ipynb)

> Replace `<YOUR-GITHUB-USERNAME>` in the badge URL, in the notebook's `REPO_URL`
> variable, and in `paper.tex` once the repository is published.

### Together API key

Needed for E3 variant D and for all of E5. Everything else runs without it.

- **Colab Secrets** (preferred): add a secret named `TOGETHER_API_KEY`, enable notebook access.
- **Prompt**: the notebook asks for the key interactively if no secret is found.
- **Shell**: `export TOGETHER_API_KEY=...`

Select a model with `TOGETHER_MODEL` (default `meta-llama/Llama-3.3-70B-Instruct-Turbo`)
for E3-D, and with `XAGENTIC_AGENT_MODEL` for E5.

Once `results/llm_cache.json` is committed, E5 replays with **no key at all**:
`XAGENTIC_LLM_OFFLINE=1 python run_all.py --with-agent`.

---

## Repository layout

```
experiment/
  xagentic_common.py     corpus generator, deterministic Intent Agent, slice
                         catalogue, feature builder, shared metrics
  agents.py              E5 -- LLM Intent + Validation agents, typed tools,
                         tool-calling loop, deterministic disk cache
  exp1_posthoc.py        E1 -- post-hoc SHAP
  exp2_inhoc.py          E2 -- in-hoc vs post-hoc (incl. NumPy SENN)
  exp3_semantic.py       E3 -- semantic Intent Agent variants
  exp4_combined.py       E4 -- 2x2 factorial, do E2+E3 gains compose?
  exp5_agentic.py        E5 -- agentic Intent Agent vs the deterministic one
  run_all.py             runs E1-E5, emits comparison + LaTeX macros
  run_seeds.py           multi-seed driver, emits the gamma confidence interval
  results/               all generated artefacts (created on first run)
notebooks/
  ICIITCEE2027_experiments.ipynb   Colab notebook
paper.tex                manuscript
requirements.txt
```

---

## Outputs

Everything lands in `experiment/results/`.

| File | Contents | Needed by `paper.tex`? |
|---|---|---|
| `results_macros.tex` | **LaTeX macros consumed by `paper.tex`** | ✅ **essential** |
| `fig_shap_global.pdf` | global SHAP attribution ranking (Fig. 3a) | ✅ |
| `fig_escalation.pdf` | escalation sweep over τ (Fig. 3b) | ✅ |
| `fig_e2_latency.pdf` | explanation latency vs faithfulness (Fig. 4a) | ✅ |
| `fig_e3_oov.pdf` | open-vocabulary gap by variant (Fig. 4b) | ✅ |
| `fig_e4_combined.pdf` | E4 2×2 factorial outcome (Fig. 5) | ✅ |
| `fig_e5_agentic.pdf` | E5 agent vs lexicon, escalation quality (Fig. 6) | ✅ with `--with-agent` |
| `comparison.csv` | **cross-experiment summary** | reference |
| `all_summary.json` | every metric from every experiment | reference |
| `corpus.csv` | the 3,000-intent corpus with ground-truth labels | — |
| `e1_models.csv` | E1 Selection Agent model sweep | — |
| `e1_per_class.csv` | E1 per-slice precision / recall / F1 | — |
| `e1_shap_global.csv` | global SHAP attribution ranking | — |
| `e1_escalation.csv` | escalation policy across confidence thresholds | — |
| `e1_local_explanations.json` | representative per-decision justifications | — |
| `e1_summary.json` | E1 metrics | — |
| `e2_explainers.csv` | E2 in-hoc vs post-hoc comparison | — |
| `e3_variants.csv` | E3 Intent Agent variant comparison | — |
| `e4_factorial.csv` | E4 2×2 factorial, one row per cell | — |
| `e4_summary.json` | E4 main effects and interaction | — |
| `e5_agentic.csv` | E5 vs E1 by use-case vocabulary | — |
| `e5_decisions.csv` | E5 per-intent verdict, retries, escalation | — |
| `e5_summary.json` | E5 metrics: Δ_OOV, retry rate, tools, tokens, cost | — |
| `llm_cache.json` | **commit this** — replays E5 bit-exactly with no API key | reproducibility |

### Wiring results into the paper

`paper.tex` hard-codes **no** experimental number. Every value is a macro defined in
`results_macros.tex`, which `run_all.py` regenerates, and `paper.tex` picks it up
automatically via:

```latex
\IfFileExists{results_macros.tex}{\input{results_macros}}{%
  \IfFileExists{experiment/results/results_macros.tex}%
    {\input{experiment/results/results_macros}}{}}
```

So unpacking the results archive into `experiment/results/` is sufficient — **no copying and
no edits to `paper.tex`**. Figures resolve through `\graphicspath{{experiment/results/}}`.

```bash
unzip xagentic_results.zip -d experiment/results/
pdflatex paper.tex && pdflatex paper.tex && pdflatex paper.tex
```

> Until you run the experiments, the `\providecommand` fallbacks in the `paper.tex` preamble
> supply **placeholder values** (`XX.X`, `XXXX`) so the manuscript still compiles, and Fig. 4
> renders as two labelled placeholder boxes. Real values silently override them, because
> `\providecommand` never overwrites an existing definition.

> **Font note.** `run_all.py` and `xagentic_slice.py` both set `pdf.fonttype: 42`, so
> generated figures embed TrueType rather than Type 3 fonts. This is required — IEEE PDF
> eXpress **rejects Type 3 fonts**. Verify with `pdffonts paper.pdf`.

---

## Reproducibility notes

- **Seed** `42` throughout: corpus generation, train/test split, cross-validation, model
  initialisation, SHAP background sampling, and the random-ablation control.
- **Split** 70/30 stratified by slice type. Cross-validation (5-fold stratified, macro F1) runs
  on the training split only; the held-out split plays no role in selection or tuning.
- **Ground truth** is the *pre-blending* latent slice type, so strongly blended intents are
  under-determined by construction — this is intended, and such intents belong in the
  escalation path rather than being guessed.
- **Scaling statistics** are fitted inside pipelines so nothing leaks across the split boundary.
- **Determinism caveat**: wall-clock latency figures (`expl_ms`, `infer_ms`) depend on the
  host. Relative ratios are the transferable quantity, not the absolute milliseconds.

---

## Corpus

Synthesised from 3GPP TS 22.261 KPI envelopes. 3,000 intents balanced across eMBB, URLLC
and mMTC, with two independently controlled difficulty dimensions:

- **KPI ambiguity** (p = 0.30): KPI targets log-domain blended towards a neighbouring envelope
  with mixing weight `a ~ U(0.4, 0.75)`, placing the request in the overlap region between two
  slice types while the correct answer remains the originally intended slice.
- **Open-vocabulary drift**: 40% of intents use a use-case phrase in the design-time lexicon,
  30% use slice-appropriate but lexically unseen phrases, and 30% use vertical-agnostic
  phrasings with no lexical signal at all.

Seven templates differ in which slots the operator states, so 14–56% of intents omit any given
slot — matching the under-specification endemic to real intent declarations. Unmatched slots
are recorded as **missing**, never imputed to zero, and explicit missingness indicators are
carried downstream.

**Limitation.** This is a synthetic corpus. Real intents exhibit syntactic variety,
self-contradiction and multi-service composition the templates do not reproduce. Absolute
accuracies should be read as upper bounds; the *relative* findings — rule-based vs learned,
in- vs out-of-lexicon, in-hoc vs post-hoc, SHAP vs random ablation — are the transferable ones.
No standard benchmark yet pairs natural-language intents with ground-truth slice assignments.

---

## Citation

```bibtex
@inproceedings{xagentic2027,
  title     = {Explainable Agentic {AI} for Intent-Based Network Slice Selection},
  booktitle = {Proc. 5th Int. Conf. Intelligent and Innovative Technologies in
               Computing, Electrical and Electronics (ICIITCEE)},
  address   = {Bengaluru, India},
  year      = {2027},
  note      = {Author names withheld -- under double-blind review}
}
```

## License

MIT — see `LICENSE`.
