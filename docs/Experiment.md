# Distributed AI · Phase 0 · Omega Metric Validation

Technical documentation of the Omega metric validation experiment.

---

## Result

```
AUC-ROC:    0.9539  ✅ (target: >0.85)
Separation: 0.369
Verdict:    APPROVED — proceed to Phase 1
```

## What was validated

The Ω formula can discriminate semantically coherent fragments
from divergent fragments with AUC above 0.85 using real-world data.

**Model:** `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers)
**Dataset:** Wikipedia Spanish 20231101 · 1,000 pairs · 500/500 balance

## Repository structure

```
ia-distribuida-omega/
├── README.md                  ← installation and quick start
├── requirements.txt           ← proxy dependencies
├── data/
│   └── dataset_wiki.jsonl     ← 1,000 real Wikipedia pairs
├── docs/
│   ├── EXPERIMENTO.md         ← technical documentation Spanish
│   └── EXPERIMENT.md          ← technical documentation English
├── resultados/
│   └── resultados_fase0.json  ← final metrics
└── scripts/
    ├── omega_proxy.py         ← semantic verification proxy
    ├── omega_wiki.py          ← main experiment
    ├── generar_pares.py       ← dataset generator
    └── pares_manuales.py      ← 45 manual control pairs
```

## How to reproduce the experiment

### On Google Colab with T4 GPU

```python
# Cell 1: install
!pip install sentence-transformers scikit-learn -q

# Cell 2: upload dataset_wiki.jsonl to the data/ folder

# Cell 3: run
!python scripts/omega_wiki.py
```

### Expected output

```
AUC real: 0.9539  (target: >0.85)
Coherent    mu= 0.433  n=500
Divergent   mu= 0.064  n=500
Separation:     0.369
✅ IMMUNE SYSTEM WORKS — proceed to Phase 1
```

## What we learned

| Experiment | Model | Dataset | AUC |
|------------|-------|---------|-----|
| Attempt 1 | Phi-3 mini · activations | Synthetic | 0.59 |
| Attempt 2 | Sentence-transformers | Synthetic | 0.55 |
| Attempt 3 | Sentence-transformers | 45 manual pairs | 0.906 |
| **Final** | **Sentence-transformers** | **Real Wikipedia** | **0.9539** |

**Conclusion:** Dataset quality is the critical factor.
With real, well-structured data the metric works clearly.

## Dataset: dataset_wiki.jsonl

Format of each line:
```json
{
  "texto_a": "...",
  "texto_b": "...",
  "etiqueta": 1,
  "subgrupo": "A",
  "tipo": "consecutivas"
}
```

- `etiqueta=1`: coherent pair (consecutive sentences from the same article)
- `etiqueta=0`: divergent pair (sentences from different articles)
- `subgrupo A`: coherent pairs
- `subgrupo B`: divergent pairs

## Result on real network — Phase 1

The Ω proxy was tested on a real network on May 16, 2026:

- Node 1: Mac M2 with llama3.2:3b (Apple Silicon GPU)
- Node 2: Windows CPU with llama3.2:1b (no GPU)
- Ω measured: **0.7155** between different models on different hardware ✅

The system detected semantic divergence, retried, and accepted when it exceeded the 0.65 threshold.
First case of imperfect semantic coherence measured on a real distributed network.

## Academic paper

Title: Semantic Coherence Verification for Distributed AI Inference
Author: Cristian Cano González
DOI: https://doi.org/10.5281/zenodo.15520283
License: CC BY 4.0

---

**Author:** Cristian Cano González · Orizaba, Veracruz, México · 2026
