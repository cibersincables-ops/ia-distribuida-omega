# IA Distribuida · Fase 0 · Validación de Ω

Documentación técnica del experimento de validación de la métrica Omega.

---

## Resultado

```
AUC-ROC:    0.9539  ✅ (meta: >0.85)
Separación: 0.369
Veredicto:  APROBADO — avanzar a Fase 1
```

## Qué se validó

La fórmula Ω puede discriminar fragmentos semánticamente coherentes
de fragmentos divergentes con AUC superior a 0.85 usando datos reales.

**Modelo:** `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers)
**Dataset:** Wikipedia español 20231101 · 1,000 pares · 500/500 balance

## Estructura del repositorio

```
ia-distribuida-omega/
├── README.md                  ← instalación e inicio rápido
├── requirements.txt           ← dependencias del proxy
├── data/
│   └── dataset_wiki.jsonl     ← 1,000 pares reales de Wikipedia
├── docs/
│   ├── EXPERIMENTO.md         ← documentación técnica español
│   └── EXPERIMENT.md          ← documentación técnica inglés
├── resultados/
│   └── resultados_fase0.json  ← métricas finales
└── scripts/
    ├── omega_proxy.py         ← proxy de verificación semántica
    ├── omega_wiki.py          ← experimento principal
    ├── generar_pares.py       ← genera el dataset
    └── pares_manuales.py      ← 45 pares de control
```
## Cómo reproducir el experimento

### En Google Colab con GPU T4

```python
# Celda 1: instalar
!pip install sentence-transformers scikit-learn -q

# Celda 2: subir dataset_wiki.jsonl a la carpeta data/

# Celda 3: correr
!python scripts/omega_wiki.py
```

### Resultado esperado

```
AUC real: 0.9539  (meta: >0.85)
Coherentes  mu= 0.433  n=500
Divergentes mu= 0.064  n=500
Separacion:     0.369
✅ SISTEMA INMUNE FUNCIONA — avanzar a Fase 1
```

## Lo que aprendimos

| Experimento | Modelo | Dataset | AUC |
|-------------|--------|---------|-----|
| Intento 1 | Phi-3 mini · activaciones | Sintético | 0.59 |
| Intento 2 | Sentence-transformers | Sintético | 0.55 |
| Intento 3 | Sentence-transformers | 45 pares manuales | 0.906 |
| **Final** | **Sentence-transformers** | **Wikipedia real** | **0.9539** |

**Conclusión:** La calidad del dataset es el factor crítico.
Con datos reales y bien estructurados la métrica funciona claramente.

## Dataset: dataset_wiki.jsonl

Formato de cada línea:
```json
{
  "texto_a": "...",
  "texto_b": "...",
  "etiqueta": 1,
  "subgrupo": "A",
  "tipo": "consecutivas"
}
```

- `etiqueta=1`: par coherente (oraciones consecutivas del mismo artículo)
- `etiqueta=0`: par divergente (oraciones de artículos distintos)
- `subgrupo A`: coherentes
- `subgrupo B`: divergentes

## Resultado en red real — Fase 1

El proxy Ω fue probado en red real el 16 de mayo de 2026:

- Nodo 1: Mac M2 con llama3.2:3b (GPU Apple Silicon)
- Nodo 2: Windows CPU con llama3.2:1b (sin GPU)
- Ω medido: **0.7155** entre modelos distintos en hardware distinto ✅

El sistema detectó divergencia semántica, reintentó y aceptó al superar el umbral de 0.65.
Primer caso de coherencia semántica imperfecta medida en red distribuida real.

## Paper académico

Título: Semantic Coherence Verification for Distributed AI Inference
Autor: Cristian Cano González
DOI: https://doi.org/10.5281/zenodo.15520283
Licencia: CC BY 4.0

---

**Autor:** Cristian Cano González · Orizaba, Veracruz, México · 2026
