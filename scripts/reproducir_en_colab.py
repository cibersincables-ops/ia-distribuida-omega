#!/usr/bin/env python3
"""
reproducir_en_colab.py
Script completo para reproducir el experimento Fase 0 en Google Colab.

Instrucciones:
1. Abrir Google Colab con GPU T4
2. Subir este archivo y dataset_wiki.jsonl
3. Ejecutar: !python reproducir_en_colab.py
"""

import subprocess, sys, os, json
from pathlib import Path

# ── Instalar dependencias ──────────────────────────────────────────
print("Instalando dependencias...")
subprocess.run([sys.executable, "-m", "pip", "install",
                "sentence-transformers", "scikit-learn", "-q"],
               check=True)

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import roc_auc_score

# ── Buscar dataset ─────────────────────────────────────────────────
rutas = [
    "data/dataset_wiki.jsonl",
    "dataset_wiki.jsonl",
    "/content/dataset_wiki.jsonl",
    "/content/data/dataset_wiki.jsonl",
]
dataset_path = None
for ruta in rutas:
    if Path(ruta).exists():
        dataset_path = ruta
        break

if dataset_path is None:
    print("ERROR: No se encontró dataset_wiki.jsonl")
    print("Sube el archivo a Colab primero.")
    sys.exit(1)

print(f"Dataset encontrado: {dataset_path}")

# ── Cargar dataset ─────────────────────────────────────────────────
with open(dataset_path) as f:
    dataset = [json.loads(l) for l in f]

print(f"Pares cargados: {len(dataset)}")

# ── Cargar modelo ──────────────────────────────────────────────────
print("\nCargando modelo...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ── Calcular embeddings ────────────────────────────────────────────
textos_a = [p["texto_a"] for p in dataset]
textos_b = [p["texto_b"] for p in dataset]
labels   = [p["etiqueta"] for p in dataset]

print("Codificando fragmentos A...")
emb_a = model.encode(textos_a, convert_to_tensor=True, show_progress_bar=True)
print("Codificando fragmentos B...")
emb_b = model.encode(textos_b, convert_to_tensor=True, show_progress_bar=True)

cosenos = util.cos_sim(emb_a, emb_b).diagonal().tolist()

# ── Evaluar ────────────────────────────────────────────────────────
auc     = roc_auc_score(labels, cosenos)
auc_inv = roc_auc_score(labels, [-c for c in cosenos])
auc_real = max(auc, auc_inv)

pos = [cosenos[i] for i in range(len(labels)) if labels[i]==1]
neg = [cosenos[i] for i in range(len(labels)) if labels[i]==0]

print(f"\n{'='*50}")
print(f"RESULTADOS FASE 0")
print(f"{'='*50}")
print(f"Modelo:           paraphrase-multilingual-MiniLM-L12-v2")
print(f"Dataset:          Wikipedia español · {len(dataset)} pares")
print(f"AUC-ROC:          {auc_real:.4f}  (meta: >0.85)")
print(f"Coherentes  mu=   {sum(pos)/len(pos):.3f}  n={len(pos)}")
print(f"Divergentes mu=   {sum(neg)/len(neg):.3f}  n={len(neg)}")
print(f"Separacion:       {sum(pos)/len(pos) - sum(neg)/len(neg):.3f}")
print(f"{'='*50}")

if auc_real >= 0.85:
    print(f"\n✅ SISTEMA INMUNE FUNCIONA — avanzar a Fase 1")
    veredicto = "APROBADO"
elif auc_real >= 0.75:
    print(f"\n⚠️  MARGINAL — recalibrar dataset")
    veredicto = "MARGINAL"
else:
    print(f"\n❌ INSUFICIENTE — revisar señal")
    veredicto = "RECHAZADO"

# ── Guardar resultados ─────────────────────────────────────────────
resultados = {
    "experimento": "Fase 0 validacion Omega",
    "modelo": "paraphrase-multilingual-MiniLM-L12-v2",
    "dataset": "Wikipedia español 20231101",
    "n_pares": len(dataset),
    "auc_roc": round(auc_real, 4),
    "mu_coherentes": round(sum(pos)/len(pos), 3),
    "mu_divergentes": round(sum(neg)/len(neg), 3),
    "separacion": round(sum(pos)/len(pos) - sum(neg)/len(neg), 3),
    "umbral_meta": 0.85,
    "veredicto": veredicto
}

with open("resultados_fase0.json", "w") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

print(f"\nResultados guardados en resultados_fase0.json")
