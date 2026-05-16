import json
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import roc_auc_score
from collections import defaultdict

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

with open("data/dataset_wiki.jsonl") as f:
    dataset = [json.loads(l) for l in f]

textos_a  = [p["texto_a"] for p in dataset]
textos_b  = [p["texto_b"] for p in dataset]
labels    = [p["etiqueta"] for p in dataset]

print(f"Procesando {len(dataset)} pares de Wikipedia...")
emb_a = model.encode(textos_a, convert_to_tensor=True, show_progress_bar=True)
emb_b = model.encode(textos_b, convert_to_tensor=True, show_progress_bar=True)
cosenos = util.cos_sim(emb_a, emb_b).diagonal().tolist()

auc     = roc_auc_score(labels, cosenos)
auc_inv = roc_auc_score(labels, [-c for c in cosenos])
auc_real = max(auc, auc_inv)

pos = [cosenos[i] for i in range(len(labels)) if labels[i]==1]
neg = [cosenos[i] for i in range(len(labels)) if labels[i]==0]

print(f"\n{'='*50}")
print(f"AUC real:         {auc_real:.4f}  (meta: >0.85)")
print(f"Coherentes  mu=   {sum(pos)/len(pos):.3f}  n={len(pos)}")
print(f"Divergentes mu=   {sum(neg)/len(neg):.3f}  n={len(neg)}")
print(f"Separacion:       {sum(pos)/len(pos) - sum(neg)/len(neg):.3f}")
print(f"{'='*50}")

if auc_real >= 0.85:
    print("\n✅ SISTEMA INMUNE FUNCIONA — avanzar a Fase 1")
elif auc_real >= 0.75:
    print("\n⚠️  MARGINAL — recalibrar o mejorar dataset")
else:
    print("\n❌ INSUFICIENTE — revisar señal")
