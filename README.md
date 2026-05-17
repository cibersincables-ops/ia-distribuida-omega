# Distributed AI Omega

Semantic verification system for distributed AI inference on volunteer nodes.
Queries two Ollama nodes in parallel, measures semantic coherence Ω between their responses,
and only returns if the score exceeds the threshold of 0.65.

**Validated metric: AUC-ROC = 0.9539** · Paper: https://doi.org/10.5281/zenodo.15520283

---

## Quick Start

### On each machine that acts as a node

**Mac / Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
ollama serve
```

**Windows:**
```powershell
# Download and install from ollama.com
ollama pull llama3.2:3b
$env:OLLAMA_HOST="0.0.0.0"; ollama serve
```

### Only on the machine running the proxy

**Mac / Linux:**
```bash
git clone https://github.com/cibersincables-ops/ia-distribuida-omega
cd ia-distribuida-omega
pip3 install -r requirements.txt
python3 scripts/omega_proxy.py --nodos "NODE1_IP:11434,NODE2_IP:11434" --puerto 8000
```

**Windows:**
```powershell
git clone https://github.com/cibersincables-ops/ia-distribuida-omega
cd ia-distribuida-omega
pip install -r requirements.txt
python scripts/omega_proxy.py --nodos "NODE1_IP:11434,NODE2_IP:11434" --puerto 8000
```

---

## Verify it works

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"ok": true}
```

## Make a query

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.2:3b","messages":[{"role":"user","content":"Hello"}]}'
```

Response with Ω included:
```json
{
  "choices": [{"message": {"content": "Hello, how can I help you?"}}],
  "omega_meta": {"omega": 0.7155, "nodo": "192.168.0.102:11434", "verificado": true}
}
```

## Check proxy status

```bash
curl http://localhost:8000/status
```

---

## Timeout note

The proxy comes configured with `TIMEOUT_NODO = 90` seconds, optimized for nodes without GPU.
If all your nodes have GPU you can lower it to 30 for faster responses:

```python
TIMEOUT_NODO = 30  # reduce if nodes have GPU
```

---

## Additional documentation

- `EXPERIMENTO.md` — empirical validation of the Ω metric, AUC 0.9539
- `REGISTRO_SESION.md` — complete technical session log
- Academic paper: https://doi.org/10.5281/zenodo.15520283

---

**Author:** Cristian Cano González · Orizaba, Veracruz, México · 2026
**License:** MIT
