# IA Distribuida Omega

Sistema de verificación semántica para inferencia de IA distribuida en nodos voluntarios.
Consulta dos nodos Ollama en paralelo, mide coherencia semántica Ω entre sus respuestas
y solo devuelve si supera el umbral de 0.65.

**Métrica validada: AUC-ROC = 0.9539** · Paper: https://doi.org/10.5281/zenodo.15520283

---

## Inicio rápido

### Para cada máquina que sea nodo

**Mac / Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
ollama serve
```

**Windows:**
```powershell
# Descargar e instalar desde ollama.com
ollama pull llama3.2:3b
$env:OLLAMA_HOST="0.0.0.0"; ollama serve
```

### Solo en la máquina que corre el proxy

**Mac / Linux:**
```bash
git clone https://github.com/cibersincables-ops/ia-distribuida-omega
cd ia-distribuida-omega
pip3 install -r requirements.txt
python3 scripts/omega_proxy.py --nodos "IP_NODO1:11434,IP_NODO2:11434" --puerto 8000
```

**Windows:**
```powershell
git clone https://github.com/cibersincables-ops/ia-distribuida-omega
cd ia-distribuida-omega
pip install -r requirements.txt
python scripts/omega_proxy.py --nodos "IP_NODO1:11434,IP_NODO2:11434" --puerto 8000
```

---

## Verificar que funciona

```bash
curl http://localhost:8000/health
```

Respuesta esperada:
```json
{"ok": true}
```

## Hacer una consulta

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.2:3b","messages":[{"role":"user","content":"Hola"}]}'
```

Respuesta con Ω incluido:
```json
{
  "choices": [{"message": {"content": "Hola, en que puedo ayudarte?"}}],
  "omega_meta": {"omega": 0.7155, "nodo": "192.168.0.102:11434", "verificado": true}
}
```

## Ver estado del proxy

```bash
curl http://localhost:8000/status
```

---

## Nota sobre el timeout

El proxy viene configurado con `TIMEOUT_NODO = 90` segundos, optimizado para nodos sin GPU.
Si todos tus nodos tienen GPU puedes bajarlo a 30 para respuestas más rápidas:

```python
TIMEOUT_NODO = 30  # reducir si los nodos tienen GPU
```

---

## Documentación adicional

- `EXPERIMENTO.md` — validación empírica de la métrica Ω, AUC 0.9539
- `REGISTRO_SESION.md` — diario técnico completo de la sesión
- Paper académico: https://doi.org/10.5281/zenodo.15520283

---

**Autor:** Cristian Cano González · Orizaba, Veracruz, México · 2026
**Licencia:** MIT
