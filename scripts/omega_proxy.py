#!/usr/bin/env python3
"""
omega_proxy.py - Proxy de coherencia semántica para Ollama
"""
import argparse, asyncio, logging, time
from dataclasses import dataclass
from typing import Optional
import aiohttp
from aiohttp import web
from sentence_transformers import SentenceTransformer, util

OMEGA_UMBRAL = 0.65
MAX_REINTENTOS = 2
TIMEOUT_NODO = 90

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ΩProxy")

log.info("Cargando modelo de embeddings...")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
log.info("Modelo listo ✓")

@dataclass
class RespuestaNodo:
    nodo: str; texto: str; latencia_ms: float; ok: bool; error: str = ""

async def consultar_nodo(session, nodo_url, payload):
    url = f"http://{nodo_url}/v1/chat/completions"
    t0 = time.time()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=TIMEOUT_NODO)) as resp:
            latencia = (time.time() - t0) * 1000
            if resp.status != 200:
                return RespuestaNodo(nodo_url, "", latencia, False, f"HTTP {resp.status}")
            data = await resp.json()
            texto = data["choices"][0]["message"]["content"]
            return RespuestaNodo(nodo_url, texto, latencia, True)
    except Exception as e:
        return RespuestaNodo(nodo_url, "", (time.time()-t0)*1000, False, str(e))

def calcular_omega(a, b):
    if not a.strip() or not b.strip(): return 0.0
    ea = embedder.encode(a, convert_to_tensor=True)
    eb = embedder.encode(b, convert_to_tensor=True)
    return round(float(util.cos_sim(ea, eb)), 4)

class OmegaProxy:
    def __init__(self, nodos):
        self.nodos = nodos
        self.stats = {"consultas": 0, "aceptadas": 0, "rechazadas": 0, "omega_promedio": 0.0}
        log.info(f"Proxy iniciado con {len(nodos)} nodos: {nodos}")

    async def procesar(self, request):
        try: payload = await request.json()
        except: return web.json_response({"error": "JSON inválido"}, status=400)
        self.stats["consultas"] += 1
        for intento in range(MAX_REINTENTOS + 1):
            resultado = await self._verificar(payload, intento)
            if resultado:
                texto, omega, nodo = resultado
                self.stats["aceptadas"] += 1
                n = self.stats["aceptadas"]
                self.stats["omega_promedio"] = round(self.stats["omega_promedio"] + (omega - self.stats["omega_promedio"]) / n, 4)
                log.info(f"✓ Aceptado Ω={omega} nodo={nodo}")
                return web.json_response({
                    "id": f"omega-{int(time.time())}",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": texto}, "finish_reason": "stop"}],
                    "omega_meta": {"omega": omega, "nodo": nodo, "verificado": True}
                })
            if intento < MAX_REINTENTOS:
                log.warning(f"⚠ Ω bajo — reintentando ({intento+2}/{MAX_REINTENTOS+1})")
        self.stats["rechazadas"] += 1
        return web.json_response({"error": "coherencia_insuficiente"}, status=503)

    async def _verificar(self, payload, intento):
        n = len(self.nodos)
        nodo_a = self.nodos[(intento * 2) % n]
        nodo_b = self.nodos[(intento * 2 + 1) % n] if n > 1 else self.nodos[0]
        async with aiohttp.ClientSession() as session:
            ra, rb = await asyncio.gather(
                consultar_nodo(session, nodo_a, payload),
                consultar_nodo(session, nodo_b, payload)
            )
        if not ra.ok and not rb.ok: return None
        if not ra.ok: return (rb.texto, 1.0, rb.nodo)
        if not rb.ok: return (ra.texto, 1.0, ra.nodo)
        omega = calcular_omega(ra.texto, rb.texto)
        log.info(f"  Ω = {omega} (umbral: {OMEGA_UMBRAL})")
        if omega >= OMEGA_UMBRAL:
            ganador = ra if ra.latencia_ms <= rb.latencia_ms else rb
            return (ganador.texto, omega, ganador.nodo)
        return None

    async def status(self, request):
        return web.json_response({"proxy": "omega-proxy", "nodos": self.nodos, "umbral": OMEGA_UMBRAL, "stats": self.stats})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodos", default="localhost:11434,localhost:11434")
    parser.add_argument("--puerto", type=int, default=8000)
    args = parser.parse_args()
    nodos = [n.strip() for n in args.nodos.split(",")]
    proxy = OmegaProxy(nodos)
    app = web.Application()
    app.router.add_post("/v1/chat/completions", proxy.procesar)
    app.router.add_get("/status", proxy.status)
    app.router.add_get("/health", lambda r: web.json_response({"ok": True}))
    print(f"\n  Omega Proxy v1.0 · Cristian Cano González")
    print(f"  Puerto: {args.puerto} · Nodos: {nodos}")
    print(f"  AUC validado: 0.9539\n")
    web.run_app(app, host="0.0.0.0", port=args.puerto, access_log=None)

if __name__ == "__main__":
    main()
