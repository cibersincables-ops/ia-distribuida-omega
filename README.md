# IA Distribuida · Fase 0 · Validación de Ω

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

## Estructura

```
ia-distribuida-fase0/
├── README.md                      ← este archivo
├── data/
│   └── dataset_wiki.jsonl         ← 1,000 pares reales de Wikipedia
├── scripts/
│   ├── generar_pares.py           ← genera el dataset (sintético o real)
│   ├── omega_wiki.py              ← experimento principal · AUC 0.9539
│   └── pares_manuales.py          ← 45 pares de control manual · AUC 0.906
├── resultados/
│   └── resultados_fase0.json      ← métricas finales
└── docs/
    └── (documentación adicional)
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

## Siguiente paso: Fase 1

Con la métrica Ω validada, el siguiente paso es implementar
el prototipo de 3 nodos con Tailscale y verificar que el sistema
completo funciona end-to-end.

Ver documentación completa en: `ia-distribuida-v3.html`

🛠️ Evolución de la Arquitectura: Fase 1 (En Desarrollo)
Originalmente planteada como una red desde cero, la arquitectura ha evolucionado para aprovechar la potencia de Exo como motor de inferencia distribuida, añadiendo una capa de control soberana basada en redes ISP.

1. El Motor: Exo Integration

Utilizaremos Exo para el "sharding" de modelos entre dispositivos (MacOS, Linux, iOS). Exo proporciona el músculo computacional, permitiendo que el Acueducto escale sin necesidad de servidores centralizados.

2. El Sistema Inmune: Proxy Ω

Sobre Exo, implementaremos un Middleware Proxy que actúa como una aduana semántica:

Validación en Tiempo Real: Cada respuesta de los nodos es analizada por nuestra métrica Ω (validada al 95%) antes de llegar al usuario.

Protocolo FIX: Los mensajes se empaquetan bajo el estándar de mensajería financiera FIX para asegurar trazabilidad y auditoría total de cada fragmento.

3. Infraestructura de Red (Basado en ia-distribuida-v3.html)

Para garantizar la resiliencia, el proyecto hereda conceptos de redes de nivel ISP:

Capa de Transporte: Uso de VPNs Mesh (Tailscale/WireGuard) para crear un túnel seguro entre nodos voluntarios.

Ruteo Dinámico (ACO): Implementación de algoritmos de colonia de hormigas para priorizar rutas a través de nodos con mayor historial de honestidad.

Hardware Ready: El diseño contempla la integración en equipos MikroTik/Cisco para una gestión de tráfico eficiente mediante protocolos como OSPF adaptados.
