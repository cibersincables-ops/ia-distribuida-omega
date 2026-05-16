#!/usr/bin/env python3
"""
generar_pares.py
Genera el dataset de 1,000 pares etiquetados para validar la fórmula Ω.

Subgrupos:
  A: 400 pares coherentes (ventana deslizante + paráfrasis)
  B: 300 pares divergentes (dominios distintos + alucinaciones)
  C: 200 pares difíciles (mismo dominio, argumento opuesto)
  D: 100 pares maliciosos (tokens corrompidos)

Uso:
  python generar_pares.py --modo sintetico        # sin GPU, para probar el pipeline
  python generar_pares.py --modo real             # con GPU, Wikipedia real
  python generar_pares.py --modo sintetico --n 100  # dataset pequeño para debug
"""

import argparse
import json
import random
import re
import hashlib
from pathlib import Path
from collections import Counter
import pandas as pd

SEED = 42
random.seed(SEED)

# ── CORPUS SINTÉTICO ──────────────────────────────────────────────────────────
# Corpus amplio por dominio — suficiente para generar 400 pares coherentes
CORPUS_SINTETICO = {
    "ciencia": [
        "Las redes neuronales procesan información en capas sucesivas donde cada neurona aplica una función de activación no lineal sobre sus entradas ponderadas.",
        "La transformada de Fourier descompone una señal en sus componentes de frecuencia permitiendo analizar el espectro de cualquier función periódica.",
        "El algoritmo de gradiente descendente ajusta los parámetros del modelo minimizando iterativamente la función de pérdida en la dirección opuesta al gradiente.",
        "Los árboles de decisión particionan el espacio de características de forma recursiva usando criterios de impureza como la entropía de Shannon.",
        "La regularización L2 penaliza la norma cuadrada de los pesos evitando el sobreajuste al mantener los coeficientes pequeños durante el entrenamiento.",
        "El teorema de Bayes relaciona la probabilidad condicional de un evento con su probabilidad a priori y la verosimilitud de los datos observados.",
        "Las redes convolucionales aprovechan la invarianza traslacional de las imágenes compartiendo pesos en filtros locales aplicados sobre toda la entrada.",
        "El problema del gradiente que desaparece dificulta el entrenamiento de redes profundas porque los gradientes se vuelven exponencialmente pequeños.",
        "Los modelos de atención calculan una suma ponderada de los valores basada en la similitud entre consultas y claves en el espacio de embeddings.",
        "La normalización por lotes estabiliza el entrenamiento estandarizando las activaciones de cada capa reduciendo la dependencia de la tasa de aprendizaje.",
        "La entropía de Shannon mide la cantidad de incertidumbre o información en una distribución de probabilidad y es fundamental en teoría de la información.",
        "El algoritmo de retropropagación calcula gradientes de la función de pérdida respecto a cada parámetro aplicando la regla de la cadena de forma eficiente.",
        "Los modelos de lenguaje de gran escala aprenden distribuciones sobre secuencias de tokens entrenando sobre corpus masivos de texto mediante predicción.",
        "La complejidad computacional clasifica los problemas según los recursos necesarios para resolverlos distinguiendo entre clases P NP y NP-completo.",
        "Los sistemas de recomendación filtran información relevante para cada usuario usando técnicas de filtrado colaborativo y basado en contenido.",
        "La criptografía de curva elíptica ofrece seguridad equivalente a RSA con claves mucho más cortas reduciendo el costo computacional de las operaciones.",
        "El aprendizaje por refuerzo entrena agentes que maximizan una recompensa acumulada interactuando con un entorno mediante ensayo y error.",
        "Los transformadores usan mecanismos de atención multi-cabeza que permiten modelar dependencias de largo alcance en secuencias de forma eficiente.",
        "La inferencia variacional aproxima distribuciones posteriores intratables mediante optimización convirtiendo la inferencia en un problema de optimización.",
        "Los grafos de conocimiento representan entidades y relaciones en forma estructurada permitiendo razonamiento simbólico sobre dominios específicos.",
    ],
    "medicina": [
        "El sistema inmune adaptativo genera anticuerpos específicos contra antígenos extraños mediante la diferenciación clonal de linfocitos B activados.",
        "La resistencia a antibióticos surge cuando las bacterias desarrollan mecanismos de defensa como la producción de beta-lactamasas que inactivan el fármaco.",
        "La hipertensión arterial sostenida daña las paredes vasculares incrementando el riesgo de accidente cerebrovascular e infarto de miocardio.",
        "Los inhibidores de la ECA reducen la presión arterial bloqueando la conversión de angiotensina I en angiotensina II un potente vasoconstrictor endógeno.",
        "La resonancia magnética funcional mide cambios en la oxigenación de la sangre como proxy de la actividad neuronal en regiones cerebrales específicas.",
        "El cáncer surge cuando mutaciones en genes supresores de tumores o proto-oncogenes desregulan el ciclo celular permitiendo proliferación descontrolada.",
        "La vacunación induce inmunidad adaptativa sin causar enfermedad al exponer al sistema inmune a antígenos inactivados o fragmentos proteicos purificados.",
        "Los ensayos clínicos aleatorizados son el estándar de oro para evaluar la eficacia de intervenciones médicas minimizando sesgos de selección sistemáticos.",
        "La diabetes tipo 2 resulta de resistencia a la insulina en tejidos periféricos combinada con disfunción progresiva de las células beta pancreáticas.",
        "Los biomarcadores son moléculas medibles en tejidos o fluidos biológicos que indican estados fisiológicos o la respuesta a tratamientos específicos.",
        "La terapia génica introduce material genético en células del paciente para corregir defectos hereditarios o conferir nuevas funciones terapéuticas.",
        "El microbioma intestinal influye en la salud metabólica inmunológica y neurológica a través de metabolitos que actúan como señales sistémicas.",
        "Los anticuerpos monoclonales son proteínas diseñadas para unirse con alta especificidad a dianas moleculares usados en tratamientos oncológicos.",
        "La epigenética estudia cambios heredables en la expresión génica no codificados en la secuencia de ADN incluyendo metilación e histonas modificadas.",
        "El síndrome metabólico agrupa obesidad abdominal hipertensión dislipidemia e hiperglucemia elevando el riesgo cardiovascular de forma sinérgica.",
    ],
    "economia": [
        "La curva de Phillips describe la relación inversa entre inflación y desempleo sugiriendo que las políticas de estímulo tienen costos en términos de precios.",
        "El modelo de valoración de activos de capital relaciona el rendimiento esperado de un activo con su riesgo sistemático medido por el coeficiente beta.",
        "La teoría de juegos analiza decisiones estratégicas entre agentes racionales cuyas utilidades dependen de las elecciones de los demás participantes.",
        "La política monetaria expansiva reduce las tasas de interés incrementando la oferta monetaria para estimular la inversión y el consumo agregado.",
        "Los mercados eficientes incorporan toda la información disponible en los precios haciendo imposible obtener rendimientos superiores de forma consistente.",
        "La externalidad negativa ocurre cuando la producción o consumo de un bien impone costos a terceros no reflejados en el precio de mercado.",
        "El índice de Gini mide la desigualdad en la distribución del ingreso siendo cero perfecta igualdad y uno desigualdad máxima en una sociedad.",
        "La trampa de liquidez ocurre cuando las tasas de interés son tan bajas que la política monetaria convencional pierde efectividad para estimular la economía.",
        "El comercio internacional genera ganancias mutuas cuando los países se especializan en bienes con ventaja comparativa según la teoría ricardiana.",
        "La inflación erosiona el poder adquisitivo real de los salarios y ahorros redistribuyendo riqueza de acreedores a deudores en términos reales.",
        "El multiplicador fiscal estima cuánto aumenta el PIB por cada unidad de gasto público siendo mayor en economías con alta propensión marginal al consumo.",
        "Los bonos soberanos son instrumentos de deuda emitidos por gobiernos cuyo rendimiento refleja la percepción de riesgo crediticio del emisor.",
        "La teoría del capital humano sostiene que la educación y la salud aumentan la productividad de los trabajadores y sus ingresos futuros esperados.",
    ],
    "tecnologia": [
        "El protocolo TCP garantiza la entrega ordenada de paquetes mediante números de secuencia y acuses de recibo entre emisor y receptor conectados.",
        "La arquitectura de microservicios descompone una aplicación en servicios independientes que se comunican por APIs facilitando el despliegue y escalado.",
        "Los contenedores virtualizan el entorno de ejecución de una aplicación empaquetando código y dependencias para garantizar reproducibilidad entre entornos.",
        "El cifrado asimétrico usa pares de claves pública y privada permitiendo comunicaciones seguras sin intercambiar secretos previamente entre las partes.",
        "La computación en la nube ofrece recursos de cómputo almacenamiento y red como servicio escalable bajo demanda con modelo de pago por uso.",
        "Los sistemas de control de versiones registran el historial de cambios en el código facilitando la colaboración y la recuperación ante errores humanos.",
        "La inteligencia artificial generativa aprende distribuciones de datos para producir nuevas muestras con características similares al conjunto original.",
        "El protocolo HTTP define la estructura de las solicitudes y respuestas que permiten la comunicación entre clientes y servidores web en internet.",
        "Los sistemas de archivos distribuidos replican datos en múltiples nodos para garantizar disponibilidad y tolerancia a fallos en sistemas de producción.",
        "La virtualización permite ejecutar múltiples sistemas operativos en un mismo hardware físico mediante un hipervisor que gestiona los recursos compartidos.",
        "Las bases de datos NoSQL sacrifican consistencia fuerte por disponibilidad y tolerancia a particiones según el teorema CAP en sistemas distribuidos.",
        "El algoritmo de consenso Raft garantiza que todos los nodos de un sistema distribuido acuerden el mismo estado incluso con fallos parciales de red.",
        "La programación funcional trata la computación como evaluación de funciones matemáticas evitando estados mutables y efectos secundarios en el código.",
        "Los sistemas operativos en tiempo real garantizan tiempos de respuesta deterministas críticos en aplicaciones industriales aeroespaciales y médicas.",
        "El aprendizaje federado entrena modelos en datos distribuidos sin centralizar la información respetando la privacidad de cada participante.",
    ],
    "historia": [
        "La Revolución Industrial transformó las economías agrarias en industriales mediante la mecanización de la producción y el uso del vapor como fuente energética.",
        "El sistema de Bretton Woods estableció el dólar como moneda de reserva mundial y creó el FMI y el Banco Mundial para estabilizar la economía global.",
        "La imprenta de Gutenberg democratizó el acceso al conocimiento escrito reduciendo el costo de los libros y acelerando la difusión de ideas en Europa.",
        "El colonialismo europeo extrajo recursos de Asia África y América Latina estableciendo estructuras de dependencia económica que persisten hasta hoy.",
        "La Revolución Francesa abolió el Antiguo Régimen instaurando principios de soberanía popular libertad e igualdad que influyeron en constituciones globales.",
        "La Guerra Fría dividió al mundo en bloques ideológicos opuestos generando conflictos por proxy y una carrera armamentista nuclear de décadas.",
        "La Revolución Verde aumentó dramáticamente la producción agrícola en países en desarrollo mediante variedades mejoradas fertilizantes y mecanización.",
        "El comercio triangular conectó Europa África y América en un sistema de explotación que trasladó millones de personas esclavizadas al continente americano.",
        "La descolonización del siglo XX transformó el mapa político global al crear decenas de nuevos estados independientes en Asia y África principalmente.",
        "La Revolución Científica del siglo XVII estableció el método experimental como base del conocimiento superando la autoridad aristotélica medieval.",
    ],
    "cocina": [
        "La reacción de Maillard ocurre entre aminoácidos y azúcares reductores a altas temperaturas produciendo los compuestos aromáticos del dorado superficial.",
        "La emulsificación combina líquidos inmiscibles como aceite y agua mediante agentes tensoactivos que reducen la tensión superficial entre ambas fases.",
        "El proceso de fermentación alcohólica convierte azúcares en etanol y dióxido de carbono mediante la actividad metabólica de levaduras en condiciones anaerobias.",
        "La gelatinización del almidón ocurre cuando los gránulos absorben agua y se hinchan al calentarse formando una estructura viscosa en salsas y cremas.",
        "El sous vide cocina alimentos al vacío a temperatura precisa durante períodos prolongados logrando texturas imposibles con métodos convencionales.",
        "La fermentación láctica produce ácido láctico a partir de azúcares mediante bacterias como Lactobacillus preservando alimentos y creando sabores complejos.",
    ],
    "filosofia": [
        "El problema mente-cuerpo examina la relación entre estados mentales subjetivos y procesos físicos objetivos del cerebro sin resolución consensuada aún.",
        "El utilitarismo evalúa la corrección moral de una acción por sus consecuencias maximizando la felicidad total del mayor número de individuos afectados.",
        "El existencialismo sostiene que la existencia precede a la esencia y que el ser humano se define completamente por sus elecciones en un mundo sin sentido.",
        "La epistemología investiga la naturaleza del conocimiento sus fuentes sus límites y la distinción entre creencia justificada verdadera y mera opinión.",
        "El determinismo sostiene que todos los eventos están causalmente determinados por estados anteriores haciendo imposible la libertad de acción genuina.",
        "El constructivismo social afirma que el conocimiento la realidad y el significado son construidos intersubjetivamente mediante prácticas sociales compartidas.",
        "La ética del cuidado prioriza las relaciones interpersonales y la responsabilidad hacia los vulnerables sobre principios abstractos universales de justicia.",
        "El pragmatismo evalúa las ideas por sus consecuencias prácticas sosteniendo que la verdad es lo que funciona en la experiencia concreta de las personas.",
    ],
    "matematicas": [
        "El teorema de Gödel demuestra que en todo sistema formal suficientemente expresivo existen proposiciones verdaderas que no pueden ser probadas dentro del sistema.",
        "La topología estudia propiedades de los espacios que se preservan bajo deformaciones continuas como la conectividad y la compacidad de conjuntos.",
        "El cálculo diferencial e integral proporciona herramientas para analizar tasas de cambio y acumulación fundamentales en física economía e ingeniería.",
        "La teoría de grupos estudia estructuras algebraicas con una operación binaria asociativa elemento neutro e inverso con aplicaciones en física y criptografía.",
        "Los números primos son divisibles únicamente por uno y por sí mismos y su distribución entre los naturales sigue patrones descritos por la hipótesis de Riemann.",
        "La probabilidad bayesiana interpreta la probabilidad como grado de creencia actualizable con nueva evidencia mediante el teorema fundamental de Bayes.",
        "El álgebra lineal estudia vectores matrices transformaciones lineales y sus propiedades fundamentales para la computación científica moderna.",
        "La teoría de grafos modela relaciones entre objetos y tiene aplicaciones en redes sociales rutas óptimas circuitos y algoritmos de búsqueda eficientes.",
    ],
}

# Argumentos opuestos para subgrupo C
ARGUMENTOS_OPUESTOS = [
    (
        "El código abierto acelera la innovación porque permite que miles de desarrolladores colaboren detecten errores y construyan sobre el trabajo ajeno.",
        "El software propietario garantiza mayor seguridad y calidad porque centraliza el control del código y permite auditorías internas rigurosas."
    ),
    (
        "El teletrabajo aumenta la productividad al eliminar desplazamientos reducir interrupciones y permitir a cada persona trabajar en sus horas óptimas.",
        "El teletrabajo reduce la productividad porque debilita la colaboración espontánea dificulta la supervisión y aísla a los empleados del equipo."
    ),
    (
        "La educación en línea democratiza el acceso al conocimiento permitiendo que personas de cualquier geografía accedan a contenido de calidad mundial.",
        "La educación en línea amplía la brecha digital porque requiere dispositivos y conexión que muchos no tienen y elimina la socialización presencial."
    ),
    (
        "La inteligencia artificial creará más empleos de los que destruya al generar nuevas industrias que hoy no podemos imaginar como ocurrió con internet.",
        "La inteligencia artificial destruirá más empleos de los que crea porque automatiza tareas cognitivas que antes solo humanos podían realizar."
    ),
    (
        "El ayuno intermitente mejora la salud metabólica al reducir la insulina basal promover la autofagia y facilitar la pérdida de peso sostenida.",
        "El ayuno intermitente puede ser perjudicial para personas con historial de trastornos alimentarios y no es superior a una dieta equilibrada constante."
    ),
    (
        "La energía nuclear es esencial para la transición energética porque provee electricidad limpia base estable que las renovables intermitentes no pueden garantizar.",
        "La energía nuclear es demasiado costosa lenta de construir y riesgosa para ser parte de la solución climática cuando solar y eólica son más baratas."
    ),
    (
        "Las criptomonedas ofrecen una alternativa real al sistema financiero tradicional al permitir transacciones sin intermediarios resistentes a la censura.",
        "Las criptomonedas son principalmente instrumentos especulativos que consumen energía masiva facilitan actividades ilícitas y carecen de respaldo real."
    ),
    (
        "La globalización ha reducido la pobreza extrema al integrar economías emergentes en cadenas de valor que generan empleos y transfieren tecnología.",
        "La globalización ha aumentado la desigualdad dentro de los países ricos al deslocalizar empleos industriales y concentrar beneficios en el capital."
    ),
    (
        "Las redes sociales fortalecen la democracia al dar voz a ciudadanos comunes facilitar la organización colectiva y hacer responsables a los poderosos.",
        "Las redes sociales dañan la democracia al amplificar la desinformación crear cámaras de eco polarizar a la sociedad y manipular la opinión pública."
    ),
    (
        "El veganismo es la dieta más sostenible ambientalmente porque reduce drásticamente las emisiones de gases de efecto invernadero y el uso de tierra y agua.",
        "Una dieta omnívora bien planificada con ganadería regenerativa puede ser igual de sostenible que el veganismo y más adecuada para muchos contextos culturales."
    ),
]


def generar_ventana_deslizante(textos, n=200):
    """
    Pares coherentes del mismo dominio.
    Con corpus de frases, combina frases adyacentes del mismo dominio.
    """
    pares = []
    dominios = list(CORPUS_SINTETICO.keys())
    intentos = 0
    while len(pares) < n and intentos < n * 20:
        intentos += 1
        dominio = random.choice(dominios)
        frases = CORPUS_SINTETICO[dominio]
        if len(frases) < 2:
            continue
        i = random.randint(0, len(frases) - 2)
        a, b = frases[i], frases[i + 1]
        if a != b:
            pares.append({
                "texto_a": a, "texto_b": b,
                "etiqueta": 1, "subgrupo": "A",
                "tipo": "mismo_dominio_adyacente"
            })
    return pares[:n]


def generar_parafrasis_sintetica(textos, n=200):
    """Pares coherentes: misma idea con variación sintáctica simple."""
    pares = []
    transformaciones = [
        lambda t: t.replace("porque", "ya que").replace("permite", "posibilita"),
        lambda t: t.replace("mediante", "a través de").replace("usando", "empleando"),
        lambda t: t.replace("cuando", "en el momento en que"),
        lambda t: t.replace("produce", "genera").replace("reduce", "disminuye"),
        lambda t: "Es importante notar que " + t[:120] + ".",
        lambda t: t[:90] + ", lo cual tiene implicaciones relevantes en múltiples contextos.",
        lambda t: "En términos generales, " + t[:130] + ".",
        lambda t: t.replace("Los ", "Estos ").replace("La ", "Esta "),
        lambda t: t.replace("El ", "Dicho ").replace("una ", "cierta "),
        lambda t: "Se sabe que " + t[:130] + ".",
    ]
    candidatos = [t for textos_d in CORPUS_SINTETICO.values() for t in textos_d]
    intentos = 0
    while len(pares) < n and intentos < n * 20:
        intentos += 1
        texto = random.choice(candidatos)
        transform = random.choice(transformaciones)
        try:
            parafrasis = transform(texto)
            if parafrasis != texto and len(parafrasis) > 40:
                pares.append({
                    "texto_a": texto, "texto_b": parafrasis,
                    "etiqueta": 1, "subgrupo": "A",
                    "tipo": "parafrasis"
                })
        except Exception:
            pass
    return pares[:n]


def generar_dominios_distintos(n=150):
    """Pares divergentes: dominios completamente distintos."""
    pares = []
    dominios = list(CORPUS_SINTETICO.keys())
    textos_por_dominio = CORPUS_SINTETICO
    intentos = 0
    while len(pares) < n and intentos < n * 10:
        intentos += 1
        dom_a, dom_b = random.sample(dominios, 2)
        texto_a = random.choice(textos_por_dominio[dom_a])
        texto_b = random.choice(textos_por_dominio[dom_b])
        pares.append({
            "texto_a": texto_a, "texto_b": texto_b,
            "etiqueta": 0, "subgrupo": "B",
            "tipo": f"dominios_{dom_a}_vs_{dom_b}"
        })
    return pares[:n]


def generar_alucinaciones(n=150):
    """Pares divergentes: texto real vs versión con datos falsos."""
    candidatos = [t for textos_d in CORPUS_SINTETICO.values() for t in textos_d]
    sustituciones = [
        ("reduce", "incrementa"), ("aumenta", "disminuye"),
        ("positivo", "negativo"), ("mejora", "empeora"),
        ("estable", "inestable"), ("eficiente", "ineficiente"),
        ("seguro", "peligroso"), ("preciso", "impreciso"),
        ("lineal", "exponencial"), ("directo", "inverso"),
        ("mayor", "menor"), ("alto", "bajo"),
        ("permite", "impide"), ("garantiza", "elimina"),
    ]
    pares = []
    intentos = 0
    while len(pares) < n and intentos < n * 20:
        intentos += 1
        texto = random.choice(candidatos)
        alucinacion = texto
        sust_shuffled = sustituciones.copy()
        random.shuffle(sust_shuffled)
        aplicadas = 0
        for original, falso in sust_shuffled:
            if original.lower() in alucinacion.lower() and aplicadas < 2:
                alucinacion = re.sub(original, falso, alucinacion,
                                     flags=re.IGNORECASE, count=1)
                aplicadas += 1
        if aplicadas > 0 and alucinacion != texto:
            pares.append({
                "texto_a": texto, "texto_b": alucinacion,
                "etiqueta": 0, "subgrupo": "B",
                "tipo": "alucinacion"
            })
    return pares[:n]


def generar_argumentos_opuestos(n=200):
    """Pares difíciles: mismo dominio, postura contraria (subgrupo C)."""
    pares = []
    pares_base = ARGUMENTOS_OPUESTOS.copy()
    while len(pares) < n:
        random.shuffle(pares_base)
        for a, b in pares_base:
            if len(pares) >= n:
                break
            # Alternar quién va primero
            if random.random() > 0.5:
                a, b = b, a
            pares.append({
                "texto_a": a, "texto_b": b,
                "etiqueta": 0, "subgrupo": "C",
                "tipo": "argumento_opuesto"
            })
    return pares[:n]


def generar_corrompidos(n=100):
    """Pares maliciosos: texto real vs versión con tokens corrompidos (subgrupo D)."""
    candidatos = [t for textos_d in CORPUS_SINTETICO.values() for t in textos_d]
    pares = []
    random.shuffle(candidatos)
    for texto in candidatos:
        if len(pares) >= n:
            break
        palabras = texto.split()
        if len(palabras) < 8:
            continue
        corrompido = palabras.copy()
        n_corruptos = max(2, len(palabras) // 4)
        indices = random.sample(range(len(palabras)), min(n_corruptos, len(palabras)))
        for idx in indices:
            tipo = random.randint(0, 3)
            if tipo == 0:
                corrompido[idx] = "x" + str(random.randint(100, 999))
            elif tipo == 1:
                corrompido[idx] = corrompido[idx][::-1]
            elif tipo == 2:
                corrompido[idx] = hashlib.md5(
                    corrompido[idx].encode()).hexdigest()[:8]
            else:
                corrompido[idx] = "NULL_TOKEN"
        pares.append({
            "texto_a": texto,
            "texto_b": " ".join(corrompido),
            "etiqueta": 0, "subgrupo": "D",
            "tipo": "corrompido"
        })
    return pares[:n]


def generar_modo_sintetico(n_total=1000):
    """Genera el dataset completo en modo sintético."""
    print("Generando dataset sintético...")
    todos_textos = [t for textos in CORPUS_SINTETICO.values() for t in textos]

    # Proporciones por subgrupo
    n_A = int(n_total * 0.40)   # 400
    n_B = int(n_total * 0.30)   # 300
    n_C = int(n_total * 0.20)   # 200
    n_D = int(n_total * 0.10)   # 100

    print(f"  Subgrupo A (coherentes):      {n_A} pares")
    pares_A1 = generar_ventana_deslizante(todos_textos, n=n_A // 2)
    pares_A2 = generar_parafrasis_sintetica(todos_textos, n=n_A // 2)
    pares_A = pares_A1 + pares_A2

    print(f"  Subgrupo B (divergentes):     {n_B} pares")
    pares_B1 = generar_dominios_distintos(n=n_B // 2)
    pares_B2 = generar_alucinaciones(n=n_B // 2)
    pares_B = pares_B1 + pares_B2

    print(f"  Subgrupo C (opuestos):        {n_C} pares  ← el más difícil")
    pares_C = generar_argumentos_opuestos(n=n_C)

    print(f"  Subgrupo D (corrompidos):     {n_D} pares")
    pares_D = generar_corrompidos(n=n_D)

    dataset = pares_A + pares_B + pares_C + pares_D
    random.shuffle(dataset)

    return dataset


def generar_modo_real(n_total=1000, modelo="llama3"):
    """
    Modo real: usa Wikipedia en español.
    Requiere GPU y conexión a internet.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: instala datasets con: pip install datasets")
        return []

    print("Descargando Wikipedia en español (puede tardar)...")
    wiki = load_dataset("wikipedia", "20220301.es", split="train[:3000]",
                        trust_remote_code=True)
    textos = [r["text"][:1500] for r in wiki if len(r["text"]) > 200]
    random.shuffle(textos)

    print(f"Textos descargados: {len(textos)}")

    n_A = int(n_total * 0.40)
    n_B = int(n_total * 0.30)
    n_C = int(n_total * 0.20)
    n_D = int(n_total * 0.10)

    corpus_completo = " ".join(textos[:500])
    pares_A1 = generar_ventana_deslizante(textos[:500], n=n_A // 2)

    # Paráfrasis con modelo real (si está disponible)
    pares_A2 = generar_parafrasis_sintetica(textos, n=n_A // 2)

    pares_B = generar_dominios_distintos(n=n_B // 2)
    pares_B += [{
        "texto_a": textos[i], "texto_b": textos[i + 500],
        "etiqueta": 0, "subgrupo": "B", "tipo": "articulos_distintos"
    } for i in range(min(n_B // 2, len(textos) - 500))]

    pares_C = generar_argumentos_opuestos(n=n_C)
    pares_D = generar_corrompidos(n=n_D)

    dataset = pares_A1 + pares_A2 + pares_B + pares_C + pares_D
    random.shuffle(dataset)
    return dataset


def validar_dataset(dataset):
    """Validaciones básicas de integridad."""
    print("\nValidando dataset...")
    errores = []

    labels = [d["etiqueta"] for d in dataset]
    conteo = Counter(labels)
    print(f"  Etiquetas: {dict(conteo)}")
    if conteo[0] == 0 or conteo[1] == 0:
        errores.append("ERROR: faltan pares de alguna clase")

    ratio = conteo[1] / len(dataset)
    if not (0.3 <= ratio <= 0.5):
        errores.append(f"AVISO: balance inusual ({ratio:.1%} positivos)")

    subgrupos = Counter(d["subgrupo"] for d in dataset)
    print(f"  Subgrupos: {dict(subgrupos)}")

    # Verificar duplicados
    textos = set()
    duplicados = 0
    for d in dataset:
        key = d["texto_a"][:50]
        if key in textos:
            duplicados += 1
        textos.add(key)
    if duplicados > 0:
        print(f"  AVISO: {duplicados} posibles duplicados en texto_a")

    # Verificar longitud mínima
    cortos = sum(1 for d in dataset if len(d["texto_a"]) < 30 or len(d["texto_b"]) < 30)
    if cortos > 0:
        print(f"  AVISO: {cortos} pares con textos muy cortos (<30 chars)")

    if errores:
        for e in errores:
            print(f"  {e}")
        return False

    print("  Dataset válido ✓")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generador de dataset para validación de fórmula Ω")
    parser.add_argument("--modo", choices=["sintetico", "real"],
                        default="sintetico", help="Modo de generación")
    parser.add_argument("--n", type=int, default=1000,
                        help="Número total de pares (default: 1000)")
    parser.add_argument("--salida", type=str, default="data/dataset_pares.jsonl",
                        help="Archivo de salida")
    parser.add_argument("--semilla", type=int, default=42, help="Semilla aleatoria")
    args = parser.parse_args()

    random.seed(args.semilla)
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)

    if args.modo == "sintetico":
        dataset = generar_modo_sintetico(n_total=args.n)
    else:
        dataset = generar_modo_real(n_total=args.n)

    if not dataset:
        print("ERROR: dataset vacío")
        return

    if not validar_dataset(dataset):
        print("ERROR: dataset con problemas, revisar antes de continuar")
        return

    # Guardar JSONL
    with open(args.salida, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Guardar CSV para inspección
    csv_path = args.salida.replace(".jsonl", ".csv")
    df = pd.DataFrame(dataset)
    df.to_csv(csv_path, index=False)

    print(f"\nDataset guardado:")
    print(f"  JSONL: {args.salida}  ({len(dataset)} pares)")
    print(f"  CSV:   {csv_path}")
    print(f"\nInspección manual recomendada:")
    print(f"  python -c \"import pandas as pd; df=pd.read_csv('{csv_path}'); print(df.sample(20).to_string())\"")


if __name__ == "__main__":
    main()
