import json
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import roc_auc_score

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

pares = [
    # COHERENTES etiqueta=1
    {"a": "El agua hierve a 100 grados centígrados a nivel del mar.", "b": "El punto de ebullición del agua es 100 grados a presión atmosférica normal.", "etiqueta": 1},
    {"a": "La fotosíntesis convierte luz solar en energía química.", "b": "Las plantas usan la luz del sol para producir glucosa mediante fotosíntesis.", "etiqueta": 1},
    {"a": "Python es un lenguaje de programación interpretado y de alto nivel.", "b": "Python se interpreta en tiempo de ejecución y tiene sintaxis de alto nivel.", "etiqueta": 1},
    {"a": "El corazón bombea sangre por todo el cuerpo.", "b": "La función principal del corazón es circular la sangre por el organismo.", "etiqueta": 1},
    {"a": "La gravedad atrae los objetos hacia el centro de la Tierra.", "b": "Los objetos caen porque la gravedad terrestre los atrae hacia abajo.", "etiqueta": 1},
    {"a": "El ADN contiene la información genética de los seres vivos.", "b": "La información hereditaria está codificada en la molécula de ADN.", "etiqueta": 1},
    {"a": "Los antibióticos combaten infecciones causadas por bacterias.", "b": "Las bacterias pueden eliminarse con tratamientos antibióticos.", "etiqueta": 1},
    {"a": "La velocidad de la luz es aproximadamente 300,000 kilómetros por segundo.", "b": "La luz viaja a unos 300,000 km/s en el vacío.", "etiqueta": 1},
    {"a": "El dióxido de carbono es un gas de efecto invernadero.", "b": "El CO2 contribuye al calentamiento global por ser un gas invernadero.", "etiqueta": 1},
    {"a": "Los volcanes liberan magma y gases desde el interior de la Tierra.", "b": "La actividad volcánica expulsa material fundido del interior terrestre.", "etiqueta": 1},
    {"a": "El cerebro humano tiene aproximadamente 86 mil millones de neuronas.", "b": "Se estima que el cerebro contiene cerca de 86 billones de células nerviosas.", "etiqueta": 1},
    {"a": "La luna tarda 27 días en girar alrededor de la Tierra.", "b": "El período orbital de la luna alrededor de la Tierra es de unos 27 días.", "etiqueta": 1},
    {"a": "El oxígeno es esencial para la respiración celular.", "b": "Sin oxígeno las células no pueden realizar la respiración aeróbica.", "etiqueta": 1},
    {"a": "Los terremotos ocurren por el movimiento de las placas tectónicas.", "b": "El desplazamiento de las placas tectónicas genera sismos.", "etiqueta": 1},
    {"a": "La penicilina fue descubierta por Alexander Fleming en 1928.", "b": "Fleming descubrió la penicilina accidentalmente en su laboratorio en 1928.", "etiqueta": 1},
    {"a": "El sol es una estrella de tipo G ubicada en la Vía Láctea.", "b": "Nuestra estrella el sol pertenece a la categoría G y está en la Vía Láctea.", "etiqueta": 1},
    {"a": "La inflación reduce el poder adquisitivo de la moneda.", "b": "Cuando hay inflación el dinero vale menos y se puede comprar menos.", "etiqueta": 1},
    {"a": "Los mamíferos son animales de sangre caliente que amamantan a sus crías.", "b": "Los mamíferos regulan su temperatura y alimentan a sus crías con leche.", "etiqueta": 1},
    {"a": "El teorema de Pitágoras relaciona los lados de un triángulo rectángulo.", "b": "En un triángulo rectángulo la hipotenusa al cuadrado es igual a la suma de los catetos al cuadrado.", "etiqueta": 1},
    {"a": "La democracia es un sistema de gobierno donde el pueblo elige a sus representantes.", "b": "En la democracia los ciudadanos votan para elegir a quienes los gobiernan.", "etiqueta": 1},
    {"a": "El café contiene cafeína que estimula el sistema nervioso central.", "b": "La cafeína del café actúa como estimulante del sistema nervioso.", "etiqueta": 1},
    {"a": "Los dinosaurios se extinguieron hace aproximadamente 66 millones de años.", "b": "Hace unos 66 millones de años desaparecieron los dinosaurios de la Tierra.", "etiqueta": 1},
    {"a": "La fotocopia funciona usando luz y electricidad estática para transferir tinta.", "b": "Las fotocopiadoras emplean principios electrostáticos y ópticos para reproducir documentos.", "etiqueta": 1},
    {"a": "El sistema inmune protege al cuerpo contra enfermedades e infecciones.", "b": "Las defensas inmunológicas del organismo combaten patógenos y enfermedades.", "etiqueta": 1},
    {"a": "El internet conecta millones de computadoras alrededor del mundo.", "b": "La red de internet une a computadoras y dispositivos en todo el planeta.", "etiqueta": 1},
    # DIVERGENTES etiqueta=0
    {"a": "El agua hierve a 100 grados centígrados a nivel del mar.", "b": "La fotosíntesis convierte luz solar en energía química en las plantas.", "etiqueta": 0},
    {"a": "Python es un lenguaje de programación interpretado.", "b": "Los antibióticos combaten infecciones causadas por bacterias.", "etiqueta": 0},
    {"a": "El corazón bombea sangre por todo el cuerpo.", "b": "La velocidad de la luz es aproximadamente 300,000 kilómetros por segundo.", "etiqueta": 0},
    {"a": "La democracia es un sistema donde el pueblo elige representantes.", "b": "Los volcanes liberan magma desde el interior de la Tierra.", "etiqueta": 0},
    {"a": "El ADN contiene la información genética de los seres vivos.", "b": "La inflación reduce el poder adquisitivo de la moneda.", "etiqueta": 0},
    {"a": "Los mamíferos amamantan a sus crías.", "b": "El teorema de Pitágoras relaciona los lados de un triángulo.", "etiqueta": 0},
    {"a": "El sol es una estrella de tipo G.", "b": "Los terremotos ocurren por el movimiento de placas tectónicas.", "etiqueta": 0},
    {"a": "La penicilina fue descubierta por Fleming en 1928.", "b": "El café contiene cafeína que estimula el sistema nervioso.", "etiqueta": 0},
    {"a": "Los dinosaurios se extinguieron hace 66 millones de años.", "b": "El internet conecta millones de computadoras en el mundo.", "etiqueta": 0},
    {"a": "El oxígeno es esencial para la respiración celular.", "b": "La luna tarda 27 días en girar alrededor de la Tierra.", "etiqueta": 0},
    # OPUESTOS etiqueta=0
    {"a": "Los antibióticos son efectivos para tratar infecciones virales.", "b": "Los antibióticos no funcionan contra virus, solo contra bacterias.", "etiqueta": 0},
    {"a": "El ejercicio regular aumenta el riesgo de enfermedades cardíacas.", "b": "El ejercicio regular reduce el riesgo de enfermedades cardíacas.", "etiqueta": 0},
    {"a": "Comer azúcar causa directamente diabetes tipo 2.", "b": "La diabetes tipo 2 no es causada directamente por comer azúcar.", "etiqueta": 0},
    {"a": "Las vacunas causan autismo según estudios recientes.", "b": "No existe evidencia científica que relacione las vacunas con el autismo.", "etiqueta": 0},
    {"a": "El calentamiento global es un fenómeno natural sin influencia humana.", "b": "El calentamiento global está causado principalmente por actividades humanas.", "etiqueta": 0},
    {"a": "Toda la energía renovable es completamente libre de emisiones.", "b": "Las energías renovables también generan emisiones durante su fabricación.", "etiqueta": 0},
    {"a": "El cerebro humano adulto no puede generar nuevas neuronas.", "b": "Estudios recientes muestran que el cerebro adulto puede generar nuevas neuronas.", "etiqueta": 0},
    {"a": "Los humanos solo usamos el 10 por ciento de nuestro cerebro.", "b": "Los humanos usamos prácticamente todas las regiones del cerebro.", "etiqueta": 0},
    {"a": "La evolución es solo una teoría sin evidencia sólida.", "b": "La teoría de la evolución está respaldada por evidencia científica abundante.", "etiqueta": 0},
    {"a": "Tomar vitamina C en grandes dosis previene los resfriados.", "b": "No hay evidencia de que altas dosis de vitamina C prevengan los resfriados.", "etiqueta": 0},
]

textos_a = [p["a"] for p in pares]
textos_b = [p["b"] for p in pares]
labels   = [p["etiqueta"] for p in pares]

emb_a = model.encode(textos_a, convert_to_tensor=True)
emb_b = model.encode(textos_b, convert_to_tensor=True)
cosenos = util.cos_sim(emb_a, emb_b).diagonal().tolist()

auc = roc_auc_score(labels, cosenos)
pos = [cosenos[i] for i in range(len(labels)) if labels[i]==1]
neg = [cosenos[i] for i in range(len(labels)) if labels[i]==0]

print(f"\nAUC real: {max(auc, roc_auc_score(labels, [-c for c in cosenos])):.4f}")
print(f"Coherentes  mu={sum(pos)/len(pos):.3f}  n={len(pos)}")
print(f"Divergentes mu={sum(neg)/len(neg):.3f}  n={len(neg)}")
print(f"Separacion:    {sum(pos)/len(pos) - sum(neg)/len(neg):.3f}")

print("\nPrimeros 5 pares:")
for i in range(5):
    print(f"  cos={cosenos[i]:.3f} etiqueta={labels[i]}: {textos_a[i][:50]}...")
