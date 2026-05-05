import numpy as np
import re

# ─────────────────────────────────────────────────────────────
# CORPUS: 10 artículos del reglamento universitario
# ─────────────────────────────────────────────────────────────
documents = {
    "Art. 1 - Admisiones":
        "para ser admitido como estudiante regular el aspirante debe presentar los documentos "
        "requeridos por la institucion aprobar las pruebas de admision establecidas y cancelar "
        "los derechos de matricula dentro del plazo fijado la institucion podra rechazar "
        "solicitudes que no cumplan los requisitos minimos academicos establecidos",

    "Art. 2 - Matriculas":
        "la matricula es el acto mediante el cual el estudiante formaliza su vinculacion a la "
        "institucion por cada periodo academico el estudiante debe cancelar el valor de la "
        "matricula dentro de las fechas establecidas en el calendario academico el no pago "
        "oportuno genera recargos y puede impedir la inscripcion de asignaturas",

    "Art. 3 - Becas y Financiacion":
        "la institucion ofrece becas por merito academico a los estudiantes con promedio "
        "acumulado superior a cuatro punto cinco tambien existen becas deportivas para "
        "estudiantes que representen a la institucion en competencias nacionales o internacionales "
        "las becas cubren hasta el cincuenta por ciento del valor de la matricula y se renuevan "
        "cada semestre siempre que el estudiante mantenga el promedio exigido",

    "Art. 4 - Evaluaciones":
        "las evaluaciones academicas buscan medir el nivel de competencias adquiridas por el "
        "estudiante durante el periodo lectivo cada asignatura puede tener como maximo cuatro "
        "evaluaciones parciales la nota minima aprobatoria es tres punto cero sobre cinco punto "
        "cero el estudiante tiene derecho a conocer sus calificaciones dentro de los ocho dias "
        "siguientes a la realizacion de cada evaluacion",

    "Art. 5 - Cancelaciones":
        "el estudiante podra solicitar la cancelacion de una asignatura antes de cumplirse el "
        "treinta por ciento del periodo academico sin que ello afecte su promedio las "
        "cancelaciones posteriores a dicha fecha requeriran autorizacion del director de programa "
        "y seran registradas con nota cero la cancelacion total del semestre debe tramitarse "
        "ante la secretaria academica con al menos diez dias habiles de anticipacion",

    "Art. 6 - Faltas Disciplinarias":
        "se consideran faltas disciplinarias graves el fraude en evaluaciones la suplantacion "
        "de identidad el hurto de bienes institucionales y la agresion fisica o verbal a "
        "miembros de la comunidad universitaria las faltas leves incluyen el incumplimiento "
        "reiterado de horarios el uso inadecuado de las instalaciones y la falta de respeto "
        "a los reglamentos internos",

    "Art. 7 - Sanciones":
        "las sanciones por faltas disciplinarias pueden ser amonestacion escrita matricula "
        "condicional suspension temporal de actividades academicas o expulsion definitiva "
        "de la institucion la sancion se determinara segun la gravedad de la falta cometida "
        "y los antecedentes disciplinarios del estudiante todo proceso disciplinario garantiza "
        "el derecho a la defensa y al debido proceso",

    "Art. 8 - Grados":
        "para optar al titulo de pregrado el estudiante debe haber aprobado la totalidad de "
        "las asignaturas del plan de estudios haber cumplido el servicio social obligatorio "
        "y haber presentado y aprobado el trabajo de grado la ceremonia de grados se realiza "
        "dos veces al anio el estudiante debe estar a paz y salvo con la institucion en todos "
        "los conceptos financieros y academicos",

    "Art. 9 - Transferencias":
        "los estudiantes de otras instituciones pueden solicitar transferencia externa presentando "
        "el certificado de notas oficial el programa de estudios cursado y una carta de "
        "motivacion la homologacion de asignaturas sera evaluada por el comite curricular "
        "del programa receptor se aceptan transferencias unicamente al inicio de cada "
        "periodo academico y sujetas a disponibilidad de cupos",

    "Art. 10 - Derechos Estudiantiles":
        "todo estudiante tiene derecho a recibir educacion de calidad a ser tratado con "
        "respeto y dignidad a conocer oportunamente los reglamentos y normativas vigentes "
        "a participar en los organos de gobierno estudiantil y a presentar peticiones "
        "quejas y recursos ante las instancias competentes la institucion garantiza "
        "mecanismos de atencion y solucion oportuna de las solicitudes estudiantiles",
}

# ─────────────────────────────────────────────────────────────
# PREPROCESAMIENTO
# ─────────────────────────────────────────────────────────────
stopwords = {
    'el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'en', 'a', 'y', 'o',
    'que', 'por', 'para', 'con', 'se', 'su', 'sus', 'al', 'es', 'son', 'ha',
    'no', 'lo', 'le', 'si', 'pero', 'ante', 'sobre', 'todo', 'toda', 'todos',
    'este', 'esta', 'hasta', 'desde', 'muy', 'e', 'como', 'cada', 'tambien'
}

def tokenizar(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-z\s]', '', texto)
    return [t for t in texto.split() if t not in stopwords and len(t) > 2]

# Tokenizar todo el corpus una sola vez
corpus_tokens = {nombre: tokenizar(texto) for nombre, texto in documents.items()}
N = len(corpus_tokens)  # total de documentos = 10

# ─────────────────────────────────────────────────────────────
# FUNCIÓN 1 — calcular_tf
# TF = frecuencia_palabra / total_palabras_doc
# ─────────────────────────────────────────────────────────────
def calcular_tf(termino, doc_tokens):
    total_palabras_doc = len(doc_tokens)
    frecuencia_palabra = doc_tokens.count(termino)
    return frecuencia_palabra / total_palabras_doc

# ─────────────────────────────────────────────────────────────
# FUNCIÓN 2 — calcular_idf
# IDF = log(Total_Docs / Docs_con_la_palabra)
# ─────────────────────────────────────────────────────────────
def calcular_idf(termino, corpus_tokens):
    Total_Docs = len(corpus_tokens)
    Docs_con_la_palabra = sum(1 for tokens in corpus_tokens.values() if termino in tokens)
    if Docs_con_la_palabra == 0:
        return 0.0
    return np.log(Total_Docs / Docs_con_la_palabra)

# ─────────────────────────────────────────────────────────────
# FUNCIÓN 3 — calcular_score_final
# Score = suma de TF-IDF de cada palabra de la consulta
# ─────────────────────────────────────────────────────────────
def calcular_score_final(consulta_tokens, doc_tokens, corpus_tokens):
    score_final = 0.0
    for termino in consulta_tokens:
        tf  = calcular_tf(termino, doc_tokens)
        idf = calcular_idf(termino, corpus_tokens)
        score_final += tf * idf
    return round(score_final, 6)

# ─────────────────────────────────────────────────────────────
# MOTOR DE BÚSQUEDA
# ─────────────────────────────────────────────────────────────
def buscar(consulta):
    consulta_tokens = tokenizar(consulta)
    scores = {}
    for nombre_doc, doc_tokens in corpus_tokens.items():
        scores[nombre_doc] = calcular_score_final(consulta_tokens, doc_tokens, corpus_tokens)
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return consulta_tokens, scores, ranking

# ─────────────────────────────────────────────────────────────
# ANÁLISIS IDF GLOBAL
# ─────────────────────────────────────────────────────────────
todas_palabras = set(p for tokens in corpus_tokens.values() for p in tokens)
idf_global   = {p: calcular_idf(p, corpus_tokens) for p in todas_palabras}
idf_ordenado = sorted(idf_global.items(), key=lambda x: x[1], reverse=True)

print("=" * 65)
print("  MOTOR DE BÚSQUEDA TF-IDF — REGLAMENTO UNIVERSITARIO")
print("=" * 65)

print("\n TOP 5 — IDF MAS ALTO (palabras más específicas):")
print(f"   {'#':<4} {'Palabra':<22} {'IDF':>8}")
print(f"   {'-'*36}")
for i, (palabra, valor) in enumerate(idf_ordenado[:5], 1):
    print(f"   {i:<4} {palabra:<22} {valor:>8.4f}")

print("\n TOP 5 — IDF MAS BAJO (palabras más comunes):")
print(f"   {'#':<4} {'Palabra':<22} {'IDF':>8}")
print(f"   {'-'*36}")
for i, (palabra, valor) in enumerate(idf_ordenado[-5:], 1):
    print(f"   {i:<4} {palabra:<22} {valor:>8.4f}")

# ─────────────────────────────────────────────────────────────
# 3 CASOS DE PRUEBA
# ─────────────────────────────────────────────────────────────
consultas = [
    "becas merito academico promedio",
    "faltas disciplinarias fraude evaluaciones sanciones",
    "cancelacion asignatura semestre",
]

for consulta in consultas:
    print(f"\n{'='*65}")
    print(f"  CONSULTA: \"{consulta}\"")
    tokens, scores, ranking = buscar(consulta)
    print(f"  Terminos procesados: {tokens}")

    print(f"\n  TOP 3 DOCUMENTOS:")
    medallas = ["1ro", "2do", "3ro"]
    for i, (doc, score) in enumerate(ranking[:3]):
        print(f"  {medallas[i]}  {doc:<35}  Score: {score:.6f}")

    print(f"\n  TABLA COMPARATIVA:")
    print(f"  {'Documento':<38} {'Score':>10}   Barra")
    print(f"  {'-'*68}")
    max_score = ranking[0][1] if ranking[0][1] > 0 else 1
    for doc, score in ranking:
        barra = chr(9608) * int((score / max_score) * 25)
        print(f"  {doc:<38} {score:>10.6f}   {barra}")

print(f"\n{'='*65}")
print("  Ejecucion completada.")
print(f"{'='*65}\n")

# ─────────────────────────────────────────────────────────────
# MODO INTERACTIVO
# ─────────────────────────────────────────────────────────────
print("MODO INTERACTIVO — escribe una consulta (o 'salir' para terminar)")
while True:
    consulta = input("\n  Consulta: ").strip()
    if consulta.lower() in ("salir", "exit", "q"):
        print("  Hasta luego.")
        break
    if not consulta:
        continue
    tokens, scores, ranking = buscar(consulta)
    if not any(s > 0 for s in scores.values()):
        print("  Sin resultados. Intenta con otras palabras.")
        continue
    print(f"  Terminos: {tokens}")
    for i, (doc, score) in enumerate(ranking[:3], 1):
        if score > 0:
            print(f"  {i}. {doc:<35}  Score: {score:.6f}")