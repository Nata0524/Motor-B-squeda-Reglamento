from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Cargar base vectorial
db = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)

# 🔥 Modelo correcto (NO usar distilgpt2)
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=200,
    do_sample=False
)

# Pregunta
query = input("Haz una pregunta: ")

# Buscar contexto
docs = db.similarity_search(query, k=3)
context = "\n".join([doc.page_content for doc in docs])

# 🔥 Prompt mejorado
prompt = f"""
Responde únicamente con la información del contexto.
Si no encuentras la respuesta, responde: "No se encuentra en el contexto".

Contexto:
{context}

Pregunta:
{query}

Respuesta:
"""

# Generar respuesta
result = generator(prompt)

print("\n--- CONTEXTO ---\n")
print(context)

print("\n--- RESPUESTA ---\n")
print(result[0]["generated_text"])