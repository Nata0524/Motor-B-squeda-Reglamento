from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Cargar documento
loader = PyPDFLoader("data/documento.pdf")
documents = loader.load()

# 2. Dividir en chunks (más pequeños 🔥)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)
chunks = splitter.split_documents(documents)

# 3. Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Base vectorial
db = FAISS.from_documents(chunks, embeddings)

# 5. Guardar
db.save_local("vector_db")

print("Base de datos creada ✅")