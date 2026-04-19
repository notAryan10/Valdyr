from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge"
FAISS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "faiss_index"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vector_store():
    loader = DirectoryLoader(
        str(KNOWLEDGE_DIR), glob="*.md", loader_cls=TextLoader
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(chunks, get_embeddings())
    db.save_local(str(FAISS_PATH))
    return db

def load_vector_store():
    return FAISS.load_local(
        str(FAISS_PATH), get_embeddings(), allow_dangerous_deserialization=True
    )

if __name__ == "__main__":
    print("building vector store...")
    build_vector_store()
    print("done")
