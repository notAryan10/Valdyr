from src.rag.vector_store import load_vector_store

def get_relevant_docs(query, k=3):
    db = load_vector_store()
    results = db.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])
