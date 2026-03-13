from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.core.rag.index.embeddings import E5Embeddings
from src.core.util.jsonl import load_jsonl
from src.config import rag_chunks_path, fiass_index_dir, rag_embedding_model, rag_embedding_batch_size


def build_index(

):
    chunks = load_jsonl(rag_chunks_path)
    docs: list[Document] = []
    for chunk in chunks:
        text = chunk["normalized_text"].strip()
        meta = {k: v for k, v in chunk.items() if k != "normalized_text"}
        docs.append(Document(page_content=text, metadata=meta))

    embeddings = E5Embeddings(rag_embedding_model, batch_size=rag_embedding_batch_size)
    vs = FAISS.from_documents(docs, embeddings)

    fiass_index_dir.parent.mkdir(parents=True, exist_ok=True)
    vs.save_local(fiass_index_dir.as_posix())

if __name__ == '__main__':
    build_index()
