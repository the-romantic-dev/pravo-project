from pathlib import Path

project_dir = Path(__file__).parent.parent

tk_path = project_dir / "data" / "Трудовой_кодекс_Российской_Федерации_от_30_12_2001_N_197_ФЗ.docx"

rag_chunks_path = project_dir / "artifacts" / "tk_chunks_new.jsonl"
definitions_path = project_dir / "data" / "definitions.json"
fiass_index_dir = project_dir / "artifacts" / "faiss_index"
definitions_index_dir = project_dir / "artifacts" / "definitions_faiss_index"
rag_embedding_model = "intfloat/multilingual-e5-small"
rag_embedding_batch_size = 64
model_device = "auto"
reranker_model = "BAAI/bge-reranker-v2-m3"
reranker_batch_size = 16
reranker_max_length = 512
reranker_candidate_multiplier = 10

nli_model = "cointegrated/rubert-base-cased-nli-threeway"
