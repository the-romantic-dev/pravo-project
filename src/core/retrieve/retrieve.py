import time

from langchain_community.vectorstores import FAISS

from src.config import fiass_index_dir, rag_embedding_model, rag_embedding_batch_size
from src.core.rag.index.embeddings import E5Embeddings


def load_fiass() -> FAISS:
    embeddings = E5Embeddings(rag_embedding_model, batch_size=rag_embedding_batch_size)
    return load_vectorstore(fiass_index_dir, embeddings)


def retrieve_top_k(
        query_text: str,
        faiss: FAISS,
        top_k: int = 5,
) -> list[dict]:
    search_data = faiss.similarity_search_with_score(query_text, k=top_k)
    result = []
    for doc, score in search_data:
        similarity = 1 - score
        normalized_text = doc.page_content
        meta_data = doc.metadata.copy()
        meta_data['normalized_text'] = normalized_text
        result.append({
            'meta_data': meta_data,
            'similarity': similarity
        })
    return result


def load_vectorstore(path: str, embeddings: E5Embeddings) -> FAISS:
    try:
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    except TypeError:
        return FAISS.load_local(path, embeddings)


def print_retrieve_results(query, retrieve_result: dict):
    result = retrieve_result
    meta_data = retrieve_result['meta_data']
    result_score = result['similarity']
    print(f'Цитата из договора:\n{query}')
    print(f'Соотвествующая ей норма из Трудового кодекса (уверенность = {result_score:.2f}):')
    result_text = meta_data['original_text']
    result_hierarchy = meta_data['hierarchy_path']
    for h in result_hierarchy:
        print(h)
    print(result_text)


def retrieve_top_k_list(queries: list[str], faiss: FAISS, top_k: int):
    result = {}
    for q in queries:
        if q in result:
            continue
        result[q] = retrieve_top_k(q, faiss, top_k)
    return result


if __name__ == '__main__':
    query = 'работодатель не обязан выплачивать своевременно и в полном размере причитающуюся работнику заработную плату, а также осуществлять иные выплаты в сроки, установленные в соответствии с трудовым кодексом рф, правилами внутреннего трудового распорядка.'
    # query = 'За выполнение трудовых обязанностей Работнику устанавливается должностной оклад в размере 2000 рублей в месяц' #TODO ОТФИЛЬТРОВАТЬ ОБЩИЕ ПОЛОЖЕНИЯ ИЗ ТК НАХОДИТ ИХ ПО ЭТОМУ ЗАПРОСУ
    # res = retrieve_chunks(query, top_k=1)
    print('Start')
    start = time.perf_counter()
    faiss = load_fiass()
    end_faiss = time.perf_counter()
    print(f'End FAISS for {end_faiss - start:.2f} sec')
    print_retrieve_results(query, retrieve_top_k(query, faiss, top_k=1)[0])
    print(f'End retrieve for {time.perf_counter() - end_faiss:.2f} sec')