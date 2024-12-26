from factchecker.indexing.llama_colbert_indexer import LlamaColBERTIndexer
from factchecker.retrieval.llama_base_retriever import LlamaBaseRetriever

def main():

    indexer_options = {
        # 'top_k': 4,
        # 'index_name': 'test_llama_colbert_index',
        # 'source_directory': 'data',
        # 'show_progress': True,
    }
    retriever_options = {
    }

    indexer = LlamaColBERTIndexer(indexer_options)
    indexer.create_index()

    retriever = LlamaBaseRetriever(indexer, retriever_options)

    query = "Climate change would have happened without humans"

    nodes = retriever.retrieve(query)

    for node in nodes:
        print(node)

if __name__ == "__main__":
    main()

