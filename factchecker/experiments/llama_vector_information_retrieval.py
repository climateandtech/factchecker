from factchecker.indexing.llama_vector_store_indexer import LlamaVectorStoreIndexer
from factchecker.retrieval.llama_base_retriever import LlamaBaseRetriever

def main():

    indexer_options = {
    }
    retriever_options = {
        'top_k': 4,
    }

    indexer = LlamaVectorStoreIndexer(indexer_options)
    retriever = LlamaBaseRetriever(indexer, retriever_options)

    query = "Climate change would have happened without humans"

    nodes = retriever.retrieve(query)

    for node in nodes:
        print(node)

if __name__ == "__main__":
    main()

