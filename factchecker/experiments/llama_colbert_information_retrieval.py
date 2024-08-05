from factchecker.indexing.llama_colbert_indexer import LlamaColBERTIndexer
from factchecker.retrieval.llama_base_retriever import LlamaBaseRetriever

def main():

    indexer_options = {
    }
    retriever_options = {
        'top_k': 4,
    }

    indexer = LlamaColBERTIndexer(indexer_options)
    retriever = LlamaBaseRetriever(indexer, retriever_options)

    query = "Climate change would have happened without humans"

    nodes = retriever.retrieve(query)

    for node in nodes:
        print(node)

if __name__ == "__main__":
    main()

