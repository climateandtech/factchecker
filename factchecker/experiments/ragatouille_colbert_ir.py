from factchecker.indexing.ragatouille_colbert_indexer import RagatouilleColBERTIndexer
from factchecker.retrieval.ragatouille_colbert_retriever import RagatouilleColBERTRetriever

def main():

    index_name = 'ragatouille_colbert_experiment_index'

    indexer_options = {
        'source_directory': 'data',
        'index_name': index_name,
        'index_path': f'indexes/ragatouille/colbert/indexes/{index_name}', 
        'overwrite_index': True,
    }
    retriever_options = {
        'top_k': 4,
    }


    indexer = RagatouilleColBERTIndexer(indexer_options)
    indexer.initialize_index()

    retriever = RagatouilleColBERTRetriever(indexer, retriever_options)

    query = "Climate change would have happened without humans"

    nodes = retriever.retrieve(query)

    for node in nodes:
        print(node)

if __name__ == "__main__":
    main()

