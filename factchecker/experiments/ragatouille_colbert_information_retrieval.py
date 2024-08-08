from factchecker.indexing.ragatouille_colbert_indexer import RagatouilleColBERTIndexer
from factchecker.retrieval.ragatouille_colbert_retriever import RagatouilleColBERTRetriever

def main():
    indexer_options = {
        # 'source_directory': 'data',
        'index_name': 'ragatouille_colbert_experiment_index', # Name of the index which will be stored in .ragatouille/colbert/indexes
        # 'max_document_length': 180, # Longer documents will be split into chunks
        # 'split_documents': True, # Split documents into chunks
        # 'checkpoint': 'colbert-ir/colbertv2.0'  # Pretrained model checkpoint
    }
    retriever_options = {
    }

    indexer = RagatouilleColBERTIndexer(indexer_options)
    indexer.create_index()

    retriever = RagatouilleColBERTRetriever(indexer, retriever_options)

    query = "Climate change would have happened without humans"

    nodes = retriever.retrieve(query)

    for node in nodes:
        print(node)
        print("\n")

if __name__ == "__main__":
    main()

