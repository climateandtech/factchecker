from factchecker.indexing.ragatouille_colbert_indexer import RagatouilleColBERTIndexer
from factchecker.retrieval.ragatouille_colbert_retriever import RagatouilleColBERTRetriever

def main():

    index_name = 'ragatouille_colbert_experiment_index'

    indexer_options = {
        # 'source_directory': 'data',
        'index_name': index_name, # Name of the index which will be stored in .ragatouille/colbert/indexes
        'index_path': f'.ragatouille/colbert/indexes/{index_name}', # Path to the directory where the index is stored on disk
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

