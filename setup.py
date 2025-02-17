from setuptools import setup, find_packages

setup(
    name="factchecker",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "scikit-learn>=1.0.0",
        "llama-index-core>=0.11.0.post1",
        "llama-index-embeddings-huggingface>=0.3.0",
        "llama-index-embeddings-ollama>=0.3.0",
        "llama-index-embeddings-openai>=0.2.5",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
        "ragatouille>=0.0.4",
    ],
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "pytest-mock>=3.10.0",
            "pytest-cov>=4.0.0",
            "llama-index-llms-ollama>=0.3.6",
        ],
    },
) 