from setuptools import setup, find_packages

setup(
    name="factchecker",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "llama-index",
        "tqdm",
        "python-dotenv",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-mock",
            "pytest-cov",
        ],
    },
    python_requires=">=3.8",
) 