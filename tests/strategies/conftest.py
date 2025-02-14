
from typing import Generator
from unittest.mock import MagicMock, Mock, patch

import pytest
from llama_index.core import Document

from factchecker.strategies.advocate_mediator import AdvocateMediatorStrategy

# Fixture to create test documents.
@pytest.fixture
def get_test_documents() -> list[Document]:
    """Create a sequence of LlamaIndex Document objects from text strings."""
    texts = [
        "This is the first test document.",
        "This is the second test document.",
        "This is the third test document."
    ]
    return [Document(text=txt) for txt in texts]

# Autouse fixture that patches expensive operations.
@pytest.fixture(autouse=True)
def patch_expensive_operations() -> Generator:
    """Patch methods that would trigger expensive operations so that they return quickly without side effects."""
    # Create a dummy LLM whose chat method always returns a dummy response.
    dummy_llm = MagicMock()
    dummy_llm.chat.return_value = MagicMock(
        message=MagicMock(content="((SUPPORTS)) Dummy reasoning")
    )
    
    with patch(
        'factchecker.indexing.llama_vector_store_indexer.LlamaVectorStoreIndexer.initialize_index',
        return_value=None
    ), patch(
        'factchecker.indexing.llama_vector_store_indexer.LlamaVectorStoreIndexer.build_index',
        return_value=None
    ), patch(
        'factchecker.retrieval.llama_base_retriever.LlamaBaseRetriever.retrieve',
        return_value=[]
    ), patch(
        'factchecker.steps.advocate.load_llm',
        return_value=dummy_llm
    ), patch(
        'factchecker.steps.evidence.EvidenceStep.gather_evidence',
        return_value=["Dummy evidence"]
    ):
        yield

# Fixture to create a reusable AdvocateMediatorStrategy instance.
@pytest.fixture
def advocate_mediator_strategy(get_test_documents: list[Document]) -> AdvocateMediatorStrategy:
    """Create an AdvocateMediatorStrategy instance with test documents."""
    strategy = AdvocateMediatorStrategy(
        indexer_options_list=[
            {"index_name": f"test_{i}", "documents": get_test_documents}
            for i in range(3)
        ],
        retriever_options_list=[{"top_k": 3} for _ in range(3)],
        advocate_options={
            "system_prompt": "test_system_prompt",
            "label_options": ["SUPPORTS", "PARTIALLY_SUPPORTS", "REFUTES"]
        },
        evidence_options={"min_score": 0.7},
        mediator_options={"system_prompt": "test_system_prompt"},
    )
    return strategy

@pytest.fixture
def mock_llama_retriever() -> Generator[Mock, None, None]:
    """Fixture to mock the LlamaBaseRetriever class."""
    with patch('factchecker.retrieval.llama_base_retriever.LlamaBaseRetriever', autospec=True) as mock:
        retriever = Mock()
        # Mock specific evidence retrieval
        retriever.retrieve.return_value = [
            {"text": "Strong evidence supporting the claim", "score": 0.95},
            {"text": "Moderate evidence supporting the claim", "score": 0.85},
            {"text": "Weak evidence supporting the claim", "score": 0.75}
        ]
        mock.return_value = retriever
        yield mock

@pytest.fixture
def mock_llama_indexer()-> Generator[Mock, None, None]:
    """Fixture to mock the LlamaVectorStoreIndexer class."""
    with patch('factchecker.strategies.advocate_mediator.LlamaVectorStoreIndexer', autospec=True) as mock:
        indexer = Mock()
        indexer.index.return_value = True
        mock.return_value = indexer
        yield mock

@pytest.fixture
def mock_advocate_step()-> Generator[Mock, None, None]:
    """Fixture to mock the AdvocateStep class."""
    with patch('factchecker.strategies.advocate_mediator.AdvocateStep', autospec=True) as mock:
        advocate = Mock()
        advocate.evaluate_claim.return_value = ("SUPPORTS", "Based on strong evidence, this claim is supported")
        mock.return_value = advocate
        yield mock

@pytest.fixture
def mock_mediator_step()-> Generator[Mock, None, None]:
    """Fixture to mock the MediatorStep class."""
    with patch('factchecker.strategies.advocate_mediator.MediatorStep', autospec=True) as mock:
        mediator = Mock()
        mediator.synthesize_verdicts.return_value = "FINAL_SUPPORTS"
        mock.return_value = mediator
        yield mock