import pytest
from unittest.mock import Mock, patch, MagicMock
from factchecker.steps.evidence import EvidenceStep
from factchecker.retrieval.llama_base_retriever import LlamaBaseRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore, TextNode

@pytest.fixture
def mock_indexer():
    """Fixture for mocked indexer"""
    mock = MagicMock()
    mock.index = Mock()
    return mock

@pytest.fixture
def mock_evidence() -> list[NodeWithScore]:
    """Fixture for mock evidence nodes."""
    def create_node(text: str, score: float) -> NodeWithScore:
        node = NodeWithScore(node=TextNode(text=text), score=score)
        return node
    return [create_node("Evidence 1", 0.9), create_node("Evidence 2", 0.8)]

@pytest.fixture
def mock_retriever(mock_indexer: MagicMock, mock_evidence: list[NodeWithScore]) -> MagicMock:
    """Fixture for mocked retriever."""
    mock = MagicMock(spec=LlamaBaseRetriever)
    mock.indexer = mock_indexer
    mock.options = {'top_k': 5}
    mock.retrieve.return_value = mock_evidence
    return mock

def test_evidence_initialization(mock_retriever: MagicMock) -> None:
    """Test evidence initialization with different options."""
    options = {
        'query_template': "evidence for: {claim}",
        'min_score': 0.75
    }
    
    evidence_step = EvidenceStep(retriever=mock_retriever, options=options.copy())
    assert evidence_step.query_template == "evidence for: {claim}"
    assert evidence_step.min_score == 0.75

def test_default_options(mock_retriever: MagicMock) -> None:
    """Test default options when none provided."""
    evidence_step = EvidenceStep(retriever=mock_retriever)
    assert evidence_step.query_template == "{claim}"
    assert evidence_step.min_score == 0.0

def test_build_query(mock_retriever: MagicMock) -> None:
    """Test query building from template."""
    evidence_step = EvidenceStep(retriever=mock_retriever)
    claim = "The Earth is round"
    query = evidence_step.build_query(claim)
    assert query == "The Earth is round"

def test_gather_evidence(mock_retriever: MagicMock, mock_evidence: list[NodeWithScore]) -> None:
    """Test evidence gathering process."""
    evidence_step = EvidenceStep(retriever=mock_retriever)
    evidence = evidence_step.gather_evidence("Test claim")
    assert mock_retriever.retrieve.called
    assert len(evidence) == 2
    assert all(isinstance(e, str) for e in evidence)

def test_classify_evidence(mock_retriever: MagicMock) -> None:
    """Test evidence classification and filtering."""
    evidence_step = EvidenceStep(
        retriever=mock_retriever,
        options={'min_score': 0.75})
    mock_evidence = [
        NodeWithScore(node=TextNode(text="High confidence"), score=0.9),
        NodeWithScore(node=TextNode(text="Low confidence"), score=0.6),
        NodeWithScore(node=TextNode(text="Medium confidence"), score=0.8)
    ]
    
    filtered = evidence_step.classify_evidence(mock_evidence)
    assert len(filtered) == 2
    assert all(node.score >= 0.75 for node in filtered)

def test_retriever_error_handling(mock_retriever: MagicMock) -> None:
    """Test handling of retriever errors."""
    evidence_step = EvidenceStep(retriever=mock_retriever)
    mock_retriever.retrieve.side_effect = Exception("Retriever error")
    
    with pytest.raises(Exception):
        evidence_step.gather_evidence("Test claim")

def test_custom_query_template(mock_retriever: MagicMock) -> None:
    """Test custom query template."""
    options = {
        'query_template': "Find evidence about {claim} in scientific papers"
    }
    evidence_step = EvidenceStep(retriever=mock_retriever, options=options)
    query = evidence_step.build_query("climate change")
    assert query == "Find evidence about climate change in scientific papers" 