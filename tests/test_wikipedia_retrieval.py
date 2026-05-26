from unittest.mock import patch

from filtering.retrieval.wikipedia import retrieve_passages


@patch("filtering.retrieval.wikipedia._api_search")
def test_retrieve_passages_returns_snippets(mock_search, tmp_path):
    mock_search.return_value = ["Soccer is a sport.", "Football history."]
    passages = retrieve_passages(
        "What sport is this?", "soccer", k=2, cache_dir=tmp_path
    )
    assert len(passages) == 2
    passages2 = retrieve_passages(
        "What sport is this?", "soccer", k=2, cache_dir=tmp_path
    )
    assert passages2 == passages
    mock_search.assert_called_once()
