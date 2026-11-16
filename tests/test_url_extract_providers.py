"""Offline tests for the URL-based extract providers (HTTP is mocked)."""

from __future__ import annotations

import copy
import unicodedata
from datetime import datetime
from typing import Any

import pytest
import requests

from parse_bench.inference.pipelines import get_pipeline
from parse_bench.inference.providers.base import (
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse import _url_extract_base as base_mod
from parse_bench.inference.providers.parse._url_extract_base import (
    HF_BASE,
    HF_BASE_V2,
    HF_REVISION_V2,
    hf_url,
    raise_for_status,
)
from parse_bench.inference.providers.parse.exa_contents import ExaContentsProvider
from parse_bench.inference.providers.parse.parallel_extract import ParallelExtractProvider
from parse_bench.inference.providers.parse.tavily_extract import TavilyExtractProvider
from parse_bench.inference.providers.parse.youcom_extract import YouComExtractProvider
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult
from parse_bench.schemas.product import ProductType

URL = HF_BASE + "main/docs/table/x.pdf"
PIPELINES = (
    "youcom_extract",
    "exa_contents",
    "parallel_extract",
    "tavily_extract",
)


class _FakeResponse:
    def __init__(self, status: int, payload: Any = None, headers: dict[str, str] | None = None):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}
        self.text = str(self._payload)

    def json(self) -> Any:
        return self._payload


def _mock_post(monkeypatch: pytest.MonkeyPatch, status: int, payload: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_post(endpoint: str, headers: dict[str, str], json: dict[str, Any], timeout: int) -> _FakeResponse:
        calls.append({"endpoint": endpoint, "headers": headers, "json": json, "timeout": timeout})
        return _FakeResponse(status, payload)

    monkeypatch.setattr(base_mod.requests, "post", fake_post)
    return calls


def _raw_result(raw_output: dict[str, Any]) -> RawInferenceResult:
    now = datetime.now()
    pipeline = get_pipeline("youcom_extract")
    return RawInferenceResult(
        request=InferenceRequest(
            example_id="ex", source_file_path="/d/docs/table/x.pdf", product_type=ProductType.PARSE
        ),
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type=ProductType.PARSE,
        raw_output=raw_output,
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )


# --- HF URL mapping --------------------------------------------------------


def test_hf_url_encodes_spaces() -> None:
    assert hf_url("/x/data/docs/table/SERFF_TX_random_pages 1_page1.pdf") == (
        HF_BASE + "main/docs/table/SERFF_TX_random_pages%201_page1.pdf"
    )


def test_hf_url_uses_last_three_parts_and_revision() -> None:
    assert hf_url("/a/b/data/test/docs/chart/c.pdf", "test-data") == HF_BASE + "test-data/docs/chart/c.pdf"


@pytest.mark.parametrize("form", ["NFC", "NFD"])
def test_hf_url_nfd_normalizes(form: str) -> None:
    path = unicodedata.normalize(form, "/d/docs/table/245804406-CV-Nevers-signé-EX_page10.pdf")
    assert hf_url(path) == HF_BASE + "main/docs/table/245804406-CV-Nevers-signe%CC%81-EX_page10.pdf"


def test_hf_url_parsebench_2_uses_pinned_repo_and_subset_path() -> None:
    expected = HF_BASE_V2 + HF_REVISION_V2 + "/forms/blank__a--p2.pdf"
    assert hf_url("/r/data_parsebench-2/forms/blank__a--p2.pdf") == expected
    assert hf_url("data_parsebench-2/forms/blank__a--p2.pdf", "main") == expected


def test_hf_url_rejects_non_docs_path() -> None:
    with pytest.raises(ProviderPermanentError):
        hf_url("/tmp/abc123/random.pdf")


# --- status mapping --------------------------------------------------------


@pytest.mark.parametrize(
    "status,exc",
    [
        (401, ProviderConfigError),
        (403, ProviderConfigError),
        (402, ProviderPermanentError),
        (429, ProviderRateLimitError),
        (500, ProviderTransientError),
        (503, ProviderTransientError),
        (504, ProviderTransientError),
        (422, ProviderPermanentError),
        (432, ProviderPermanentError),
        (433, ProviderPermanentError),
    ],
)
def test_status_mapping(status: int, exc: type[Exception]) -> None:
    with pytest.raises(exc):
        raise_for_status("x", _FakeResponse(status))  # type: ignore[arg-type]


def test_status_ok_does_not_raise() -> None:
    raise_for_status("x", _FakeResponse(200))  # type: ignore[arg-type]


def test_429_honors_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(base_mod.time, "sleep", slept.append)
    with pytest.raises(ProviderRateLimitError):
        raise_for_status("x", _FakeResponse(429, headers={"Retry-After": "7"}))  # type: ignore[arg-type]
    assert slept == [7.0]


def test_network_error_is_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr(base_mod.requests, "post", boom)
    with pytest.raises(ProviderTransientError):
        YouComExtractProvider("youcom_extract", {"api_key": "k"})._call(URL)


def test_missing_api_key_is_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("YOU_API_KEY", raising=False)
    with pytest.raises(ProviderConfigError):
        YouComExtractProvider("youcom_extract", {})


# --- request bodies and markdown extraction --------------------------------


def test_youcom_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _mock_post(monkeypatch, 200, {"contents": {"full_page": {"markdown": "# hi", "html": "<h1>hi</h1>"}}})
    body, _, md = YouComExtractProvider("youcom_extract", {"api_key": "k"})._call(URL)
    assert md == "# hi"
    assert calls[0]["endpoint"] == "https://ydc-index.io/v1/extract"
    assert calls[0]["headers"] == {"X-Api-Key": "k"}
    assert body == {
        "url": URL,
        "full_page": {"extraction_formats": ["markdown", "html"]},
        "timeout": 100,
        "max_age": 0,
    }
    assert calls[0]["timeout"] > body["timeout"]


def test_youcom_call_lite_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _mock_post(monkeypatch, 200, {"contents": {"full_page": {"markdown": "# hi"}}})
    body, _, md = YouComExtractProvider("youcom_extract", {"api_key": "k", "parse_effort": "lite"})._call(URL)
    assert md == "# hi"
    assert body["parse_effort"] == "lite"
    assert calls[0]["json"]["parse_effort"] == "lite"


def test_exa_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _mock_post(monkeypatch, 200, {"results": [{"text": "exa md"}], "costDollars": {"total": 0.001}})
    body, _, md = ExaContentsProvider("exa_contents", {"api_key": "k"})._call(URL)
    assert md == "exa md"
    assert calls[0]["endpoint"] == "https://api.exa.ai/contents"
    assert calls[0]["headers"] == {"x-api-key": "k"}
    assert body == {"urls": [URL], "text": True, "maxAgeHours": 0, "livecrawlTimeout": 60000}


def test_exa_status_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_post(monkeypatch, 200, {"results": [], "statuses": [{"id": URL, "status": "error", "error": "CRAWL_FAIL"}]})
    with pytest.raises(ProviderPermanentError, match="CRAWL_FAIL"):
        ExaContentsProvider("exa_contents", {"api_key": "k"})._call(URL)


def test_parallel_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _mock_post(monkeypatch, 200, {"results": [{"full_content": "par md"}], "errors": []})
    body, _, md = ParallelExtractProvider("parallel_extract", {"api_key": "k"})._call(URL)
    assert md == "par md"
    assert calls[0]["endpoint"] == "https://api.parallel.ai/v1/extract"
    assert calls[0]["headers"] == {"x-api-key": "k"}
    assert body["advanced_settings"]["fetch_policy"] == {"max_age_seconds": 600, "timeout_seconds": 60}
    assert body["advanced_settings"]["full_content"] == {"max_chars_per_result": 200000}


def test_parallel_errors_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_post(monkeypatch, 200, {"results": [], "errors": [{"url": URL, "message": "fetch failed"}]})
    with pytest.raises(ProviderPermanentError, match="fetch failed"):
        ParallelExtractProvider("parallel_extract", {"api_key": "k"})._call(URL)


def test_tavily_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _mock_post(monkeypatch, 200, {"results": [{"raw_content": "tav md"}], "failed_results": []})
    body, _, md = TavilyExtractProvider("tavily_extract", {"api_key": "k"})._call(URL)
    assert md == "tav md"
    assert calls[0]["endpoint"] == "https://api.tavily.com/extract"
    assert calls[0]["headers"] == {"Authorization": "Bearer k"}
    assert body["extract_depth"] == "advanced" and body["include_usage"] is True
    assert body["timeout"] <= 60


def test_tavily_failed_results_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_post(monkeypatch, 200, {"results": [], "failed_results": [{"error": "Failed to fetch url"}]})
    with pytest.raises(ProviderPermanentError, match="Failed to fetch url"):
        TavilyExtractProvider("tavily_extract", {"api_key": "k"})._call(URL)


# --- run_inference / normalize ---------------------------------------------


def _request(tmp_path: Any) -> InferenceRequest:
    pdf = tmp_path / "docs" / "table" / "x.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"not a real pdf")
    return InferenceRequest(example_id="ex", source_file_path=str(pdf), product_type=ProductType.PARSE)


def test_empty_markdown_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _mock_post(monkeypatch, 200, {"contents": {"full_page": {"markdown": "  \n"}}})
    provider = YouComExtractProvider("youcom_extract", {"api_key": "k"})
    with pytest.raises(ProviderPermanentError, match="no content"):
        provider.run_inference(get_pipeline("youcom_extract"), _request(tmp_path))


def test_run_inference_raw_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _mock_post(monkeypatch, 200, {"results": [{"text": "md"}], "costDollars": {"total": 0.001}})
    monkeypatch.setattr(base_mod, "count_pages", lambda path: 2)
    provider = ExaContentsProvider("exa_contents", {"api_key": "secret-key"})
    raw = provider.run_inference(get_pipeline("exa_contents"), _request(tmp_path))
    out = raw.raw_output
    assert out["url"] == URL and out["markdown"] == "md"
    assert out["num_pages"] == 2 and out["num_pages_billed"] == 2
    assert out["cost_usd"] == 0.001
    assert "secret-key" not in str(out)


def test_normalize_converts_pipe_tables() -> None:
    raw = _raw_result({"markdown": "| a | b |\n|---|---|\n| 1 | 2 |"})
    out = YouComExtractProvider("youcom_extract", {"api_key": "k"}).normalize(raw).output
    assert "<table>" in out.markdown and out.pages[0].markdown == out.markdown
    assert out.layout_pages == []


# --- cost -------------------------------------------------------------------


def _priced(
    name: str,
    provider_cls: type,
    response: dict[str, Any],
    num_pages: int = 3,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provider = provider_cls(name, {"api_key": "k", **get_pipeline(name).config, **(config or {})})
    raw: dict[str, Any] = {"response": response, "num_pages": num_pages, "num_pages_billed": num_pages}
    raw.update(provider._cost(raw))
    return raw


def test_youcom_unpriced_by_default() -> None:
    assert "cost_usd" not in _priced("youcom_extract", YouComExtractProvider, {})


def test_youcom_cost_from_config_rates() -> None:
    provider = YouComExtractProvider(
        "youcom_extract", {"api_key": "k", "cost_per_request_usd": 0.01, "cost_per_pdf_page_usd": 0.002}
    )
    raw: dict[str, Any] = {"response": {}, "num_pages": 3, "num_pages_billed": 3}
    raw.update(provider._cost(raw))
    assert raw["cost_usd"] == pytest.approx(0.01 + 3 * 0.002)
    assert raw["cost_per_page_usd"] == pytest.approx((0.01 + 3 * 0.002) / 3)


def test_youcom_unknown_pages_unpriced() -> None:
    provider = YouComExtractProvider(
        "youcom_extract", {"api_key": "k", "cost_per_request_usd": 0.01, "cost_per_pdf_page_usd": 0.002}
    )
    assert provider._cost({"response": {}, "num_pages": 0}) == {}


def test_youcom_zero_page_rate_needs_no_pages() -> None:
    provider = YouComExtractProvider(
        "youcom_extract", {"api_key": "k", "cost_per_request_usd": 0.01, "cost_per_pdf_page_usd": 0}
    )
    assert provider._cost({"response": {}, "num_pages": None}) == {"cost_usd": 0.01}
    assert provider._cost({"response": {}, "num_pages": 3}) == {
        "cost_usd": 0.01,
        "cost_per_page_usd": pytest.approx(0.01 / 3),
    }


def test_exa_cost_copied_through() -> None:
    assert _priced("exa_contents", ExaContentsProvider, {"costDollars": {"total": 0.0012}})["cost_usd"] == 0.0012
    assert "cost_usd" not in _priced("exa_contents", ExaContentsProvider, {})


def test_parallel_flat_rate() -> None:
    raw = _priced("parallel_extract", ParallelExtractProvider, {}, config={"cost_per_request_usd": 0.001})
    assert raw["cost_usd"] == 0.001


def test_tavily_credits_and_rate() -> None:
    raw = _priced("tavily_extract", TavilyExtractProvider, {"usage": {"credits": 0}}, config={"credit_rate_usd": 0.005})
    assert raw["credits_used"] == 0.4
    assert raw["cost_usd"] == pytest.approx(0.002)


@pytest.mark.parametrize(
    "name,provider_cls",
    [
        ("youcom_extract", YouComExtractProvider),
        ("parallel_extract", ParallelExtractProvider),
        ("tavily_extract", TavilyExtractProvider),
    ],
)
def test_no_rate_writes_no_cost(name: str, provider_cls: type) -> None:
    provider = provider_cls(name, {"api_key": "k"})
    raw = {"response": {}, "num_pages": 1}
    assert "cost_usd" not in provider._cost(raw)


def test_recompute_cost_is_idempotent() -> None:
    provider = TavilyExtractProvider("tavily_extract", {"api_key": "k", "credit_rate_usd": 0.001})
    raw = {"response": {"usage": {"credits": 0}}, "num_pages": 3}
    raw.update(provider._cost(raw))
    first = copy.deepcopy(raw)
    provider.recompute_cost(raw)
    assert raw == first
    provider.recompute_cost(raw)
    assert raw == first


def test_recompute_cost_reprices_and_drops_stale_keys() -> None:
    raw = {"response": {}, "num_pages": 2, "cost_usd": 9.0, "cost_per_page_usd": 4.5}
    YouComExtractProvider("youcom_extract", {"api_key": "k"}).recompute_cost(raw)
    assert "cost_usd" not in raw and "cost_per_page_usd" not in raw
    YouComExtractProvider(
        "youcom_extract", {"api_key": "k", "cost_per_request_usd": 0.01, "cost_per_pdf_page_usd": 0.01}
    ).recompute_cost(raw)
    assert raw["cost_usd"] == pytest.approx(0.03)
    assert raw["cost_per_page_usd"] == pytest.approx(0.015)


# --- registration -----------------------------------------------------------


@pytest.mark.parametrize("name", PIPELINES)
def test_pipelines_registered_without_filename_randomization(name: str) -> None:
    spec = get_pipeline(name)
    assert spec.provider_name == name
    assert spec.product_type == ProductType.PARSE
    assert spec.config["randomize_external_filename"] is False


def test_count_pages_reads_pdf(tmp_path: Any) -> None:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    path = tmp_path / "two.pdf"
    with path.open("wb") as f:
        writer.write(f)
    assert base_mod.count_pages(str(path)) == 2
    assert base_mod.count_pages(str(tmp_path / "missing.pdf")) is None
