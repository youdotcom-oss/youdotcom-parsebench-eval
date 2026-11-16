"""Base for extract APIs that only accept a document URL.

ParseBench hands providers a local PDF path. These APIs need a public link, so each
doc is sent as its Hugging Face URL in the llamaindex/ParseBench (or ParseBench-2) dataset
instead of being uploaded. Pipelines must set randomize_external_filename=False, or the runner
replaces the path with a randomly named temp file and the URL can't be derived.
"""

import os
import time
import unicodedata
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse.mistral_ocr import _convert_pipe_tables_to_html
from parse_bench.schemas.parse_output import PageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from parse_bench.schemas.product import ProductType

HF_BASE = "https://huggingface.co/datasets/llamaindex/ParseBench/resolve/"
HF_BASE_V2 = "https://huggingface.co/datasets/llamaindex/ParseBench-2/resolve/"
HF_REVISION_V2 = "a12c9785cd8cd015b5284e0ad0d74722091762f2"
_V2_DIR = "data_parsebench-2"

_COST_KEYS = ("cost_usd", "cost_per_page_usd", "credits_used")

# Cap on a server-requested Retry-After sleep; the runner adds its own backoff on top.
_MAX_RETRY_AFTER_S = 60.0


def hf_url(source_file_path: str, revision: str = "main") -> str:
    parts = Path(source_file_path).parts
    if _V2_DIR in parts:
        # ParseBench 2 is pinned to one revision so the URLs stay reproducible: <subset>/<name>.pdf.
        base, revision = HF_BASE_V2, HF_REVISION_V2
        key = "/".join(parts[len(parts) - parts[::-1].index(_V2_DIR) :])
    else:
        base = HF_BASE
        key = "/".join(parts[-3:])  # docs/<category>/<name>.pdf
        if not key.startswith("docs/"):
            raise ProviderPermanentError(f"not a ParseBench docs path: {source_file_path}")
    # HF matches paths byte-for-byte and the dataset stores accented names decomposed (NFD).
    return f"{base}{revision}/{urllib.parse.quote(unicodedata.normalize('NFD', key))}"


def count_pages(path: str) -> int | None:
    try:
        from pypdf import PdfReader

        return len(PdfReader(path).pages)
    except Exception:
        return None


def raise_for_status(name: str, resp: requests.Response) -> None:
    sc, body = resp.status_code, resp.text[:300]
    if sc in (401, 403):
        raise ProviderConfigError(f"{name} unauthorized ({sc}): {body}")
    if sc == 402:
        raise ProviderPermanentError(f"{name} out of credits (402): {body}")
    if sc == 429:
        retry_after = resp.headers.get("Retry-After")
        try:
            wait = float(retry_after) if retry_after else 0.0
        except ValueError:
            wait = 0.0
        if wait > 0:
            time.sleep(min(wait, _MAX_RETRY_AFTER_S))
        raise ProviderRateLimitError(f"{name} rate limited (429): {body}")
    if sc >= 500:
        raise ProviderTransientError(f"{name} server error ({sc}): {body}")
    if sc >= 400:
        raise ProviderPermanentError(f"{name} client error ({sc}): {body}")


class UrlExtractProvider(Provider):
    """Base for APIs that take a document URL and return markdown."""

    display_name: str = ""
    api_key_env: str = ""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        key = self.base_config.get("api_key") or os.getenv(self.api_key_env)
        if not key:
            raise ProviderConfigError(f"Set the {self.api_key_env} environment variable.")
        self._api_key = str(key)
        # HTTP read timeout; must exceed the server-side timeout each provider sends in its body.
        self._timeout = int(self.base_config.get("timeout", 150))

    def _call(self, url: str) -> tuple[dict[str, Any], dict[str, Any], str]:
        """Return (request_body, response_json, markdown)."""
        raise NotImplementedError

    def _cost(self, raw_output: dict[str, Any]) -> dict[str, Any]:
        """Cost keys for one saved doc, derived only from raw_output and config. Default: usage only."""
        return {}

    def _post(self, endpoint: str, headers: dict[str, str], body: dict[str, Any]) -> Any:
        try:
            resp = requests.post(endpoint, headers=headers, json=body, timeout=self._timeout)
        except requests.exceptions.RequestException as e:
            raise ProviderTransientError(f"{self.display_name} request error: {e}") from e
        raise_for_status(self.display_name, resp)
        try:
            data = resp.json()
        except ValueError as e:
            raise ProviderTransientError(f"{self.display_name} returned non-JSON: {resp.text[:300]}") from e
        if not isinstance(data, (dict, list)):
            raise ProviderPermanentError(f"{self.display_name} returned unexpected JSON: {str(data)[:300]}")
        return data

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"{self.display_name} only supports PARSE, got {request.product_type}")
        url = hf_url(request.source_file_path, self.base_config.get("hf_revision", "main"))
        started_at = datetime.now()
        body, response, markdown = self._call(url)
        completed_at = datetime.now()
        if not markdown.strip():
            raise ProviderPermanentError(f"{self.display_name} returned no content for {url}")
        num_pages = count_pages(request.source_file_path)
        raw_output: dict[str, Any] = {
            "url": url,
            "request_body": body,
            "response": response,
            "markdown": markdown,
            "num_pages": num_pages,
            "num_pages_billed": num_pages,
        }
        raw_output.update(self._cost(raw_output))
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output=raw_output,
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
        )

    def recompute_cost(self, raw_output: dict[str, Any]) -> None:
        if "response" not in raw_output:
            return
        for key in _COST_KEYS:
            raw_output.pop(key, None)
        raw_output.update(self._cost(raw_output))

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        markdown = _convert_pipe_tables_to_html(raw_result.raw_output.get("markdown", ""))
        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=[PageIR(page_index=0, markdown=markdown)],
            layout_pages=[],
            markdown=markdown,
        )
        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw_result.raw_output,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )
