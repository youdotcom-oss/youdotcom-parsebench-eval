"""Provider for the Parallel Extract API (``POST https://api.parallel.ai/v1/extract``).

Sends the doc's Hugging Face URL (see ``_url_extract_base``) and scores
``results[0].full_content``. The response has no dollar amount, so cost is a flat
``cost_per_request_usd``; the ``usage`` SKU block stays in ``response`` for re-pricing.
Recommended ``--max_concurrent 10``.
"""

from typing import Any

from parse_bench.inference.providers.base import ProviderPermanentError
from parse_bench.inference.providers.parse._url_extract_base import UrlExtractProvider
from parse_bench.inference.providers.registry import register_provider

_DEFAULT_ENDPOINT = "https://api.parallel.ai/v1/extract"


@register_provider("parallel_extract")
class ParallelExtractProvider(UrlExtractProvider):
    display_name = "Parallel extract"
    api_key_env = "PARALLEL_API_KEY"

    def _call(self, url: str) -> tuple[dict[str, Any], dict[str, Any], str]:
        endpoint = self.base_config.get("endpoint", _DEFAULT_ENDPOINT)
        body = {
            "urls": [url],
            "advanced_settings": {
                "full_content": {"max_chars_per_result": 200000},
                # 600s is the API minimum, so a cached copy up to 10 minutes old is still possible.
                "fetch_policy": {"max_age_seconds": 600, "timeout_seconds": 60},
            },
        }
        data = self._post(endpoint, {"x-api-key": self._api_key}, body)
        results = data.get("results") or []
        if not results:
            # Per-URL failures come back as HTTP 200 with the reason in errors[].
            error = (data.get("errors") or [{}])[0]
            raise ProviderPermanentError(f"{self.display_name} failed for {url}: {error.get('message') or error}")
        return body, data, results[0].get("full_content") or ""

    def _cost(self, raw_output: dict[str, Any]) -> dict[str, Any]:
        rate = self.base_config.get("cost_per_request_usd")
        return {} if rate is None else {"cost_usd": rate}
