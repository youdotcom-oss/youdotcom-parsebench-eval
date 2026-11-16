"""Provider for the Tavily Extract API (``POST https://api.tavily.com/extract``).

Sends the doc's Hugging Face URL (see ``_url_extract_base``) and scores
``results[0].raw_content``. Advanced depth bills 2 credits per 5 successful URLs;
with one URL per request the reported ``usage.credits`` often reads 0, so each
successful doc records 0.4 credits. Recommended ``--max_concurrent 10``.
"""

from typing import Any

from parse_bench.inference.providers.base import ProviderPermanentError
from parse_bench.inference.providers.parse._url_extract_base import UrlExtractProvider
from parse_bench.inference.providers.registry import register_provider

_DEFAULT_ENDPOINT = "https://api.tavily.com/extract"
_CREDITS_PER_URL = 0.4


@register_provider("tavily_extract")
class TavilyExtractProvider(UrlExtractProvider):
    display_name = "Tavily extract"
    api_key_env = "TAVILY_API_KEY"

    def _call(self, url: str) -> tuple[dict[str, Any], dict[str, Any], str]:
        endpoint = self.base_config.get("endpoint", _DEFAULT_ENDPOINT)
        # basic depth fails on every ParseBench PDF; advanced is required. The API has no cache control.
        body = {
            "urls": [url],
            "extract_depth": "advanced",
            "format": "markdown",
            # Reported usage is kept in raw.json for checking; billing uses _CREDITS_PER_URL.
            "include_usage": True,
            # Server-side limit in seconds; the API maximum is 60.
            "timeout": 60,
        }
        data = self._post(endpoint, {"Authorization": f"Bearer {self._api_key}"}, body)
        results = data.get("results") or []
        if not results:
            failed = (data.get("failed_results") or [{}])[0]
            raise ProviderPermanentError(f"{self.display_name} failed for {url}: {failed.get('error') or data}")
        return body, data, results[0].get("raw_content") or ""

    def _cost(self, raw_output: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {"credits_used": _CREDITS_PER_URL}
        rate = self.base_config.get("credit_rate_usd")
        if rate is not None:
            out["cost_usd"] = _CREDITS_PER_URL * rate
        return out
