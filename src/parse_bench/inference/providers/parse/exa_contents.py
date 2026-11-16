"""Provider for the Exa Contents API (``POST https://api.exa.ai/contents``).

Sends the doc's Hugging Face URL (see ``_url_extract_base``) and scores
``results[0].text``. Cost comes from the response's own ``costDollars.total``.
Recommended ``--max_concurrent 10``.
"""

from typing import Any

from parse_bench.inference.providers.base import ProviderPermanentError
from parse_bench.inference.providers.parse._url_extract_base import UrlExtractProvider
from parse_bench.inference.providers.registry import register_provider

_DEFAULT_ENDPOINT = "https://api.exa.ai/contents"


@register_provider("exa_contents")
class ExaContentsProvider(UrlExtractProvider):
    display_name = "Exa contents"
    api_key_env = "EXA_API_KEY"

    def _call(self, url: str) -> tuple[dict[str, Any], dict[str, Any], str]:
        endpoint = self.base_config.get("endpoint", _DEFAULT_ENDPOINT)
        body = {
            "urls": [url],
            "text": True,
            # 0 forces a live crawl; `livecrawl` is deprecated in favor of maxAgeHours.
            "maxAgeHours": 0,
            "livecrawlTimeout": 60000,
        }
        data = self._post(endpoint, {"x-api-key": self._api_key}, body)
        results = data.get("results") or []
        if not results:
            # Per-URL failures come back as HTTP 200 with the reason in statuses[].
            status = (data.get("statuses") or [{}])[0]
            raise ProviderPermanentError(f"{self.display_name} failed for {url}: {status.get('error') or status}")
        return body, data, results[0].get("text") or ""

    def _cost(self, raw_output: dict[str, Any]) -> dict[str, Any]:
        total = (raw_output["response"].get("costDollars") or {}).get("total")
        return {} if total is None else {"cost_usd": total}
