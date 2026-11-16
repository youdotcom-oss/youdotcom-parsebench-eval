"""Provider for the you.com Extract API (``POST https://ydc-index.io/v1/extract``).

Sends the doc's Hugging Face URL (see ``_url_extract_base``) and scores
``contents.full_page.markdown``. The response reports no usage, so cost is
``cost_per_request_usd + num_pages * cost_per_pdf_page_usd`` when both rates
are configured; the doc is left unpriced otherwise (a page rate of 0 needs no
page count). Recommended ``--max_concurrent 10``.
"""

from typing import Any

from parse_bench.inference.providers.parse._url_extract_base import UrlExtractProvider
from parse_bench.inference.providers.registry import register_provider

_DEFAULT_ENDPOINT = "https://ydc-index.io/v1/extract"


@register_provider("youcom_extract")
class YouComExtractProvider(UrlExtractProvider):
    display_name = "you.com extract"
    api_key_env = "YOU_API_KEY"

    def _call(self, url: str) -> tuple[dict[str, Any], dict[str, Any], str]:
        endpoint = self.base_config.get("endpoint", _DEFAULT_ENDPOINT)
        body = {
            "url": url,
            # html is stored in raw.json for a possible HTML arm; only markdown is scored.
            "full_page": {"extraction_formats": ["markdown", "html"]},
            # API maximum
            "timeout": 100,
            # max_age 0 requests a fresh crawl and parse instead of a cached result,
            # so scoring never sees a stale pipeline output.
            "max_age": 0,
        }
        # Only set when configured, so the default standard-tier body stays unchanged.
        effort = self.base_config.get("parse_effort")
        if effort:
            body["parse_effort"] = effort
        data = self._post(endpoint, {"X-Api-Key": self._api_key}, body)
        markdown = ((data.get("contents") or {}).get("full_page") or {}).get("markdown") or ""
        return body, data, markdown

    def _cost(self, raw_output: dict[str, Any]) -> dict[str, Any]:
        url_rate = self.base_config.get("cost_per_request_usd")
        page_rate = self.base_config.get("cost_per_pdf_page_usd")
        pages = raw_output.get("num_pages")
        # A per-page fee can't be priced without a page count; leave the doc unpriced.
        if url_rate is None or page_rate is None or (page_rate and not pages):
            return {}
        cost = url_rate + page_rate * (pages or 0)
        return {"cost_usd": cost, "cost_per_page_usd": cost / pages} if pages else {"cost_usd": cost}
