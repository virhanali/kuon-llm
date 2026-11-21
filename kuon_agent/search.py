"""Tiny search tools built on the standard library.

- Wikipedia: the official MediaWiki API (search + plaintext extracts).
- DuckDuckGo: the no-JavaScript HTML endpoint.

Both return [] on network errors — the agent degrades gracefully and says
it does not know instead of hallucinating.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass

USER_AGENT = "KuonAgent/0.1 (educational research; https://github.com/virhanali/kuon-llm)"
WIKI_API = "https://en.wikipedia.org/w/api.php"
DDG_HTML = "https://html.duckduckgo.com/html/"


@dataclass
class Source:
    title: str
    url: str
    text: str


def _get(url: str, timeout: float = 12.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def wikipedia_search(query: str, limit: int = 2, extract_chars: int = 2500) -> list[Source]:
    """Search Wikipedia and return the intro extract of the top pages."""
    try:
        search_url = WIKI_API + "?" + urllib.parse.urlencode({
            "action": "query", "list": "search", "srsearch": query,
            "format": "json", "srlimit": limit,
        })
        hits = json.loads(_get(search_url)).get("query", {}).get("search", [])
        sources: list[Source] = []
        for hit in hits:
            title = hit["title"]
            extract_url = WIKI_API + "?" + urllib.parse.urlencode({
                "action": "query", "prop": "extracts", "explaintext": 1,
                "exintro": 0, "redirects": 1, "titles": title, "format": "json",
            })
            pages = json.loads(_get(extract_url)).get("query", {}).get("pages", {})
            text = next(iter(pages.values()), {}).get("extract", "")[:extract_chars]
            if text:
                url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
                sources.append(Source(title=title, url=url, text=text))
        return sources
    except Exception:
        return []


_TAG_RE = re.compile(r"<[^>]+>")
_RESULT_RE = re.compile(
    r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_SNIPPET_RE = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.S)


def _clean(fragment: str) -> str:
    return _TAG_RE.sub("", fragment).replace("&amp;", "&").strip()


def _unwrap_ddg_url(url: str) -> str:
    if "uddg=" in url:
        query = urllib.parse.urlparse(url).query
        target = urllib.parse.parse_qs(query).get("uddg", [""])[0]
        if target:
            return target
    return url


def duckduckgo_search(query: str, limit: int = 5) -> list[Source]:
    """Scrape the no-JS DuckDuckGo results page."""
    try:
        html = _get(DDG_HTML + "?" + urllib.parse.urlencode({"q": query})).decode("utf-8", "replace")
        results = _RESULT_RE.findall(html)
        snippets = _SNIPPET_RE.findall(html)
        sources: list[Source] = []
        for i, (href, title_html) in enumerate(results[:limit]):
            snippet = _clean(snippets[i]) if i < len(snippets) else ""
            sources.append(Source(
                title=_clean(title_html),
                url=_unwrap_ddg_url(href),
                text=snippet,
            ))
        return [s for s in sources if s.text]
    except Exception:
        return []
