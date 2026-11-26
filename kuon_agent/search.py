"""Tiny search tools built on the standard library.

- Wikipedia: the official MediaWiki API (search + plaintext extracts).
- DuckDuckGo: the no-JavaScript HTML endpoint.

Both return [] on network errors — the agent degrades gracefully and says
it does not know instead of hallucinating.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass

USER_AGENT = "KuonAgent/0.1 (educational research; https://github.com/virhanali/kuon-llm)"
WIKI_API = "https://en.wikipedia.org/w/api.php"
DDG_HTML = "https://html.duckduckgo.com/html/"
EXA_SEARCH_API = "https://api.exa.ai/search"

_TRUSTED_DOMAIN_SUFFIXES = (
    ".go.id", ".gov", ".edu", ".ac.id", "wikipedia.org", "wikidata.org",
    "bps.go.id", "bi.go.id", "ojk.go.id", "reuters.com", "apnews.com", "bbc.com",
    "antaranews.com", "kompas.com", "tempo.co", "kontan.co.id", "bisnis.com", "cnbcindonesia.com",
)


@dataclass
class Source:
    title: str
    url: str
    text: str


def _get(url: str, timeout: float = 12.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def wikipedia_search(query: str, limit: int = 2, extract_chars: int = 2500, *, lang: str = "en") -> list[Source]:
    """Search Wikipedia (en or id) and return the intro extract of the top pages."""
    wiki_lang = lang if lang in ("en", "id") else "en"
    api_url = f"https://{wiki_lang}.wikipedia.org/w/api.php"
    try:
        search_url = api_url + "?" + urllib.parse.urlencode({
            "action": "query", "list": "search", "srsearch": query,
            "format": "json", "srlimit": limit,
        })
        hits = json.loads(_get(search_url)).get("query", {}).get("search", [])
        sources: list[Source] = []
        for hit in hits:
            title = hit["title"]
            extract_url = api_url + "?" + urllib.parse.urlencode({
                "action": "query", "prop": "extracts", "explaintext": 1,
                "exintro": 0, "redirects": 1, "titles": title, "format": "json",
            })
            pages = json.loads(_get(extract_url)).get("query", {}).get("pages", {})
            text = next(iter(pages.values()), {}).get("extract", "")[:extract_chars]
            if text:
                url = f"https://{wiki_lang}.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
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


def _domain_of(url: str) -> str:
    host = urllib.parse.urlparse(url).netloc.lower()
    return host.removeprefix("www.")


def _is_trusted_domain(domain: str) -> bool:
    for suf in _TRUSTED_DOMAIN_SUFFIXES:
        clean = suf.lstrip(".")
        if domain == clean or domain.endswith("." + clean):
            return True
    return False


def exa_search(
    query: str,
    limit: int = 3,
    *,
    api_key: str | None = None,
    timeout: float = 12.0,
) -> list[Source]:
    """Query Exa AI Search API for clean, dated highlights with multi-domain source diversity.

    Reads `EXA_API_KEY` from environment when `api_key` is omitted. Returns [] gracefully if
    no key is configured or on network/API errors.
    """
    key = (api_key if api_key is not None else os.environ.get("EXA_API_KEY", "")).strip()
    if not key or not query.strip():
        return []
    body = json.dumps({
        "query": query.strip(),
        "type": "auto",
        "numResults": min(max(limit * 2, 5), 10),
        "contents": {
            "highlights": {"numSentences": 3, "highlightsPerUrl": 2},
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        EXA_SEARCH_API,
        data=body,
        headers={
            "x-api-key": key,
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return []

    raw_results = payload.get("results") or []
    candidates: list[tuple[int, str, Source]] = []
    for idx, item in enumerate(raw_results):
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or url or "Web result").strip()
        highlights = item.get("highlights") or []
        if isinstance(highlights, list):
            snippet = " ".join(str(h).strip() for h in highlights if str(h).strip())
        else:
            snippet = str(highlights).strip()
        if not snippet:
            snippet = str(item.get("text") or "").strip()[:800]
        if not snippet:
            continue
        pub_raw = str(item.get("publishedDate") or "").strip()
        pub_date = pub_raw[:10] if len(pub_raw) >= 10 and pub_raw[4:5] == "-" else ""
        if pub_date:
            title = f"{title} ({pub_date})"
        domain = _domain_of(url)
        priority = 0 if _is_trusted_domain(domain) else 1
        candidates.append((priority, domain, Source(title=title, url=url, text=snippet)))

    # Prioritize trusted/official domains and keep at most 1 result per domain for corroboration
    candidates.sort(key=lambda x: x[0])
    seen_domains: set[str] = set()
    chosen: list[Source] = []
    for _prio, domain, src in candidates:
        if domain and domain in seen_domains:
            continue
        if domain:
            seen_domains.add(domain)
        chosen.append(src)
        if len(chosen) >= limit:
            break
    return chosen


def web_search(query: str, limit: int = 3) -> list[Source]:
    """Use Exa AI when `EXA_API_KEY` is configured, falling back to DuckDuckGo otherwise."""
    if os.environ.get("EXA_API_KEY", "").strip():
        hits = exa_search(query, limit=limit)
        if hits:
            return hits
    return duckduckgo_search(query, limit=limit)
