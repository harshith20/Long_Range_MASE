"""Wikipedia access for M3: page text, disambiguation pages, and linked co-occurrence counts.

Everything goes through a SQLite cache, so a stopped run resumes and a rerun needs no network.
Stdlib only (urllib + sqlite3 + zlib): works locally and on Kaggle (internet on).

Requests follow the Wikimedia API etiquette: one request at a time, `maxlag`, a generic
User-Agent (no personal details), retries with back-off on 429 / 5xx / maxlag.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Callable, Iterable

from ..data.normalize import normalize_answer

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "hallu-hierarchy-research/0.1 (academic research script; python-urllib)"
BATCH = 50  # titles per query (API limit for normal clients)

_REF = re.compile(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", re.S | re.I)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_PAREN = re.compile(r"\s*\([^)]*\)\s*$")


def clean_wikitext(text: str) -> str:
    """Wikitext -> normalized plain words. References and comments are dropped (their source titles
    would create spurious matches); link targets and infobox values are kept ([[A|B]] -> 'a b')."""
    return normalize_answer(_COMMENT.sub(" ", _REF.sub(" ", text or "")))


def base_title(title: str) -> str:
    """'Carrie (1976 film)' -> 'Carrie'."""
    return _PAREN.sub("", title).strip() or title


class WikiCache:
    """Requested title -> page record; search query -> total hits. Text stored zlib-compressed."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS pages (title TEXT PRIMARY KEY, rec BLOB)")
        self.db.execute("CREATE TABLE IF NOT EXISTS search (query TEXT PRIMARY KEY, hits INTEGER)")
        self.db.commit()

    def get_page(self, title: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT rec FROM pages WHERE title = ?", (title,)).fetchone()
        return json.loads(zlib.decompress(row[0])) if row else None

    def put_pages(self, recs: dict[str, dict[str, Any]]) -> None:
        self.db.executemany("INSERT OR REPLACE INTO pages VALUES (?, ?)",
                            [(t, zlib.compress(json.dumps(r).encode())) for t, r in recs.items()])
        self.db.commit()

    def get_hits(self, query: str) -> int | None:
        row = self.db.execute("SELECT hits FROM search WHERE query = ?", (query,)).fetchone()
        return None if row is None else int(row[0])

    def put_hits(self, query: str, hits: int) -> None:
        self.db.execute("INSERT OR REPLACE INTO search VALUES (?, ?)", (query, hits))
        self.db.commit()

    def counts(self) -> dict[str, int]:
        return {t: self.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("pages", "search")}

    def close(self) -> None:
        self.db.close()


def http_get_json(params: dict[str, Any], timeout: float = 60.0, retries: int = 6) -> dict[str, Any]:
    """One API call with back-off. Raises after `retries` failed attempts."""
    params = {**params, "format": "json", "formatversion": 2, "maxlag": 5}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    delay = 2.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.load(r)
            if data.get("error", {}).get("code") == "maxlag":
                raise urllib.error.URLError("maxlag")
            if "error" in data:
                raise RuntimeError(f"Wikipedia API error: {data['error']}")
            return data
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            wait = float(e.headers.get("Retry-After") or delay)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
            wait = delay
        time.sleep(wait)
        delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")


class WikiClient:
    """Cached page and search lookups. `get` is injectable for tests (no network)."""

    def __init__(self, cache: WikiCache, interval_s: float = 0.1,
                 get: Callable[[dict[str, Any]], dict[str, Any]] = http_get_json) -> None:
        self.cache, self.interval_s, self._get = cache, interval_s, get
        self.n_requests = 0
        self.seconds = 0.0

    def _call(self, params: dict[str, Any]) -> dict[str, Any]:
        t = time.perf_counter()
        data = self._get(params)
        self.n_requests += 1
        if self.interval_s:
            time.sleep(self.interval_s)
        self.seconds += time.perf_counter() - t
        return data

    # ------------------------------------------------------------------ pages

    def pages(self, titles: Iterable[str], progress: Callable[[int, int], None] | None = None) -> dict[str, dict[str, Any]]:
        """title -> {canonical, missing, disambig, text (normalized words)}; follows redirects."""
        titles = list(dict.fromkeys(t for t in titles if t))
        out = {t: rec for t in titles if (rec := self.cache.get_page(t)) is not None}
        todo = [t for t in titles if t not in out]
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            recs = self._fetch_chunk(chunk)
            self.cache.put_pages(recs)
            out.update(recs)
            if progress:
                progress(min(i + BATCH, len(todo)), len(todo))
        return out

    def _fetch_chunk(self, titles: list[str]) -> dict[str, dict[str, Any]]:
        data = self._call({"action": "query", "prop": "revisions|pageprops", "rvprop": "content",
                           "rvslots": "main", "ppprop": "disambiguation", "redirects": 1,
                           "titles": "|".join(titles)})
        q = data.get("query", {})
        step = {d["from"]: d["to"] for key in ("normalized", "redirects") for d in q.get(key, [])}
        by_title = {p["title"]: p for p in q.get("pages", [])}
        recs = {}
        for t in titles:
            cur, seen = t, set()
            while cur in step and cur not in seen:  # normalized -> redirect chain
                seen.add(cur)
                cur = step[cur]
            p = by_title.get(cur)
            if p is None or p.get("missing") or p.get("invalid"):
                recs[t] = {"canonical": cur, "missing": True, "disambig": False, "text": ""}
                continue
            content = (p.get("revisions") or [{}])[0].get("slots", {}).get("main", {}).get("content", "")
            recs[t] = {"canonical": p["title"], "missing": False,
                       "disambig": "disambiguation" in (p.get("pageprops") or {}),
                       "redirected": cur != t, "text": clean_wikitext(content)}
        return recs

    # ------------------------------------------------------------------ search

    def linked_hits(self, phrase: str, linked_title: str) -> int:
        """Number of pages that contain `phrase` and link to `linked_title` (CirrusSearch `linksto:`).
        The subject page itself does not link to itself, so it is not counted here."""
        phrase = phrase.replace('"', " ").strip()
        query = f'"{phrase}" linksto:"{linked_title}"'
        hits = self.cache.get_hits(query)
        if hits is None:
            data = self._call({"action": "query", "list": "search", "srsearch": query, "srlimit": 1,
                               "srinfo": "totalhits", "srprop": ""})
            hits = int(data.get("query", {}).get("searchinfo", {}).get("totalhits", 0))
            self.cache.put_hits(query, hits)
        return hits
