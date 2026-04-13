"""
웹 검색 및 페이지 수집 도구.

OpenClaude의 WebSearch / WebFetch 개념을 참조하여 구현.
Q&A 시 실시간 외부 정보를 보강하거나, 문서에 없는 최신 정보를 제공합니다.

[v5.0 신규]
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import List, Optional


@dataclass
class WebSearchResult:
    """웹 검색 결과 항목."""
    title: str
    url: str
    snippet: str
    score: float = 0.0


@dataclass
class WebFetchResult:
    """웹 페이지 수집 결과."""
    url: str
    title: str
    content: str  # 마크다운 변환 텍스트
    status_code: int = 200
    error: Optional[str] = None


class _TextExtractor(HTMLParser):
    """HTML에서 텍스트를 추출하는 경량 파서."""

    _SKIP_TAGS = frozenset(["script", "style", "noscript", "svg", "nav", "footer", "header"])

    def __init__(self):
        super().__init__()
        self._texts: List[str] = []
        self._skip_depth = 0
        self._title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in ("p", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self._texts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title and not self._title:
            self._title = data.strip()
        if self._skip_depth == 0:
            self._texts.append(data)

    @property
    def text(self) -> str:
        raw = "".join(self._texts)
        # 연속 빈줄을 2개로 축소
        return re.sub(r"\n{3,}", "\n\n", raw).strip()

    @property
    def title(self) -> str:
        return self._title


class WebSearchTool:
    """
    Bing Search API 또는 DuckDuckGo HTML 스크래핑 기반 웹 검색 도구.

    사용 예시:
        tool = WebSearchTool()
        results = tool.search("Azure AI Search 최신 기능")
        for r in results:
            print(f"{r.title}: {r.url}")
    """

    def __init__(
        self,
        bing_api_key: Optional[str] = None,
        max_results: int = 5,
        timeout: int = 10,
    ):
        self.bing_api_key = bing_api_key
        self.max_results = max_results
        self.timeout = timeout

    def search(self, query: str, max_results: Optional[int] = None) -> List[WebSearchResult]:
        """웹 검색을 수행합니다."""
        limit = max_results or self.max_results
        if self.bing_api_key:
            return self._bing_search(query, limit)
        return self._duckduckgo_search(query, limit)

    def _bing_search(self, query: str, max_results: int) -> List[WebSearchResult]:
        """Bing Search API를 사용합니다."""
        encoded = urllib.parse.quote_plus(query)
        url = f"https://api.bing.microsoft.com/v7.0/search?q={encoded}&count={max_results}&mkt=ko-KR"
        req = urllib.request.Request(url, headers={"Ocp-Apim-Subscription-Key": self.bing_api_key})

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"   ⚠️ Bing 검색 실패: {e}")
            return []

        results = []
        for idx, item in enumerate(data.get("webPages", {}).get("value", [])[:max_results]):
            results.append(WebSearchResult(
                title=item.get("name", ""),
                url=item.get("url", ""),
                snippet=item.get("snippet", ""),
                score=1.0 - (idx * 0.1),
            ))
        return results

    def _duckduckgo_search(self, query: str, max_results: int) -> List[WebSearchResult]:
        """DuckDuckGo HTML을 스크래핑합니다 (폴백)."""
        encoded = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; AzureKoreanDocFramework/5.0)"
        })

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"   ⚠️ DuckDuckGo 검색 실패: {e}")
            return []

        results = []
        # <a class="result__a" href="...">title</a>
        # <a class="result__snippet">snippet</a>
        link_pattern = re.compile(
            r'class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>',
            re.IGNORECASE,
        )
        snippet_pattern = re.compile(
            r'class="result__snippet"[^>]*>([^<]*(?:<[^>]*>[^<]*)*)</a>',
            re.IGNORECASE,
        )

        links = link_pattern.findall(html)
        snippets = snippet_pattern.findall(html)

        for idx, (href, title) in enumerate(links[:max_results]):
            # DuckDuckGo 리다이렉트 URL에서 실제 URL 추출
            actual_url = href
            if "uddg=" in href:
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                actual_url = parsed.get("uddg", [href])[0]

            snippet = re.sub(r"<[^>]+>", "", snippets[idx]) if idx < len(snippets) else ""
            results.append(WebSearchResult(
                title=title.strip(),
                url=actual_url,
                snippet=snippet.strip(),
                score=1.0 - (idx * 0.1),
            ))

        return results


class WebFetchTool:
    """
    웹 페이지를 가져와서 텍스트로 변환하는 도구.

    사용 예시:
        tool = WebFetchTool()
        result = tool.fetch("https://learn.microsoft.com/azure/...")
        print(result.content[:500])
    """

    def __init__(self, timeout: int = 15, max_chars: int = 8000):
        self.timeout = timeout
        self.max_chars = max_chars

    def fetch(self, url: str) -> WebFetchResult:
        """URL의 내용을 가져와서 텍스트로 변환합니다."""
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; AzureKoreanDocFramework/5.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
        })

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status = resp.status
                raw = resp.read()

                # 인코딩 감지
                charset = "utf-8"
                content_type = resp.headers.get("Content-Type", "")
                if "charset=" in content_type:
                    charset = content_type.split("charset=")[-1].split(";")[0].strip()

                html = raw.decode(charset, errors="replace")

        except Exception as e:
            return WebFetchResult(url=url, title="", content="", status_code=0, error=str(e))

        parser = _TextExtractor()
        try:
            parser.feed(html)
        except Exception:
            pass

        content = parser.text
        if len(content) > self.max_chars:
            content = content[:self.max_chars] + "\n\n... (잘림)"

        return WebFetchResult(
            url=url,
            title=parser.title,
            content=content,
            status_code=status,
        )
