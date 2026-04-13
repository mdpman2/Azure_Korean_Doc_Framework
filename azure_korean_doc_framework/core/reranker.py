"""
Cross-Encoder Reranker 모듈 (v5.1 — LightRAG Reranker 참조)

LightRAG이 가장 강력히 권장하는 기능으로, 검색 결과를 Cross-Encoder 모델로
재순위하여 정밀도를 10-20% 향상시킵니다.

지원 백엔드:
1. Azure AI Search Semantic Ranker (기존, 무료 tier 제한)
2. sentence-transformers Cross-Encoder (로컬, BAAI/bge-reranker-v2-m3)
3. Jina Reranker API (클라우드, jina-reranker-v2-base-multilingual)
4. LLM 기반 Reranker (Azure OpenAI, 폴백)

참조: https://github.com/HKUDS/LightRAG — Reranker 통합
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from .schema import SearchResult


class RerankerBackend(Enum):
    """Reranker 백엔드 유형"""
    NONE = "none"                   # Reranker 비활성화
    CROSS_ENCODER = "cross_encoder"  # sentence-transformers Cross-Encoder (로컬)
    JINA = "jina"                    # Jina Reranker API
    LLM = "llm"                      # LLM 기반 재순위 (Azure OpenAI)


@dataclass
class RerankerResult:
    """Reranker 실행 결과"""
    results: List[SearchResult]
    backend_used: str = ""
    reranked: bool = False
    original_count: int = 0
    final_count: int = 0


class Reranker:
    """
    Cross-Encoder Reranker (LightRAG 방식)

    검색 결과를 query-document 쌍으로 Cross-Encoder에 입력하여
    관련성 점수를 재계산, 상위 K개를 반환합니다.

    LightRAG은 Reranker 활성화 시 Mix 모드를 기본 쿼리 모드로 권장합니다.
    """

    def __init__(
        self,
        backend: str = "cross_encoder",
        model_name: str = "",
        jina_api_key: str = "",
        top_k: int = 5,
        warm_up: bool = False,
    ):
        self.backend = RerankerBackend(backend) if backend else RerankerBackend.NONE
        self.model_name = model_name
        self.jina_api_key = jina_api_key
        self.top_k = top_k
        self._cross_encoder = None

        # [최적화] 초기화 시점에 Cross-Encoder 모델을 미리 로드
        if warm_up and self.backend == RerankerBackend.CROSS_ENCODER:
            self._get_cross_encoder()

    def _get_cross_encoder(self):
        """Cross-Encoder 모델을 lazy 로드합니다."""
        if self._cross_encoder is not None:
            return self._cross_encoder

        try:
            from sentence_transformers import CrossEncoder
        except (ImportError, Exception):
            print("   ⚠️ sentence-transformers 로드 실패. pip install sentence-transformers")
            self.backend = RerankerBackend.NONE
            return None

        model = self.model_name or "BAAI/bge-reranker-v2-m3"
        try:
            print(f"   📦 Cross-Encoder 로딩: {model}")
            self._cross_encoder = CrossEncoder(model)
        except Exception as e:
            print(f"   ⚠️ Cross-Encoder 모델 로드 실패: {e}")
            self.backend = RerankerBackend.NONE
            return None
        return self._cross_encoder

    def rerank(
        self,
        query: str,
        search_results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> RerankerResult:
        """
        검색 결과를 Reranker로 재순위합니다.

        Args:
            query: 사용자 질문
            search_results: 기존 검색 결과
            top_k: 반환할 상위 결과 수 (None이면 self.top_k 사용)

        Returns:
            RerankerResult: 재순위된 결과
        """
        if not search_results or self.backend == RerankerBackend.NONE:
            return RerankerResult(
                results=search_results,
                backend_used="none",
                reranked=False,
                original_count=len(search_results),
                final_count=len(search_results),
            )

        k = top_k or self.top_k

        if self.backend == RerankerBackend.CROSS_ENCODER:
            return self._rerank_cross_encoder(query, search_results, k)
        elif self.backend == RerankerBackend.JINA:
            return self._rerank_jina(query, search_results, k)
        elif self.backend == RerankerBackend.LLM:
            return self._rerank_llm(query, search_results, k)
        else:
            return RerankerResult(
                results=search_results[:k],
                backend_used="none",
                reranked=False,
                original_count=len(search_results),
                final_count=min(k, len(search_results)),
            )

    def _rerank_cross_encoder(
        self,
        query: str,
        search_results: List[SearchResult],
        top_k: int,
    ) -> RerankerResult:
        """sentence-transformers Cross-Encoder로 재순위"""
        encoder = self._get_cross_encoder()
        if encoder is None:
            return RerankerResult(
                results=search_results[:top_k],
                backend_used="cross_encoder_fallback",
                reranked=False,
                original_count=len(search_results),
                final_count=min(top_k, len(search_results)),
            )

        pairs = [(query, r.content) for r in search_results]
        scores = encoder.predict(pairs)

        scored = list(zip(search_results, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)

        reranked = []
        for result, score in scored[:top_k]:
            reranked.append(SearchResult(
                content=result.content,
                source=result.source,
                score=float(score),
                metadata={**result.metadata, "original_score": result.score, "reranker_score": float(score)},
            ))

        return RerankerResult(
            results=reranked,
            backend_used="cross_encoder",
            reranked=True,
            original_count=len(search_results),
            final_count=len(reranked),
        )

    def _rerank_jina(
        self,
        query: str,
        search_results: List[SearchResult],
        top_k: int,
    ) -> RerankerResult:
        """Jina Reranker API로 재순위"""
        import json
        from urllib.request import Request, urlopen
        from urllib.error import URLError

        api_key = self.jina_api_key
        if not api_key:
            print("   ⚠️ JINA_API_KEY 미설정, Reranker 건너뜀")
            return RerankerResult(
                results=search_results[:top_k],
                backend_used="jina_fallback",
                reranked=False,
                original_count=len(search_results),
                final_count=min(top_k, len(search_results)),
            )

        model = self.model_name or "jina-reranker-v2-base-multilingual"
        payload = {
            "model": model,
            "query": query,
            "top_n": top_k,
            "documents": [r.content for r in search_results],
        }

        req = Request(
            "https://api.jina.ai/v1/rerank",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (URLError, TimeoutError, OSError) as e:
            print(f"   ⚠️ Jina Reranker API 실패: {e}")
            return RerankerResult(
                results=search_results[:top_k],
                backend_used="jina_error",
                reranked=False,
                original_count=len(search_results),
                final_count=min(top_k, len(search_results)),
            )

        reranked = []
        for item in body.get("results", []):
            idx = item.get("index", 0)
            score = item.get("relevance_score", 0.0)
            if idx < len(search_results):
                original = search_results[idx]
                reranked.append(SearchResult(
                    content=original.content,
                    source=original.source,
                    score=float(score),
                    metadata={**original.metadata, "original_score": original.score, "reranker_score": float(score)},
                ))

        return RerankerResult(
            results=reranked[:top_k],
            backend_used="jina",
            reranked=True,
            original_count=len(search_results),
            final_count=len(reranked[:top_k]),
        )

    def _rerank_llm(
        self,
        query: str,
        search_results: List[SearchResult],
        top_k: int,
    ) -> RerankerResult:
        """LLM 기반 재순위 (Azure OpenAI) — 폴백용"""
        from ..utils.azure_clients import AzureClientFactory
        from ..config import Config

        client = AzureClientFactory.get_openai_client(is_advanced=False)
        deployment = Config.MODELS.get("gpt-4.1", "gpt-4.1")

        numbered = []
        for i, r in enumerate(search_results):
            snippet = r.content[:300]
            numbered.append(f"[{i}] {snippet}")
        docs_text = "\n".join(numbered)

        prompt = (
            f"질문: {query}\n\n"
            f"다음 문서들의 관련성 순위를 매겨주세요. "
            f"가장 관련성 높은 문서부터 인덱스 번호만 쉼표로 나열하세요.\n\n"
            f"{docs_text}\n\n"
            f"상위 {top_k}개 인덱스 (쉼표 구분): "
        )

        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_completion_tokens=100,
            )
            text = response.choices[0].message.content.strip()
            import re
            indices = [int(x) for x in re.findall(r'\d+', text)]
            seen = set()
            reranked = []
            for idx in indices:
                if idx in seen or idx >= len(search_results):
                    continue
                seen.add(idx)
                reranked.append(search_results[idx])
                if len(reranked) >= top_k:
                    break

            # 순위를 점수로 변환
            for rank, r in enumerate(reranked):
                r.metadata["reranker_score"] = 1.0 - (rank * 0.1)

            return RerankerResult(
                results=reranked,
                backend_used="llm",
                reranked=True,
                original_count=len(search_results),
                final_count=len(reranked),
            )
        except Exception as e:
            print(f"   ⚠️ LLM Reranker 실패: {e}")
            return RerankerResult(
                results=search_results[:top_k],
                backend_used="llm_error",
                reranked=False,
                original_count=len(search_results),
                final_count=min(top_k, len(search_results)),
            )
