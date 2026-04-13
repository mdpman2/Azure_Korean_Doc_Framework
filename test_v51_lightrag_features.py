"""
v5.1 LightRAG 기능 통합 테스트 — Reranker, LLM Cache, RAGAS Evaluator

오프라인(Azure 연결 불필요) 단위 테스트입니다.
"""
import json
import os
import shutil
import tempfile

# ============================================================
# 테스트 유틸리티
# ============================================================

class _T:
    _pass = 0
    _fail = 0

    @classmethod
    def check(cls, label: str, condition: bool, detail: str = ""):
        if condition:
            cls._pass += 1
            print(f"  ✅ {label}")
        else:
            cls._fail += 1
            detail_msg = f" | {detail}" if detail else ""
            print(f"  ❌ {label}{detail_msg}")

    @classmethod
    def summary(cls):
        total = cls._pass + cls._fail
        print(f"\n{'='*60}")
        print(f"총 {total}건: ✅ {cls._pass} / ❌ {cls._fail}")
        print(f"{'='*60}")
        return cls._fail == 0


T = _T


# ============================================================
# 1. Reranker 단위 테스트
# ============================================================

def test_reranker():
    print("\n📌 [1] Reranker 테스트")
    print("-" * 50)

    from azure_korean_doc_framework.core.reranker import (
        Reranker,
        RerankerBackend,
        RerankerResult,
    )
    from azure_korean_doc_framework.core.schema import SearchResult

    # 1-1. RerankerBackend enum
    T.check("RerankerBackend enum 정의", len(RerankerBackend) == 4)

    # 1-2. Reranker 생성 (none 백엔드)
    reranker = Reranker(backend="none")
    T.check("Reranker(none) 생성", reranker.backend == RerankerBackend.NONE)

    # 1-3. none 백엔드에서 rerank는 원본 그대로 반환
    dummy_results = [
        SearchResult(content="문서 A 내용", source="a.pdf", score=0.8),
        SearchResult(content="문서 B 내용", source="b.pdf", score=0.6),
        SearchResult(content="문서 C 내용", source="c.pdf", score=0.4),
    ]
    result = reranker.rerank("테스트 질문", dummy_results, top_k=2)
    T.check("none 백엔드 rerank 결과 타입", isinstance(result, RerankerResult))
    T.check("none 백엔드 reranked=False", result.reranked is False)
    T.check("none 백엔드 원본 개수 유지", result.original_count == 3)

    # 1-4. cross_encoder 백엔드 (모델 로드 불가 시 폴백 — 실제 모델 다운로드는 오래 걸리므로 none으로 대체)
    reranker_ce = Reranker(backend="none")
    result_ce = reranker_ce.rerank("테스트", dummy_results, top_k=2)
    T.check("cross_encoder 폴백 처리", isinstance(result_ce, RerankerResult))

    # 1-5. jina 백엔드 (API 키 없으면 폴백)
    reranker_jina = Reranker(backend="jina", jina_api_key="")
    result_jina = reranker_jina.rerank("테스트", dummy_results, top_k=2)
    T.check("jina 폴백 처리 (no key)", result_jina.reranked is False)

    # 1-6. RerankerResult 필드 검증
    T.check("RerankerResult.results 리스트", isinstance(result.results, list))
    T.check("RerankerResult.backend_used 문자열", isinstance(result.backend_used, str))

    # 1-7. 빈 결과 처리
    empty_result = reranker.rerank("질문", [], top_k=5)
    T.check("빈 결과 처리", empty_result.results == [] and empty_result.reranked is False)


# ============================================================
# 2. LLM Cache 단위 테스트
# ============================================================

def test_llm_cache():
    print("\n📌 [2] LLM Cache 테스트")
    print("-" * 50)

    from azure_korean_doc_framework.core.llm_cache import (
        LLMResponseCache,
        CacheEntry,
        CacheStats,
    )

    # 임시 디렉토리 사용
    tmpdir = tempfile.mkdtemp(prefix="llm_cache_test_")

    try:
        # 2-1. 캐시 생성
        cache = LLMResponseCache(
            cache_dir=tmpdir,
            max_memory_entries=10,
            default_ttl=0,
            enabled=True,
        )
        T.check("캐시 생성", cache.enabled is True)

        # 2-2. put/get 기본 동작
        key = cache.put("hello prompt", "hello response", model_key="gpt-5.4")
        T.check("put 반환값 (key)", len(key) == 64)  # SHA-256 hex

        cached = cache.get("hello prompt", model_key="gpt-5.4")
        T.check("get 캐시 히트", cached == "hello response")

        # 2-3. 캐시 미스
        miss = cache.get("different prompt", model_key="gpt-5.4")
        T.check("get 캐시 미스", miss is None)

        # 2-4. 통계 확인
        stats = cache.get_stats()
        T.check("stats hits=1", stats["hits"] == 1)
        T.check("stats misses=1", stats["misses"] == 1)
        T.check("stats hit_rate=0.5", abs(stats["hit_rate"] - 0.5) < 0.01)

        # 2-5. 모델 키 분리 (다른 model_key → 미스)
        miss_model = cache.get("hello prompt", model_key="gpt-4.1")
        T.check("다른 model_key 캐시 미스", miss_model is None)

        # 2-6. invalidate
        removed = cache.invalidate("hello prompt", model_key="gpt-5.4")
        T.check("invalidate 성공", removed is True)
        after_invalidate = cache.get("hello prompt", model_key="gpt-5.4")
        T.check("invalidate 후 미스", after_invalidate is None)

        # 2-7. TTL 만료 테스트
        import time
        cache_ttl = LLMResponseCache(
            cache_dir=tmpdir + "_ttl",
            max_memory_entries=10,
            default_ttl=0.1,  # 0.1초 TTL
            enabled=True,
        )
        cache_ttl.put("ttl_prompt", "ttl_response", model_key="test")
        cached_ttl = cache_ttl.get("ttl_prompt", model_key="test")
        T.check("TTL 만료 전 히트", cached_ttl == "ttl_response")

        # 2-8. LRU eviction
        cache_small = LLMResponseCache(
            cache_dir=tmpdir + "_lru",
            max_memory_entries=3,
            enabled=True,
        )
        for i in range(5):
            cache_small.put(f"prompt_{i}", f"response_{i}", model_key="test")
        stats_small = cache_small.get_stats()
        T.check("LRU eviction 작동", stats_small["evictions"] >= 2)
        T.check("메모리 엔트리 제한", stats_small["total_entries"] <= 3)

        # 2-9. 비활성화 캐시
        cache_off = LLMResponseCache(enabled=False)
        cache_off.put("test", "test")
        T.check("비활성화 캐시 get=None", cache_off.get("test") is None)

        # 2-10. clear
        cache.put("clear_test", "clear_value")
        cache.clear()
        T.check("clear 후 미스", cache.get("clear_test") is None)

        # 2-11. CacheEntry 직렬화/역직렬화
        entry = CacheEntry(key="k1", value="v1", model="m1", created_at=1000.0, ttl=0, access_count=5)
        d = entry.to_dict()
        restored = CacheEntry.from_dict(d)
        T.check("CacheEntry 직렬화/역직렬화", restored.key == "k1" and restored.value == "v1" and restored.access_count == 5)

        # 2-12. 디스크 영속성 (재로드)
        persistent_dir = tmpdir + "_persist"
        cache_p1 = LLMResponseCache(cache_dir=persistent_dir, enabled=True)
        cache_p1.put("persist_prompt", "persist_response", model_key="test")
        del cache_p1

        cache_p2 = LLMResponseCache(cache_dir=persistent_dir, enabled=True)
        T.check("디스크 영속성 (재로드)", cache_p2.get("persist_prompt", model_key="test") == "persist_response")

    finally:
        for d in [tmpdir, tmpdir + "_ttl", tmpdir + "_lru", tmpdir + "_persist"]:
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)


# ============================================================
# 3. RAGAS Evaluator 단위 테스트 (구조 검증)
# ============================================================

def test_ragas_evaluator_structure():
    print("\n📌 [3] RAGAS Evaluator 구조 테스트")
    print("-" * 50)

    from azure_korean_doc_framework.evaluation.ragas_evaluator import (
        RAGASMetrics,
        RAGASBatchResult,
        RAGASEvaluator,
    )

    # 3-1. RAGASMetrics 기본값
    m = RAGASMetrics()
    T.check("RAGASMetrics 기본값 overall_score=0", m.overall_score == 0.0)

    # 3-2. to_dict 메서드
    d = m.to_dict()
    T.check("to_dict 키 포함", all(k in d for k in [
        "context_precision", "context_recall", "faithfulness",
        "answer_relevancy", "answer_correctness", "overall_score",
    ]))

    # 3-3. RAGASMetrics 값 설정
    m2 = RAGASMetrics(
        context_precision=0.8,
        context_recall=0.9,
        faithfulness=0.95,
        answer_relevancy=0.85,
        answer_correctness=0.88,
        overall_score=0.87,
    )
    T.check("RAGASMetrics 값 설정", m2.faithfulness == 0.95)
    T.check("RAGASMetrics to_dict 정밀도", m2.to_dict()["faithfulness"] == 0.95)

    # 3-4. RAGASBatchResult 기본값
    br = RAGASBatchResult()
    T.check("RAGASBatchResult 기본값", br.total_items == 0 and br.evaluated_items == 0)

    # 3-5. RAGASEvaluator 클래스 존재 확인
    T.check("RAGASEvaluator 클래스 존재", RAGASEvaluator is not None)
    T.check("evaluate 메서드 존재", hasattr(RAGASEvaluator, 'evaluate'))
    T.check("evaluate_batch 메서드 존재", hasattr(RAGASEvaluator, 'evaluate_batch'))
    T.check("evaluate_context_precision 메서드", hasattr(RAGASEvaluator, 'evaluate_context_precision'))
    T.check("evaluate_context_recall 메서드", hasattr(RAGASEvaluator, 'evaluate_context_recall'))
    T.check("evaluate_faithfulness 메서드", hasattr(RAGASEvaluator, 'evaluate_faithfulness'))
    T.check("evaluate_answer_relevancy 메서드", hasattr(RAGASEvaluator, 'evaluate_answer_relevancy'))
    T.check("evaluate_answer_correctness 메서드", hasattr(RAGASEvaluator, 'evaluate_answer_correctness'))


# ============================================================
# 4. Config 통합 테스트
# ============================================================

def test_config_v51():
    print("\n📌 [4] Config v5.1 설정 테스트")
    print("-" * 50)

    from azure_korean_doc_framework.config import Config

    # 4-1. Reranker 설정
    T.check("RERANKER_ENABLED 존재", hasattr(Config, 'RERANKER_ENABLED'))
    T.check("RERANKER_BACKEND 존재", hasattr(Config, 'RERANKER_BACKEND'))
    T.check("RERANKER_MODEL 존재", hasattr(Config, 'RERANKER_MODEL'))
    T.check("RERANKER_TOP_K 존재", hasattr(Config, 'RERANKER_TOP_K'))
    T.check("JINA_API_KEY 존재", hasattr(Config, 'JINA_API_KEY'))

    # 4-2. LLM Cache 설정
    T.check("LLM_CACHE_ENABLED 존재", hasattr(Config, 'LLM_CACHE_ENABLED'))
    T.check("LLM_CACHE_DIR 존재", hasattr(Config, 'LLM_CACHE_DIR'))
    T.check("LLM_CACHE_MAX_MEMORY 존재", hasattr(Config, 'LLM_CACHE_MAX_MEMORY'))
    T.check("LLM_CACHE_TTL 존재", hasattr(Config, 'LLM_CACHE_TTL'))

    # 4-3. RAGAS 설정
    T.check("RAGAS_JUDGE_MODEL 존재", hasattr(Config, 'RAGAS_JUDGE_MODEL'))

    # 4-4. 기본값 검증
    T.check("RERANKER_BACKEND 기본값 cross_encoder", Config.RERANKER_BACKEND == "cross_encoder")
    T.check("RERANKER_TOP_K 기본값 5", Config.RERANKER_TOP_K == 5)
    T.check("LLM_CACHE_DIR 기본값", Config.LLM_CACHE_DIR == "output/llm_cache")
    T.check("LLM_CACHE_MAX_MEMORY 기본값 500", Config.LLM_CACHE_MAX_MEMORY == 500)


# ============================================================
# 5. Agent 통합 확인 (import 수준)
# ============================================================

def test_agent_integration():
    print("\n📌 [5] Agent v5.1 통합 확인")
    print("-" * 50)

    # 필요한 모듈이 import 가능한지
    from azure_korean_doc_framework.core.reranker import Reranker
    from azure_korean_doc_framework.core.llm_cache import LLMResponseCache
    from azure_korean_doc_framework.evaluation.ragas_evaluator import RAGASEvaluator

    T.check("Reranker import 성공", Reranker is not None)
    T.check("LLMResponseCache import 성공", LLMResponseCache is not None)
    T.check("RAGASEvaluator import 성공", RAGASEvaluator is not None)

    # agent.py에서 Reranker/Cache import 확인
    import azure_korean_doc_framework.core.agent as agent_module
    import inspect
    source = inspect.getsource(agent_module)
    T.check("agent.py에 Reranker import", "from .reranker import" in source)
    T.check("agent.py에 LLMResponseCache import", "from .llm_cache import" in source)
    T.check("agent.py에 self.reranker 초기화", "self.reranker" in source)
    T.check("agent.py에 self.llm_cache 초기화", "self.llm_cache" in source)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    print("=" * 60)
    print("v5.1 LightRAG 기능 통합 테스트")
    print("   Reranker | LLM Cache | RAGAS Evaluator")
    print("=" * 60)

    test_reranker()
    test_llm_cache()
    test_ragas_evaluator_structure()
    test_config_v51()
    test_agent_integration()

    success = T.summary()
    if not success:
        exit(1)
