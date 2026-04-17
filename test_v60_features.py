#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
azure_korean_doc_framework v6.0 신규 기능 통합 테스트

오프라인(Azure 연결 불필요) 단위 테스트입니다.

시나리오 목록:
  T1. Config v6.0 설정 검증 (임베딩, Agentic, Async, Semantic Cache, 등)
  T2. Semantic Cache (임베딩 유사도 기반 퍼지 매칭)
  T3. Responses API 분기 (MultiModelManager)
  T4. Content Understanding 파서 초기화
  T5. Neo4j 백엔드 (graph_rag — NetworkX 폴백)
  T6. Agentic Retrieval 데이터 모델
  T7. Prompt Caching 설정 검증
  T8. 크로스 모듈 통합 (Semantic Cache + LLM Cache 병행)
"""

import math
import os
import shutil
import sys
import tempfile
import time

import networkx as nx

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


# ==================== 테스트 러너 ====================

class _T:
    _results = []  # (scenario, name, status, detail)
    _scenario = ""

    @classmethod
    def scenario(cls, name: str):
        cls._scenario = name
        print(f"\n{'='*70}")
        print(f"🧪 {name}")
        print(f"{'='*70}")

    @classmethod
    def check(cls, label: str, condition: bool, detail: str = ""):
        status = "pass" if condition else "fail"
        icon = "✅" if condition else "❌"
        cls._results.append((cls._scenario, label, status, detail))
        print(f"  {icon} {label}" + (f" — {detail}" if detail else ""))

    @classmethod
    def skip(cls, label: str, reason: str = ""):
        cls._results.append((cls._scenario, label, "skip", reason))
        print(f"  ⏭️ SKIP: {label}" + (f" — {reason}" if reason else ""))

    @classmethod
    def summary(cls) -> int:
        total = len(cls._results)
        passed = sum(1 for _, _, s, _ in cls._results if s == "pass")
        skipped = sum(1 for _, _, s, _ in cls._results if s == "skip")
        failed = total - passed - skipped

        print(f"\n{'='*70}")
        print("📊 v6.0 기능 테스트 결과")
        print(f"{'='*70}")

        scenarios = {}
        for sc, _, s, _ in cls._results:
            if sc not in scenarios:
                scenarios[sc] = {"pass": 0, "fail": 0, "skip": 0}
            scenarios[sc][s] += 1

        for sc, counts in scenarios.items():
            t = counts["pass"] + counts["fail"] + counts["skip"]
            icon = "❌" if counts["fail"] > 0 else ("⏭️" if counts["skip"] > 0 else "✅")
            skip_info = f" ({counts['skip']} skip)" if counts["skip"] else ""
            print(f"  {icon} {sc}: {counts['pass']}/{t}{skip_info}")

        print(f"\n🏁 총 결과: {passed} 통과 / {skipped} 스킵 / {failed} 실패 (총 {total}개)")

        if failed == 0:
            print("\n✨ 모든 v6.0 기능 테스트 통과!")
        else:
            print("\n⚠️ 실패 항목:")
            for sc, name, s, detail in cls._results:
                if s == "fail":
                    print(f"   ❌ [{sc}] {name}" + (f": {detail}" if detail else ""))

        return 0 if failed == 0 else 1


T = _T


# ==================== T1. Config v6.0 설정 검증 ====================

def test_t1_config_v60():
    T.scenario("T1. Config v6.0 설정 검증")

    from azure_korean_doc_framework.config import Config

    # 임베딩 설정
    T.check("EMBEDDING_DEPLOYMENT 설정됨", bool(Config.EMBEDDING_DEPLOYMENT), Config.EMBEDDING_DEPLOYMENT)
    T.check("EMBEDDING_DIMENSIONS is int", isinstance(Config.EMBEDDING_DIMENSIONS, int))
    T.check("EMBEDDING_DIMENSIONS > 0", Config.EMBEDDING_DIMENSIONS > 0, str(Config.EMBEDDING_DIMENSIONS))

    # Agentic Retrieval 설정
    T.check("AGENTIC_RETRIEVAL_ENABLED is bool", isinstance(Config.AGENTIC_RETRIEVAL_ENABLED, bool))
    T.check("AGENTIC_KB_NAME is str", isinstance(Config.AGENTIC_KB_NAME, str))
    T.check("AGENTIC_OUTPUT_MODE valid",
            Config.AGENTIC_OUTPUT_MODE in ("extractive_data", "answer_synthesis"),
            Config.AGENTIC_OUTPUT_MODE)
    T.check("AGENTIC_REASONING_EFFORT valid",
            Config.AGENTIC_REASONING_EFFORT in ("minimal", "low", "medium"),
            Config.AGENTIC_REASONING_EFFORT)

    # Async Pipeline 설정
    T.check("ASYNC_PIPELINE_ENABLED is bool", isinstance(Config.ASYNC_PIPELINE_ENABLED, bool))
    T.check("ASYNC_MAX_CONCURRENT > 0", Config.ASYNC_MAX_CONCURRENT > 0, str(Config.ASYNC_MAX_CONCURRENT))

    # Semantic Cache 설정
    T.check("SEMANTIC_CACHE_ENABLED is bool", isinstance(Config.SEMANTIC_CACHE_ENABLED, bool))
    T.check("SEMANTIC_CACHE_THRESHOLD is float", isinstance(Config.SEMANTIC_CACHE_THRESHOLD, float))
    T.check("SEMANTIC_CACHE_THRESHOLD 범위 0~1",
            0.0 <= Config.SEMANTIC_CACHE_THRESHOLD <= 1.0,
            str(Config.SEMANTIC_CACHE_THRESHOLD))

    # Prompt Caching 설정
    T.check("PROMPT_CACHING_ENABLED is bool", isinstance(Config.PROMPT_CACHING_ENABLED, bool))

    # Content Understanding 설정
    T.check("CONTENT_UNDERSTANDING_ENABLED is bool", isinstance(Config.CONTENT_UNDERSTANDING_ENABLED, bool))
    T.check("CONTENT_UNDERSTANDING_ENDPOINT is str", isinstance(Config.CONTENT_UNDERSTANDING_ENDPOINT, str))
    T.check("CONTENT_UNDERSTANDING_KEY is str", isinstance(Config.CONTENT_UNDERSTANDING_KEY, str))

    # Responses API 설정
    T.check("USE_RESPONSES_API is bool", isinstance(Config.USE_RESPONSES_API, bool))

    # Graph Storage Backend 설정
    T.check("GRAPH_STORAGE_BACKEND valid",
            Config.GRAPH_STORAGE_BACKEND in ("networkx", "neo4j"),
            Config.GRAPH_STORAGE_BACKEND)
    T.check("NEO4J_URI is str", isinstance(Config.NEO4J_URI, str))
    T.check("NEO4J_USER is str", isinstance(Config.NEO4J_USER, str))
    T.check("NEO4J_PASSWORD is str", isinstance(Config.NEO4J_PASSWORD, str))


# ==================== T2. Semantic Cache ====================

def test_t2_semantic_cache():
    T.scenario("T2. Semantic Cache")

    from azure_korean_doc_framework.core.llm_cache import (
        SemanticCache,
        SemanticCacheEntry,
        _cosine_similarity,
    )

    # 코사인 유사도 함수 검증
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [1.0, 0.0, 0.0]
    vec_c = [0.0, 1.0, 0.0]
    vec_d = [0.707, 0.707, 0.0]

    T.check("cosine_sim(같은 벡터) == 1.0",
            abs(_cosine_similarity(vec_a, vec_b) - 1.0) < 1e-6)
    T.check("cosine_sim(직교 벡터) == 0.0",
            abs(_cosine_similarity(vec_a, vec_c)) < 1e-6)
    T.check("cosine_sim(45도) ≈ 0.707",
            abs(_cosine_similarity(vec_a, vec_d) - 0.707) < 0.01)

    # 영벡터 처리
    T.check("cosine_sim(영벡터) == 0.0",
            _cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0)

    # SemanticCache 기본 동작
    cache = SemanticCache(threshold=0.90, max_entries=10, enabled=True)
    T.check("SemanticCache 생성", cache.enabled is True)

    # put/get — 동일 벡터
    emb1 = [0.1, 0.2, 0.3, 0.4, 0.5]
    cache.put(emb1, "응답1", model="gpt-5.4")
    hit = cache.get(emb1)
    T.check("동일 벡터 캐시 히트", hit == "응답1")

    # 유사한 벡터 (threshold 이상)
    emb1_similar = [0.101, 0.199, 0.301, 0.399, 0.501]
    hit_similar = cache.get(emb1_similar)
    sim = _cosine_similarity(emb1, emb1_similar)
    T.check(f"유사 벡터 캐시 히트 (sim={sim:.4f})", hit_similar == "응답1", f"sim={sim:.4f}")

    # 다른 벡터 (threshold 미만)
    emb_different = [-0.5, 0.8, -0.1, 0.3, -0.9]
    hit_different = cache.get(emb_different)
    T.check("다른 벡터 캐시 미스", hit_different is None)

    # 통계 확인
    stats = cache.get_stats()
    T.check("stats hits >= 2", stats["hits"] >= 2)
    T.check("stats misses >= 1", stats["misses"] >= 1)
    T.check("stats threshold == 0.9", stats["threshold"] == 0.90)
    T.check("stats entries >= 1", stats["entries"] >= 1)

    # 복수 항목 + 가장 유사한 것 반환 확인
    emb2 = [0.9, 0.1, 0.0, 0.0, 0.0]
    emb3 = [0.0, 0.0, 0.0, 0.0, 1.0]
    cache.put(emb2, "응답2", model="gpt-5.4")
    cache.put(emb3, "응답3", model="gpt-5.4")

    query_like_emb2 = [0.89, 0.11, 0.01, 0.0, 0.0]
    hit_best = cache.get(query_like_emb2)
    T.check("복수 항목 중 가장 유사한 것 반환", hit_best == "응답2")

    # max_entries 초과 시 오래된 항목 제거
    small_cache = SemanticCache(threshold=0.5, max_entries=3, enabled=True)
    for i in range(5):
        vec = [float(i)] * 5
        small_cache.put(vec, f"r{i}")
    T.check("max_entries 초과 시 제거", len(small_cache._entries) <= 3)

    # 비활성화 캐시
    cache_off = SemanticCache(enabled=False)
    cache_off.put([1, 2, 3], "test")
    T.check("비활성화 캐시 get=None", cache_off.get([1, 2, 3]) is None)

    # clear
    cache.clear()
    T.check("clear 후 미스", cache.get(emb1) is None)
    T.check("clear 후 stats 초기화", cache.get_stats()["entries"] == 0)

    # SemanticCacheEntry TTL
    entry = SemanticCacheEntry(
        prompt_embedding=[1.0], response="test", created_at=time.time(), ttl=0
    )
    T.check("TTL=0이면 만료 안 됨", entry.is_expired is False)

    entry_expired = SemanticCacheEntry(
        prompt_embedding=[1.0], response="test",
        created_at=time.time() - 100, ttl=10
    )
    T.check("TTL 만료", entry_expired.is_expired is True)


# ==================== T3. Responses API 분기 ====================

def test_t3_responses_api():
    T.scenario("T3. Responses API (MultiModelManager)")

    from azure_korean_doc_framework.core.multi_model_manager import MultiModelManager, _classify_model
    from azure_korean_doc_framework.config import Config

    # MultiModelManager 생성
    mgr = MultiModelManager()
    T.check("default_model 설정됨", bool(mgr.default_model))

    # _call_responses_api 메서드 존재 확인
    T.check("_call_responses_api 메서드 존재", hasattr(mgr, "_call_responses_api"))

    # _classify_model 검증
    is_adv, is_gpt5, is_reasoning = _classify_model("gpt-5.4")
    T.check("gpt-5.4는 고성능 모델", is_adv is True)
    T.check("gpt-5.4는 GPT-5 시리즈", is_gpt5 is True)
    T.check("gpt-5.4는 추론 모델", is_reasoning is True)

    is_adv_41, is_gpt5_41, _ = _classify_model("gpt-4.1")
    T.check("gpt-4.1은 일반 모델", is_adv_41 is False)
    T.check("gpt-4.1은 GPT-5 시리즈 아님", is_gpt5_41 is False)

    # USE_RESPONSES_API 설정 검증
    T.check("USE_RESPONSES_API is bool", isinstance(Config.USE_RESPONSES_API, bool))

    # Prompt Caching 설정
    T.check("PROMPT_CACHING_ENABLED is bool", isinstance(Config.PROMPT_CACHING_ENABLED, bool))


# ==================== T4. Content Understanding 파서 ====================

def test_t4_content_understanding():
    T.scenario("T4. Content Understanding 파서")

    from azure_korean_doc_framework.parsing.parser import HybridDocumentParser
    from azure_korean_doc_framework.config import Config

    # 스텁 생성 (DI 초기화 불필요)
    parser_stub = HybridDocumentParser.__new__(HybridDocumentParser)

    # Content Understanding 관련 메서드 존재 확인
    T.check("_analyze_with_content_understanding 메서드",
            hasattr(parser_stub, "_analyze_with_content_understanding"))
    T.check("_poll_content_understanding 메서드",
            hasattr(parser_stub, "_poll_content_understanding"))
    T.check("_parse_cu_result 메서드",
            hasattr(parser_stub, "_parse_cu_result"))

    # _parse_cu_result 단위 테스트
    mock_result = {
        "contents": [
            {"type": "text", "content": "첫 번째 텍스트", "pageNumber": 1},
            {"type": "table", "content": "| A | B |\n|---|---|\n| 1 | 2 |", "pageNumber": 1},
            {"type": "figure", "content": "이미지", "description": "통계 그래프", "pageNumber": 2},
            {"type": "sectionHeading", "content": "제2장 결론", "pageNumber": 3},
        ]
    }
    segments = parser_stub._parse_cu_result(mock_result)
    T.check("CU 결과 세그먼트 4개", len(segments) == 4, str(len(segments)))
    T.check("첫 번째 세그먼트 type=text", segments[0]["type"] == "text")
    T.check("두 번째 세그먼트 type=table", segments[1]["type"] == "table")
    T.check("세 번째 세그먼트 type=image", segments[2]["type"] == "image")
    T.check("네 번째 세그먼트 type=header", segments[3]["type"] == "header")

    # 이미지 description 포함 확인
    T.check("figure에 description 포함", "통계 그래프" in segments[2]["content"])

    # 빈 결과 처리
    empty_segments = parser_stub._parse_cu_result({"contents": []})
    T.check("빈 결과 처리", len(empty_segments) == 0)

    # analyzeResult 형식도 처리 (다른 응답 구조)
    alt_result = {
        "analyzeResult": {
            "contents": [
                {"type": "text", "text": "대체 형식 텍스트", "page": 1},
            ]
        }
    }
    alt_segments = parser_stub._parse_cu_result(alt_result)
    T.check("analyzeResult 대체 구조 처리", len(alt_segments) == 1)

    # Config 검증
    T.check("CU ENABLED 기본값 = false", Config.CONTENT_UNDERSTANDING_ENABLED is False)


# ==================== T5. Neo4j 백엔드 (NetworkX 폴백) ====================

def test_t5_neo4j_backend():
    T.scenario("T5. Neo4j 백엔드 + NetworkX 폴백")

    from azure_korean_doc_framework.core.graph_rag import (
        KnowledgeGraphManager, Entity, Relationship,
        HAS_NEO4J,
    )
    from azure_korean_doc_framework.config import Config

    # Neo4j 드라이버 가용성 확인
    T.check("HAS_NEO4J import 확인", isinstance(HAS_NEO4J, bool), str(HAS_NEO4J))

    # Config 기본 백엔드 = networkx
    T.check("기본 백엔드 = networkx", Config.GRAPH_STORAGE_BACKEND == "networkx")

    # Neo4j 설정값 존재
    T.check("NEO4J_URI 기본값", "bolt://" in Config.NEO4J_URI or "neo4j://" in Config.NEO4J_URI,
            Config.NEO4J_URI)

    # NetworkX 폴백 — __new__로 수동 생성, _neo4j_driver 없이도 동작 확인
    mgr = KnowledgeGraphManager.__new__(KnowledgeGraphManager)
    mgr.graph = nx.DiGraph()
    mgr._entity_cache = {}
    mgr._chunk_to_entities = {}
    mgr._entity_keyword_index = {}
    mgr._relation_keyword_index = {}
    mgr._entity_name_char_index = {}
    mgr._edge_desc_char_index = {}
    mgr._normalized_name_map = {}
    mgr._communities = []
    mgr._community_summaries = {}
    mgr._injections = {}
    mgr._synonym_map = {}
    mgr.gleaning_passes = 1
    mgr.mix_graph_weight = 0.4
    mgr.client = None
    mgr.model_name = "test"
    mgr._is_gpt5 = False
    mgr.entity_types = ["기관", "기술"]
    # _neo4j_driver 를 명시적으로 설정하지 않음 → getattr 안전 체크 테스트

    # 엔티티 추가 (no _neo4j_driver → getattr fallback)
    e1 = Entity(name="테스트엔티티", entity_type="기술", description="테스트용")
    mgr._add_entity(e1)
    T.check("_neo4j_driver 없이 엔티티 추가 성공", mgr.graph.number_of_nodes() == 1)

    # 관계 추가
    e2 = Entity(name="관련엔티티", entity_type="기관", description="연결 테스트")
    mgr._add_entity(e2)
    r1 = Relationship(
        source="테스트엔티티", target="관련엔티티",
        relation_type="테스트관계", description="테스트", weight=1.0, keywords="test"
    )
    mgr._add_relationship(r1)
    T.check("_neo4j_driver 없이 관계 추가 성공", mgr.graph.number_of_edges() == 1)

    # clear도 안전 (neo4j 없이)
    mgr.clear()
    T.check("_neo4j_driver 없이 clear 성공",
            mgr.graph.number_of_nodes() == 0 and mgr.graph.number_of_edges() == 0)

    # Neo4j 관련 메서드 존재 확인
    T.check("_neo4j_sync_entity 메서드", hasattr(mgr, "_neo4j_sync_entity"))
    T.check("_neo4j_sync_relationship 메서드", hasattr(mgr, "_neo4j_sync_relationship"))
    T.check("_neo4j_clear 메서드", hasattr(mgr, "_neo4j_clear"))
    T.check("_neo4j_load_to_networkx 메서드", hasattr(mgr, "_neo4j_load_to_networkx"))


# ==================== T6. Agentic Retrieval 데이터 모델 ====================

def test_t6_agentic_retrieval():
    T.scenario("T6. Agentic Retrieval 데이터 모델")

    from azure_korean_doc_framework.core.agentic_retrieval import (
        AgenticRetrievalResult,
        AgenticRetrievalManager,
    )
    from azure_korean_doc_framework.config import Config

    # AgenticRetrievalResult 데이터 모델
    result = AgenticRetrievalResult()
    T.check("기본 answer 빈 문자열", result.answer == "")
    T.check("기본 citations 빈 리스트", result.citations == [])
    T.check("기본 query_plan 빈 리스트", result.query_plan == [])
    T.check("기본 reasoning_effort = low", result.reasoning_effort == "low")
    T.check("기본 output_mode = extractive_data", result.output_mode == "extractive_data")
    T.check("기본 raw_response = None", result.raw_response is None)

    # 필드 설정
    result_filled = AgenticRetrievalResult(
        answer="테스트 답변",
        citations=[{"title": "doc.pdf", "chunk": "관련 내용"}],
        query_plan=["서브쿼리1", "서브쿼리2"],
        reasoning_effort="medium",
        output_mode="answer_synthesis",
    )
    T.check("answer 설정", result_filled.answer == "테스트 답변")
    T.check("citations 설정", len(result_filled.citations) == 1)
    T.check("query_plan 설정", len(result_filled.query_plan) == 2)
    T.check("reasoning_effort 설정", result_filled.reasoning_effort == "medium")

    # Config 설정
    T.check("AGENTIC_RETRIEVAL_ENABLED is bool", isinstance(Config.AGENTIC_RETRIEVAL_ENABLED, bool))
    T.check("AGENTIC_KB_NAME is str", isinstance(Config.AGENTIC_KB_NAME, str))

    # AgenticRetrievalManager 클래스 확인 (초기화는 Azure 연결 필요하므로 메서드만 확인)
    T.check("retrieve 메서드 존재", hasattr(AgenticRetrievalManager, "retrieve"))
    T.check("create_knowledge_base 메서드 존재", hasattr(AgenticRetrievalManager, "create_knowledge_base"))


# ==================== T7. Prompt Caching 설정 검증 ====================

def test_t7_prompt_caching():
    T.scenario("T7. Prompt Caching 설정 검증")

    from azure_korean_doc_framework.config import Config

    # Prompt Caching은 store=true 파라미터로 구현
    T.check("PROMPT_CACHING_ENABLED is bool", isinstance(Config.PROMPT_CACHING_ENABLED, bool))

    # MultiModelManager의 get_completion에서 store 파라미터 추가 확인
    import inspect
    from azure_korean_doc_framework.core.multi_model_manager import MultiModelManager

    source = inspect.getsource(MultiModelManager.get_completion)
    T.check("get_completion에 store 파라미터 존재",
            "store" in source and "PROMPT_CACHING_ENABLED" in source)

    # Responses API에서도 store 사용 확인
    responses_source = inspect.getsource(MultiModelManager._call_responses_api)
    T.check("_call_responses_api에 store 파라미터",
            "store" in responses_source and "PROMPT_CACHING_ENABLED" in responses_source)


# ==================== T8. 크로스 모듈 통합 ====================

def test_t8_cross_module_integration():
    T.scenario("T8. 크로스 모듈 통합 (Semantic Cache + LLM Cache)")

    from azure_korean_doc_framework.core.llm_cache import (
        LLMResponseCache, SemanticCache, _cosine_similarity,
    )

    tmpdir = tempfile.mkdtemp(prefix="v60_test_")

    try:
        # LLM Cache + Semantic Cache 병행 사용
        exact_cache = LLMResponseCache(
            cache_dir=tmpdir, max_memory_entries=50, enabled=True
        )
        semantic_cache = SemanticCache(threshold=0.92, max_entries=50, enabled=True)

        # 시나리오: 동일 질문 → exact cache 히트
        prompt = "한국 GDP 성장률은?"
        exact_cache.put(prompt, "2.5%입니다.", model_key="gpt-5.4")
        exact_hit = exact_cache.get(prompt, model_key="gpt-5.4")
        T.check("Exact Cache 히트", exact_hit == "2.5%입니다.")

        # 시나리오: 유사 질문 → semantic cache 히트
        emb_original = [0.12, 0.45, 0.78, 0.23, 0.56]
        semantic_cache.put(emb_original, "2.5%입니다.", model="gpt-5.4")

        emb_similar = [0.121, 0.449, 0.781, 0.229, 0.561]
        semantic_hit = semantic_cache.get(emb_similar)
        T.check("Semantic Cache 유사 질문 히트", semantic_hit == "2.5%입니다.")

        # 시나리오: 완전 다른 질문 → 둘 다 미스
        exact_miss = exact_cache.get("미국 금리는?", model_key="gpt-5.4")
        emb_different = [0.9, 0.1, 0.05, 0.8, 0.02]
        semantic_miss = semantic_cache.get(emb_different)
        T.check("다른 질문 Exact Cache 미스", exact_miss is None)
        T.check("다른 질문 Semantic Cache 미스", semantic_miss is None)

        # 통계 확인 (두 캐시 모두 독립적)
        exact_stats = exact_cache.get_stats()
        semantic_stats = semantic_cache.get_stats()
        T.check("Exact Cache hit_rate > 0", exact_stats["hit_rate"] > 0)
        T.check("Semantic Cache hits > 0", semantic_stats["hits"] > 0)

        # LLM Cache에 Semantic Cache 결과를 매핑하는 패턴 예시
        # (실제 에이전트에서는: exact miss → semantic hit → 응답 반환)
        query = "한국의 경제 성장률 알려줘"
        exact_result = exact_cache.get(query, model_key="gpt-5.4")
        if exact_result is None:
            # exact miss → semantic cache 시도
            query_emb = [0.119, 0.451, 0.779, 0.231, 0.559]  # 유사 임베딩
            semantic_result = semantic_cache.get(query_emb)
            if semantic_result is not None:
                # semantic hit → exact cache에도 저장 (다음 호출 가속)
                exact_cache.put(query, semantic_result, model_key="gpt-5.4")
                final = semantic_result
            else:
                final = None
        else:
            final = exact_result

        T.check("Exact miss → Semantic hit → 결과 반환",
                final == "2.5%입니다.")

        # 추가 검증: exact cache에 semantic 결과가 저장되었는지
        re_exact = exact_cache.get(query, model_key="gpt-5.4")
        T.check("Semantic hit 결과가 Exact Cache에 저장됨", re_exact == "2.5%입니다.")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ==================== 메인 ====================

if __name__ == "__main__":
    test_t1_config_v60()
    test_t2_semantic_cache()
    test_t3_responses_api()
    test_t4_content_understanding()
    test_t5_neo4j_backend()
    test_t6_agentic_retrieval()
    test_t7_prompt_caching()
    test_t8_cross_module_integration()
    sys.exit(T.summary())
