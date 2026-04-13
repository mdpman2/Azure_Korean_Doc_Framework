import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
from azure.search.documents.models import VectorizedQuery
from .multi_model_manager import MultiModelManager
from .schema import AnswerArtifacts, PipelineStep, SearchResult
from .hooks import HookEvent, HookRegistry
from .error_recovery import ErrorRecoveryManager, RetryPolicy
from .streaming import ContextCompactor, StreamChunk, StreamingManager
from .web_tools import WebFetchTool, WebSearchTool
from ..utils.azure_clients import AzureClientFactory
from ..config import Config
from ..generation.evidence_extractor import EvidenceExtractor
from ..guardrails.retrieval_gate import RetrievalQualityGate
from ..guardrails.numeric_verifier import NumericVerifier
from ..guardrails.pii import KoreanPIIDetector
from ..guardrails.injection import PromptInjectionDetector
from ..guardrails.faithfulness import FaithfulnessChecker
from ..guardrails.hallucination import HallucinationDetector
from ..guardrails.question_classifier import QuestionClassifier
from ..utils.search_schema import apply_search_runtime_mapping

# 공통 RAG 시스템 프롬프트 (answer_question / graph_enhanced_answer 공유)
_RAG_SYSTEM_PROMPT = (
    "당신은 문서 분석 및 Q&A 전문가입니다. "
    "주어진 [Context] 내용을 바탕으로 사용자의 [Question]에 한국어로 친절하고 정확하게 답변하세요. "
    "\n\n### 답변 규칙:"
    "\n1. 답변 시 반드시 해당 정보의 **출처(문서명 또는 제목)**를 언급하세요. (예: '...입니다 [출처: 성균관대.pdf]')"
    "\n2. 여러 문서에서 정보를 취합한 경우, 각각의 출처를 밝히세요."
    "\n3. 추출된 정보가 부족하면 아는 범위 내에서 최선을 다해 답변하되, 정보가 전혀 없다면 솔직하게 모른다고 답하세요."
)

_GRAPH_RAG_SYSTEM_PROMPT = (
    _RAG_SYSTEM_PROMPT
    + "\n4. Knowledge Graph 정보가 있으면 엔티티 간 관계를 활용하여 더 풍부한 답변을 생성하세요."
)


class KoreanDocAgent:
    """
    한국어 문서 분석 및 Q&A 전문가 검색 에이전트.

    Azure AI Search의 Hybrid Search (BM25 키워드 + Vector 유사성 + Semantic Ranking)을 활용하여
    문맥을 찾고, GPT-5.4를 통해 지능적인 답변을 생성합니다.

    [2026-04 v5.0 업데이트]
    - 병렬 도구 실행: 안전한 도구(검색, 임베딩)를 병렬 배치로 처리
    - 자동 컨텍스트 압축: 토큰 한계 근접 시 LLM으로 핵심 요약
    - Web Search/Fetch: 외부 웹 정보 보강
    - 스트리밍 응답: 토큰 단위 실시간 출력
    - Hook 시스템: 파이프라인 단계별 콜백 등록
    - Agent Routing: 질문 유형별 최적 모델 자동 선택
    - 에러 자동 복구: 429/413/500 에러에 대한 재시도/폴백/압축
    - 서브에이전트 위임: 복합 질문 분해 → 병렬 조사 → 종합

    [2026-04 v4.6]
    - live Azure AI Search 인덱스 스키마를 조회해 런타임 필드 매핑 자동 적용
    - evidence 기반 답변에서 citation 수를 제한하고 근거 문서를 diagnostics 상단으로 재정렬
    - faithfulness / hallucination 검증에서 citation 라인을 제외한 answer body 사용

    [2026-02 v4.1 - Contextual Retrieval]
    - Contextual Retrieval (Anthropic 방식): 청크에 맥락 추가하여 BM25 + 벡터 검색 동시 개선
    - Hybrid Search: BM25 키워드 검색 + Vector 유사성 검색 결합 (Reciprocal Rank Fusion)
    - Azure AI Search 네이티브 하이브리드 검색 + 시맨틱 래킹 활용
    - 기존 Azure AI Search 인덱스도 환경 변수 기반 필드 매핑으로 재사용 가능
    - Graph-Enhanced RAG (LightRAG 기반 Knowledge Graph 연동)
    - GPT-5.4 기본 모델 사용
    - Query Rewrite 지원 (시맨틱 쿼리 확장)
    """

    def __init__(self, model_key: Optional[str] = None, graph_manager=None):
        """
        KoreanDocAgent를 초기화합니다.

        Args:
            model_key: 답변 생성 시 기본으로 사용할 모델 키 (Config.MODELS에 정의된 키).
                      기본값: Config.DEFAULT_MODEL (gpt-5.4)
            graph_manager: KnowledgeGraphManager 인스턴스 (Graph RAG 사용 시)
        """
        apply_search_runtime_mapping()
        self.model_manager = MultiModelManager(default_model=model_key or Config.DEFAULT_MODEL)
        self.search_client = AzureClientFactory.get_search_client()

        # 임베딩 클라이언트 (벡터 검색용) - 기본 엔드포인트 사용 (text-embedding-3-small)
        self.embedding_client = AzureClientFactory.get_openai_client(is_advanced=False)

        # LLM 클라이언트 (Query Rewrite용) - 고성능 엔드포인트
        self.llm_client = AzureClientFactory.get_openai_client(is_advanced=True)

        # Query Rewrite 활성화 여부 (환경 변수로 제어 가능)
        self.enable_query_rewrite = Config.QUERY_REWRITE_ENABLED

        # [v4.0] Graph RAG 매니저 (LightRAG 기반)
        self.graph_manager = graph_manager

        self.question_classifier = QuestionClassifier()
        self.evidence_extractor = EvidenceExtractor(self.model_manager)
        self.retrieval_gate = RetrievalQualityGate(
            min_top_score=Config.RETRIEVAL_GATE_MIN_TOP_SCORE,
            min_doc_count=Config.RETRIEVAL_GATE_MIN_DOC_COUNT,
            min_doc_score=Config.RETRIEVAL_GATE_MIN_DOC_SCORE,
            soft_mode=Config.RETRIEVAL_GATE_SOFT_MODE,
        )
        self.numeric_verifier = NumericVerifier()
        self.pii_detector = KoreanPIIDetector()
        self.injection_detector = PromptInjectionDetector(self.model_manager)
        self.faithfulness_checker = FaithfulnessChecker(
            self.model_manager,
            threshold=Config.FAITHFULNESS_THRESHOLD,
        )
        self.hallucination_detector = HallucinationDetector(
            self.model_manager,
            threshold=Config.HALLUCINATION_THRESHOLD,
        )

        # ── [v5.0] 신규 시스템 초기화 ──

        # Hook 시스템
        self.hook_registry = HookRegistry() if Config.HOOKS_ENABLED else None

        # 스트리밍 매니저
        self.streaming_manager = StreamingManager(model_key=model_key) if Config.STREAMING_ENABLED else None

        # 자동 컨텍스트 압축
        self.context_compactor = ContextCompactor(
            max_context_tokens=Config.AUTO_COMPACT_MAX_CONTEXT_TOKENS,
            compact_threshold_ratio=Config.AUTO_COMPACT_THRESHOLD_RATIO,
        ) if Config.AUTO_COMPACT_ENABLED else None

        # 웹 검색/수집 도구
        self.web_search_tool = WebSearchTool(
            bing_api_key=Config.BING_API_KEY or None,
            max_results=Config.WEB_SEARCH_MAX_RESULTS,
        ) if Config.WEB_SEARCH_ENABLED else None
        self.web_fetch_tool = WebFetchTool(
            max_chars=Config.WEB_FETCH_MAX_CHARS,
        ) if Config.WEB_SEARCH_ENABLED else None

        # 에러 자동 복구 매니저
        self.error_recovery = ErrorRecoveryManager(
            RetryPolicy(
                max_retries=Config.ERROR_RECOVERY_MAX_RETRIES,
                base_delay=Config.ERROR_RECOVERY_BASE_DELAY,
                fallback_models=[m.strip() for m in Config.ERROR_RECOVERY_FALLBACK_MODELS if m.strip()],
            )
        ) if Config.ERROR_RECOVERY_ENABLED else None

        # 서브에이전트 매니저 — delegate() 외부 호출 시 lazy init
        self._sub_agent_manager = None

    # ==================== [v5.0] Hook 헬퍼 ====================

    def _run_hook(self, event: HookEvent, data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """훅을 실행하고 수정된 데이터를 반환합니다. 차단 시 None 반환."""
        if not self.hook_registry:
            return data or {}
        result = self.hook_registry.run(event, data or {})
        if result.blocked:
            return None
        if result.modified_data:
            merged = dict(data or {})
            merged.update(result.modified_data)
            return merged
        return data or {}

    # ==================== [v5.0] Agent Routing ====================

    def _route_model_for_question(self, question: str, base_model_key: Optional[str] = None) -> str:
        """질문 유형에 따라 최적 모델을 선택합니다."""
        if not Config.AGENT_ROUTING_ENABLED or base_model_key:
            return base_model_key or Config.DEFAULT_MODEL

        question_type = self.question_classifier.classify(question)
        routed = Config.AGENT_ROUTING_MAP.get(question_type.category)
        if routed and routed in Config.MODELS:
            print(f"   🔀 Agent Routing: {question_type.category} → {routed}")
            return routed
        return Config.DEFAULT_MODEL

    # ==================== [v5.0] 병렬 도구 실행 ====================

    def _parallel_search(
        self,
        question: str,
        search_queries: List[str],
        top_k: int = 5,
    ) -> Tuple[List[SearchResult], Optional[str]]:
        """
        병렬로 벡터 검색 + 웹 검색을 동시 실행합니다.

        Returns:
            (search_results, web_context): 검색 결과 + 웹 보강 컨텍스트
        """
        if not Config.PARALLEL_TOOL_ENABLED:
            results = self._vector_search(question, search_queries, top_k)
            web_ctx = self._web_search_augment(question) if self.web_search_tool else None
            return results, web_ctx

        vector_results = []
        web_context = None

        with ThreadPoolExecutor(max_workers=Config.PARALLEL_TOOL_MAX_WORKERS) as executor:
            future_vector = executor.submit(self._vector_search, question, search_queries, top_k)

            future_web = None
            if self.web_search_tool:
                future_web = executor.submit(self._web_search_augment, question)

            try:
                vector_results = future_vector.result(timeout=30)
            except Exception as e:
                print(f"   ⚠️ 벡터 검색 실패: {e}")

            if future_web:
                try:
                    web_context = future_web.result(timeout=15)
                except Exception:
                    pass

        return vector_results, web_context

    # ==================== [v5.0] 웹 검색 보강 ====================

    def _web_search_augment(self, question: str) -> Optional[str]:
        """웹 검색으로 추가 컨텍스트를 수집합니다."""
        if not self.web_search_tool:
            return None

        try:
            results = self.web_search_tool.search(question)
            if not results:
                return None
            snippets = [f"[웹: {r.title}]\n{r.snippet}" for r in results[:3] if r.snippet]
            return "\n\n".join(snippets) if snippets else None
        except Exception as e:
            print(f"   ⚠️ 웹 검색 실패: {e}")
            return None

    # ==================== [v5.0] 스트리밍 답변 ====================

    def answer_question_streaming(
        self,
        question: str,
        model_key: Optional[str] = None,
        top_k: int = 5,
        use_query_rewrite: bool = True,
    ) -> Generator[StreamChunk, None, None]:
        """
        스트리밍으로 RAG 답변을 생성합니다. [v5.0 신규]

        검색 → 가드레일(injection/gate) → 스트리밍 답변 생성

        Yields:
            StreamChunk: 텍스트 청크 (is_final=True로 완료 판별)
        """
        routed_model = self._route_model_for_question(question, model_key)

        # [v4.7-fix] Injection 검사를 검색 이전에 수행 (보안 강화)
        if Config.INJECTION_DETECTION_ENABLED:
            injection = self.injection_detector.detect(question)
            if injection.blocked:
                yield StreamChunk(text="입력 내용이 안전하지 않아 요청을 처리할 수 없습니다.", is_final=True)
                return

        # Hook: pre_search
        hook_data = self._run_hook(HookEvent.PRE_SEARCH, {"question": question})
        if hook_data is None:
            yield StreamChunk(text="Hook에 의해 차단되었습니다.", is_final=True)
            return

        search_queries = self._prepare_search(question, use_query_rewrite)
        search_results, web_context = self._parallel_search(question, search_queries, top_k)

        # Hook: post_search
        self._run_hook(HookEvent.POST_SEARCH, {
            "question": question, "result_count": len(search_results),
        })

        # Retrieval gate
        if Config.RETRIEVAL_GATE_ENABLED:
            gate = self.retrieval_gate.evaluate(search_results)
            if not gate.passed and not gate.soft_fail:
                yield StreamChunk(text=Config.RETRIEVAL_GATE_NOT_FOUND_MESSAGE, is_final=True)
                return

        # 컨텍스트 구성 (자동 압축 포함)
        contexts = self._format_contexts(search_results)
        if web_context:
            contexts.append(f"[웹 검색 보강]\n{web_context}")

        if self.context_compactor and self.context_compactor.should_compact(contexts):
            compact_result = self.context_compactor.compact_contexts(contexts, question=question)
            contexts = [compact_result.summary]
            print(f"   🗜️ 컨텍스트 압축: {compact_result.original_token_count} → {compact_result.compacted_token_count} tokens")

        # 스트리밍 생성
        if self.streaming_manager:
            yield from self.streaming_manager.stream_rag_answer(
                question=question,
                contexts=contexts,
                system_prompt=_RAG_SYSTEM_PROMPT,
                model_key=routed_model,
            )
        else:
            # 폴백: 일반 생성 후 한번에 반환
            answer = self._generate_standard_answer(question, contexts, model_key=routed_model)
            yield StreamChunk(text=answer, is_final=True)

    # ==================== [v5.0] 서브에이전트 위임 ====================

    @property
    def sub_agent_manager(self):
        """서브에이전트 매니저를 lazy 초기화합니다."""
        if self._sub_agent_manager is None and Config.SUB_AGENT_ENABLED:
            from .sub_agent import SubAgentManager
            self._sub_agent_manager = SubAgentManager(
                answer_fn=self.answer_question,
                model_manager=self.model_manager,
                max_workers=Config.SUB_AGENT_MAX_WORKERS,
                timeout=Config.SUB_AGENT_TIMEOUT,
            )
        return self._sub_agent_manager

    def answer_question_with_delegation(
        self,
        question: str,
        model_key: Optional[str] = None,
        top_k: int = 5,
        use_query_rewrite: bool = True,
        force_decompose: bool = False,
    ) -> Union[str, Any]:
        """
        서브에이전트 위임이 적용된 답변. 복합 질문 자동 감지. [v5.0 신규]

        Args:
            question: 사용자 질문
            model_key: 모델 키
            top_k: 검색 결과 수
            use_query_rewrite: 쿼리 확장 여부
            force_decompose: 강제 분해 여부

        Returns:
            답변 문자열
        """
        if not self.sub_agent_manager:
            return self.answer_question(question, model_key=model_key, top_k=top_k, use_query_rewrite=use_query_rewrite)

        delegation = self.sub_agent_manager.delegate(
            question=question,
            model_key=model_key,
            force_decompose=force_decompose,
            top_k=top_k,
            use_query_rewrite=use_query_rewrite,
        )

        if delegation.was_decomposed:
            print(f"   🔀 서브에이전트 완료: {len(delegation.sub_results)}개 태스크, {delegation.total_elapsed_ms:.0f}ms")
            return delegation.synthesized_answer

        # 분해 불필요 → 일반 답변
        return self.answer_question(question, model_key=model_key, top_k=top_k, use_query_rewrite=use_query_rewrite)

    def _rewrite_query(self, question: str) -> List[str]:
        """
        GPT-5.4를 사용하여 쿼리를 의미적으로 확장합니다.
        오타 교정, 동의어 생성, 다양한 표현으로 쿼리 변형.

        Args:
            question: 원본 질문

        Returns:
            확장된 쿼리 리스트 (원본 포함)
        """
        if not self.enable_query_rewrite:
            return [question]

        try:
            rewrite_prompt = f"""다음 한국어 질문을 검색에 최적화된 여러 형태로 변형해주세요.
오타 교정, 동의어 사용, 다양한 표현 방식을 포함하세요.
원본 질문도 포함하여 최대 3개의 쿼리를 JSON 배열로 반환하세요.

원본 질문: {question}

출력 형식: ["쿼리1", "쿼리2", "쿼리3"]"""

            response = self.llm_client.chat.completions.create(
                model=Config.MODELS.get(Config.DEFAULT_MODEL, Config.DEFAULT_MODEL),
                messages=[{"role": "user", "content": rewrite_prompt}],
                temperature=0.3,
                max_completion_tokens=200
            )

            result = response.choices[0].message.content.strip()
            # JSON 배열 파싱 (LLM 응답 형식 오류 방어)
            if result.startswith("["):
                try:
                    queries = json.loads(result)
                    if isinstance(queries, list) and queries:
                        return [str(q) for q in queries[:3]]
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"   ⚠️ Query rewrite JSON 파싱 실패: {e} | 응답: {result[:200]}")
            return [question]

        except Exception as e:
            print(f"   ⚠️ Query rewrite failed, using original: {e}")
            return [question]

    # ==================== 하이브리드 검색 (BM25 + Vector + Semantic Ranking) ====================

    def _vector_search(
        self,
        question: str,
        search_queries: List[str],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Azure AI Search 하이브리드 검색 (BM25 키워드 + 벡터 유사성 + 시맨틱 랭킹)

        Contextual Retrieval 기반:
        - BM25 키워드 검색: Config.SEARCH_CONTENT_FIELD (맥락 포함 텍스트)
        - 벡터 유사성 검색: Config.SEARCH_VECTOR_FIELD (맥락 포함 임베딩)
        - 시맨틱 랭킹: Config.SEARCH_SEMANTIC_CONFIG로 최종 재순위
        - Reciprocal Rank Fusion (RRF): BM25 + Vector 결과 자동 결합

        검색 시 맥락 포함 필드로 BM25/벡터 검색 → 높은 검색 정확도
        결과 반환 시 Config.SEARCH_ORIGINAL_CONTENT_FIELD로 원본 텍스트 반환 → 깔끔한 답변 생성

        Args:
            question: 원본 질문 (임베딩용)
            search_queries: 검색 쿼리 리스트 (Query Rewrite 결과 포함)
            top_k: 검색할 문서 수

        Returns:
            검색된 컨텍스트 리스트 (원본 텍스트 기반)
        """
        # 임베딩은 원본 질문으로 1회만 생성 (각 search_query별 중복 API 호출 방지)
        embedding_response = self.embedding_client.embeddings.create(
            input=[question],
            model=Config.EMBEDDING_DEPLOYMENT
        )
        query_vector = embedding_response.data[0].embedding
        vector_query = VectorizedQuery(
            vector=query_vector,
            k=50,
            fields=Config.SEARCH_VECTOR_FIELD,
        )

        select_fields = list(dict.fromkeys(filter(None, [
            Config.SEARCH_ID_FIELD,
            Config.SEARCH_CONTENT_FIELD,
            Config.SEARCH_ORIGINAL_CONTENT_FIELD,
            Config.SEARCH_TITLE_FIELD,
            Config.SEARCH_SOURCE_FIELD,
            Config.SEARCH_CITATION_FIELD,
            Config.SEARCH_BOUNDING_BOX_FIELD,
            Config.SEARCH_SOURCE_REGIONS_FIELD,
        ])))

        deduplicated_results = {}

        try:
            for search_query in search_queries:
                # Hybrid Search: BM25 (search_text) + Vector (vector_queries) + Semantic Ranking
                # Azure AI Search는 내부적으로 RRF(Reciprocal Rank Fusion)를 사용하여
                # BM25 결과와 Vector 결과를 자동으로 결합합니다.
                results = self.search_client.search(
                    search_text=search_query,           # BM25 키워드 검색 (Contextual BM25)
                    vector_queries=[vector_query],       # 벡터 유사성 검색 (Contextual Embeddings)
                    select=select_fields,
                    query_type="semantic",               # Semantic Ranker로 최종 재순위
                    semantic_configuration_name=Config.SEARCH_SEMANTIC_CONFIG,
                    top=top_k
                )

                for r in results:
                    # 답변 생성에는 원본 텍스트 사용 (맥락 제외)
                    content = (
                        r.get(Config.SEARCH_ORIGINAL_CONTENT_FIELD)
                        or r.get(Config.SEARCH_CONTENT_FIELD)
                        or ""
                    )
                    if not content:
                        continue

                    source = (
                        r.get(Config.SEARCH_SOURCE_FIELD)
                        or r.get(Config.SEARCH_TITLE_FIELD)
                        or r.get(Config.SEARCH_ID_FIELD)
                        or "알 수 없는 출처"
                    )
                    score = self._extract_search_score(r)
                    content_key = hash((source, content))
                    candidate = SearchResult(
                        content=content,
                        source=source,
                        score=score,
                        metadata={
                            "raw_chunk": r.get(Config.SEARCH_CONTENT_FIELD, ''),
                            "citation": r.get(Config.SEARCH_CITATION_FIELD, '') if Config.SEARCH_CITATION_FIELD else '',
                            "bounding_box": self._loads_json_value(r.get(Config.SEARCH_BOUNDING_BOX_FIELD)) if Config.SEARCH_BOUNDING_BOX_FIELD else None,
                            "source_regions": self._loads_json_value(r.get(Config.SEARCH_SOURCE_REGIONS_FIELD)) if Config.SEARCH_SOURCE_REGIONS_FIELD else None,
                        },
                    )

                    existing = deduplicated_results.get(content_key)
                    if existing is None or candidate.score > existing.score:
                        deduplicated_results[content_key] = candidate

        except Exception as e:
            print(f"   ❌ Search failed: {e}")

        ranked_results = sorted(
            deduplicated_results.values(),
            key=lambda item: item.score,
            reverse=True,
        )
        return ranked_results[:top_k * 2]

    def _extract_search_score(self, result) -> float:
        for key in ("@search.reranker_score", "@search.score"):
            value = result.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return 0.0

    def _loads_json_value(self, value: Any) -> Any:
        if value in (None, ""):
            return None
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    def _build_exact_citation_label(self, result: SearchResult) -> str:
        citation = result.metadata.get("citation")
        if citation:
            return f"[출처: {citation}]"
        return f"[출처: {result.source}]"

    def _answer_body_without_citations(self, answer: str) -> str:
        """Strip trailing citation lines so downstream validators score only the answer body."""
        filtered_lines = []
        for raw_line in (answer or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[출처:"):
                continue
            filtered_lines.append(line)
        return "\n".join(filtered_lines).strip()

    def _limit_citation_candidates(
        self,
        search_results: List[SearchResult],
        preferred_sources: Optional[List[str]] = None,
        max_citations: int = 2,
    ) -> List[SearchResult]:
        """Select a small, source-deduplicated citation set, preferring evidence-matched sources."""
        if not search_results:
            return []

        preferred = set(preferred_sources or [])
        selected = []
        seen_sources = set()

        if preferred:
            for result in search_results:
                if result.source not in preferred or result.source in seen_sources:
                    continue
                selected.append(result)
                seen_sources.add(result.source)
                if len(selected) >= max(1, len(preferred_sources or [])):
                    break
            if selected:
                return selected

        for result in search_results:
            if result.source in seen_sources:
                continue
            selected.append(result)
            seen_sources.add(result.source)
            if len(selected) >= max_citations:
                break
        return selected

    def _append_exact_citations(
        self,
        answer: str,
        search_results: List[SearchResult],
        preferred_sources: Optional[List[str]] = None,
        max_citations: int = 2,
    ) -> str:
        if not Config.EXACT_CITATION_ENABLED or not search_results:
            return answer

        answer_body = self._answer_body_without_citations(answer)
        citation_candidates = self._limit_citation_candidates(
            search_results,
            preferred_sources=preferred_sources,
            max_citations=max_citations,
        )
        citation_labels = []
        seen = set()
        for result in citation_candidates:
            label = self._build_exact_citation_label(result)
            if label in answer_body or label in answer:
                continue
            if label in seen:
                continue
            seen.add(label)
            citation_labels.append(label)

        if not citation_labels:
            return answer

        citation_block = "\n".join(citation_labels)
        if citation_block in answer:
            return answer
        return f"{answer}\n\n{citation_block}" if answer else citation_block

    def _format_contexts(self, results: List[SearchResult]) -> List[str]:
        return [f"[출처: {item.source}]\n{item.content}" for item in results]

    def _normalize_match_text(self, text: str) -> str:
        return " ".join((text or "").split()).strip().lower()

    def _extract_query_terms(self, question: str) -> List[str]:
        stopwords = {
            "무엇", "무엇인가요", "무엇입니까", "인가요", "입니까", "알려주세요", "알려줘",
            "해주세요", "해줘", "관련", "대한", "에서", "질문", "답", "찾아", "찾기",
            "이름", "성함", "담당자",
        }
        tokens = re.findall(r"[0-9a-zA-Z가-힣]+", question or "")
        normalized = []
        for token in tokens:
            cleaned = token.strip().lower()
            if len(cleaned) <= 1:
                continue
            if cleaned in stopwords:
                continue
            normalized.append(cleaned)
        return list(dict.fromkeys(normalized))

    def _rerank_search_results_for_evidence(
        self,
        question: str,
        search_results: List[SearchResult],
        evidence_used,
        question_category: str,
    ) -> List[SearchResult]:
        """Promote documents that contain the extracted answer or evidence sentences.

        Retrieval score is kept as a tiebreaker, but evidence-matched sources are
        surfaced first so diagnostics and citation selection reflect the actual
        grounding document more closely.
        """
        if not search_results or evidence_used is None:
            return search_results

        normalized_answer = self._normalize_match_text(evidence_used.answer)
        normalized_evidence = [self._normalize_match_text(sentence) for sentence in evidence_used.evidence_sentences if sentence]
        preferred_sources = set(evidence_used.sources or [])
        query_terms = self._extract_query_terms(question)

        def evidence_rank(result: SearchResult) -> tuple:
            normalized_content = self._normalize_match_text(result.content)
            normalized_source = self._normalize_match_text(result.source)
            boost = 0.0
            matched_evidence_count = 0

            if result.source in preferred_sources:
                boost += 100.0
            if normalized_answer and normalized_answer in normalized_content:
                boost += 20.0
            for sentence in normalized_evidence:
                if sentence and sentence in normalized_content:
                    matched_evidence_count += 1
                    boost += 12.0

            if question_category == "extraction":
                keyword_overlap = sum(1 for term in query_terms if term in normalized_content or term in normalized_source)
                boost += keyword_overlap * 2.0

            return (boost, matched_evidence_count, result.score)

        return sorted(search_results, key=evidence_rank, reverse=True)

    def _build_diagnostics(
        self,
        question: str,
        search_queries: Optional[List[str]],
        search_results: List[SearchResult],
        model_key: Optional[str],
        graph_context_used: bool = False,
        gate_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not Config.ANSWER_DIAGNOSTICS_ENABLED:
            return {}

        query_variants = search_queries or [question]
        top_score = max((item.score for item in search_results), default=0.0)
        resolved_model_key = model_key or getattr(self.model_manager, "default_model", Config.DEFAULT_MODEL)
        unique_top_sources = list(dict.fromkeys(item.source for item in search_results if item.source))[:3]
        return {
            "question_length": len(question),
            "model_key": resolved_model_key,
            "query_rewrite_enabled": self.enable_query_rewrite,
            "query_variant_count": len(query_variants),
            "query_variants": query_variants,
            "search_result_count": len(search_results),
            "top_score": top_score,
            "top_sources": unique_top_sources,
            "graph_context_used": graph_context_used,
            "gate_reason": gate_reason,
        }

    def _finalize_artifacts(
        self,
        *,
        question: str,
        search_queries: Optional[List[str]],
        search_results: List[SearchResult],
        answer: str,
        contexts: Optional[List[str]] = None,
        steps: Optional[List[PipelineStep]] = None,
        model_key: Optional[str] = None,
        graph_context_used: bool = False,
        gate_reason: Optional[str] = None,
    ) -> AnswerArtifacts:
        return AnswerArtifacts(
            answer=answer,
            contexts=contexts or [],
            steps=steps or [],
            search_results=search_results,
            gate_reason=gate_reason,
            diagnostics=self._build_diagnostics(
                question=question,
                search_queries=search_queries,
                search_results=search_results,
                model_key=model_key,
                graph_context_used=graph_context_used,
                gate_reason=gate_reason,
            ),
        )

    def _generate_standard_answer(
        self,
        question: str,
        contexts: List[str],
        model_key: Optional[str] = None,
        system_prompt: str = _RAG_SYSTEM_PROMPT,
    ) -> str:
        context_str = "\n\n".join(contexts) if contexts else "관련된 문서 내용을 찾을 수 없습니다."
        user_prompt = f"[Context]\n{context_str}\n\n[Question]\n{question}"

        # [v5.0] Hook: pre_generation
        self._run_hook(HookEvent.PRE_GENERATION, {"question": question, "context_count": len(contexts)})

        # [v5.0] 에러 자동 복구 적용
        return self.model_manager.get_completion_with_retry(
            prompt=user_prompt,
            model_key=model_key,
            system_message=system_prompt,
        )

    def _run_guardrailed_answer(
        self,
        question: str,
        search_results: List[SearchResult],
        model_key: Optional[str] = None,
        system_prompt: str = _RAG_SYSTEM_PROMPT,
        search_queries: Optional[List[str]] = None,
        graph_context_used: bool = False,
    ) -> AnswerArtifacts:
        steps: List[PipelineStep] = []

        if Config.INJECTION_DETECTION_ENABLED:
            injection = self.injection_detector.detect(question, model_key=model_key)
            steps.append(PipelineStep(
                name="prompt_injection",
                passed=not injection.blocked,
                detail={"reason": injection.reason, "score": injection.score},
            ))
            if injection.blocked:
                return self._finalize_artifacts(
                    question=question,
                    search_queries=search_queries,
                    search_results=search_results,
                    answer="입력 내용이 안전하지 않아 요청을 처리할 수 없습니다.",
                    steps=steps,
                    model_key=model_key,
                    graph_context_used=graph_context_used,
                )

        gate_reason = None
        if Config.RETRIEVAL_GATE_ENABLED:
            gate_result = self.retrieval_gate.evaluate(search_results)
            steps.append(PipelineStep(
                name="retrieval_gate",
                passed=gate_result.passed,
                detail={
                    "reason": gate_result.reason,
                    "top_score": gate_result.top_score,
                    "qualifying_count": gate_result.qualifying_count,
                    "soft_fail": gate_result.soft_fail,
                },
            ))
            if not gate_result.passed and not gate_result.soft_fail:
                return self._finalize_artifacts(
                    question=question,
                    search_queries=search_queries,
                    search_results=search_results,
                    answer=Config.RETRIEVAL_GATE_NOT_FOUND_MESSAGE,
                    steps=steps,
                    model_key=model_key,
                    graph_context_used=graph_context_used,
                    gate_reason=gate_result.reason,
                )
            if not gate_result.passed:
                gate_reason = gate_result.reason

        question_type = self.question_classifier.classify(question)
        steps.append(PipelineStep(
            name="question_classification",
            passed=True,
            detail={"category": question_type.category, "reason": question_type.reason},
        ))

        effective_results = list(search_results)
        contexts = self._format_contexts(effective_results)
        answer = ""
        evidence_used = None
        if Config.EXACT_CITATION_ENABLED and search_results:
            if question_type.category == "regulatory":
                evidence_used = self.evidence_extractor.extract_and_answer(question, search_results, model_key=model_key)
            elif question_type.category == "extraction":
                evidence_used = self.evidence_extractor.extract_short_answer(question, search_results, model_key=model_key)

        if evidence_used is not None:
            effective_results = self._rerank_search_results_for_evidence(
                question,
                search_results,
                evidence_used,
                question_type.category,
            )
            contexts = self._format_contexts(effective_results)
            answer = evidence_used.answer
            answer = self._append_exact_citations(
                answer,
                effective_results,
                preferred_sources=evidence_used.sources,
                max_citations=1 if question_type.category == "extraction" else 2,
            )
            steps.append(PipelineStep(
                name="evidence_extraction",
                passed=True,
                detail={
                    "evidence_count": len(evidence_used.evidence_sentences),
                    "sources": evidence_used.sources,
                },
            ))
        else:
            answer = self._generate_standard_answer(question, contexts, model_key=model_key, system_prompt=system_prompt)
            steps.append(PipelineStep(name="generation", passed=not answer.startswith("❌")))

        if Config.PII_DETECTION_ENABLED:
            pii_matches = self.pii_detector.detect(answer)
            answer = self.pii_detector.mask(answer)
            steps.append(PipelineStep(
                name="pii_masking",
                passed=True,
                detail={"match_count": len(pii_matches)},
            ))

        if Config.NUMERIC_VERIFICATION_ENABLED and answer and search_results:
            verification = self.numeric_verifier.verify(answer, [item.content for item in effective_results])
            steps.append(PipelineStep(
                name="numeric_verification",
                passed=verification.passed,
                detail={
                    "total_numbers": verification.total_numbers_found,
                    "ungrounded": verification.ungrounded_numbers,
                },
            ))

        if Config.FAITHFULNESS_ENABLED and answer and search_results:
            faithfulness = self.faithfulness_checker.verify(
                self._answer_body_without_citations(answer),
                [item.content for item in effective_results],
                model_key=model_key,
            )
            steps.append(PipelineStep(
                name="faithfulness",
                passed=faithfulness.verdict == "FAITHFUL",
                detail={
                    "score": faithfulness.faithfulness_score,
                    "distortions": faithfulness.distortions,
                },
            ))

        if Config.HALLUCINATION_DETECTION_ENABLED and answer and search_results:
            hallucination = self.hallucination_detector.verify(
                self._answer_body_without_citations(answer),
                [item.content for item in effective_results],
                model_key=model_key,
            )
            steps.append(PipelineStep(
                name="hallucination",
                passed=hallucination.verdict == "PASS",
                detail={
                    "grounded_ratio": hallucination.grounded_ratio,
                    "ungrounded_claims": hallucination.ungrounded_claims,
                },
            ))

        if evidence_used is None:
            answer = self._append_exact_citations(
                answer,
                effective_results,
                max_citations=1 if question_type.category == "extraction" else 2,
            )

        return self._finalize_artifacts(
            question=question,
            search_queries=search_queries,
            search_results=effective_results,
            answer=answer,
            contexts=contexts,
            steps=steps,
            model_key=model_key,
            graph_context_used=graph_context_used,
            gate_reason=gate_reason,
        )

    def _prepare_search(self, question: str, use_query_rewrite: bool) -> List[str]:
        """
        검색 준비: Query Rewrite를 적용하여 검색 쿼리를 생성합니다.

        Args:
            question: 원본 질문
            use_query_rewrite: Query Rewrite 사용 여부

        Returns:
            검색 쿼리 리스트
        """
        search_queries = [question]
        if use_query_rewrite and self.enable_query_rewrite:
            search_queries = self._rewrite_query(question)
            if len(search_queries) > 1:
                print(f"   📝 Query expanded to {len(search_queries)} variants")
        return search_queries

    def answer_question(
        self,
        question: str,
        model_key: Optional[str] = None,
        return_context: bool = False,
        top_k: int = 5,
        use_query_rewrite: bool = True,
        return_artifacts: bool = False,
    ) -> Union[str, Tuple[str, List[str]], AnswerArtifacts]:
        """
        사용자의 질문에 대해 검색 증강 생성(RAG)을 수행합니다.

        [v5.0 통합 파이프라인]
        1. Hook(pre_search) → Agent Routing(모델 선택)
        2. Query Rewrite → 병렬 벡터/웹 검색
        3. Hook(post_search) → 자동 컨텍스트 압축
        4. Guardrails → 에러 자동 복구 답변 생성
        5. Hook(post_generation)

        Args:
            question: 사용자의 질문 문자열.
            model_key: 답변 생성에 사용할 특정 모델 키.
            return_context: True일 경우 답변과 함께 검색된 컨텍스트 리스트를 반환합니다.
            top_k: 검색할 문서의 개수 (기본값: 5).
            use_query_rewrite: Query Rewrite 사용 여부 (기본값: True).
            return_artifacts: True일 경우 AnswerArtifacts 전체를 반환합니다.

        Returns:
            답변 문자열 또는 (답변, 컨텍스트 리스트) 튜플.
        """
        if return_artifacts and return_context:
            raise ValueError("return_artifacts 와 return_context 는 동시에 사용할 수 없습니다.")

        # [v5.0] Agent Routing — 질문 유형에 따른 최적 모델 선택
        routed_model = self._route_model_for_question(question, model_key)

        # [v4.7-fix] Injection 검사를 검색 이전에 수행 (보안 강화)
        if Config.INJECTION_DETECTION_ENABLED:
            injection = self.injection_detector.detect(question, model_key=routed_model)
            if injection.blocked:
                blocked_answer = "입력 내용이 안전하지 않아 요청을 처리할 수 없습니다."
                if return_artifacts:
                    return self._finalize_artifacts(
                        question=question, search_queries=[question],
                        search_results=[], answer=blocked_answer,
                        steps=[PipelineStep(name="prompt_injection", passed=False,
                                            detail={"reason": injection.reason, "score": injection.score})],
                        model_key=routed_model,
                    )
                return (blocked_answer, []) if return_context else blocked_answer

        # [v5.0] Hook: pre_search
        hook_data = self._run_hook(HookEvent.PRE_SEARCH, {"question": question, "model_key": routed_model})
        if hook_data is None:
            blocked_answer = "Hook에 의해 요청이 차단되었습니다."
            if return_artifacts:
                return self._finalize_artifacts(
                    question=question, search_queries=[question],
                    search_results=[], answer=blocked_answer, model_key=routed_model,
                )
            return (blocked_answer, []) if return_context else blocked_answer

        print(f"🔎 Searching for: {question} (top_k={top_k}, hybrid=BM25+Vector+Semantic)")

        # 0. Query Rewrite (선택적) — 공통 로직
        search_queries = self._prepare_search(question, use_query_rewrite)

        # 1. [v5.0] 병렬 검색 (벡터 + 웹)
        search_results, web_context = self._parallel_search(question, search_queries, top_k)

        # [v5.0] Hook: post_search
        self._run_hook(HookEvent.POST_SEARCH, {
            "question": question, "result_count": len(search_results),
            "web_context": web_context,
        })

        # [v5.0] 웹 검색 결과를 검색 결과에 추가
        if web_context:
            search_results.append(SearchResult(
                content=web_context,
                source="web_search",
                score=0.5,
                metadata={"web": True},
            ))

        # [v5.0] 자동 컨텍스트 압축
        if self.context_compactor:
            raw_contexts = self._format_contexts(search_results)
            if self.context_compactor.should_compact(raw_contexts):
                compact_result = self.context_compactor.compact_contexts(raw_contexts, question=question)
                print(f"   🗜️ 컨텍스트 압축: {compact_result.original_token_count} → {compact_result.compacted_token_count} tokens")

        # [v5.0] 에러 자동 복구 래핑
        artifacts = self._run_guardrailed_answer(
            question,
            search_results,
            model_key=routed_model,
            system_prompt=_RAG_SYSTEM_PROMPT,
            search_queries=search_queries,
        )

        # [v5.0] Hook: post_generation
        self._run_hook(HookEvent.POST_GENERATION, {
            "question": question, "answer": artifacts.answer,
        })

        if return_artifacts:
            return artifacts
        if return_context:
            return artifacts.answer, artifacts.contexts
        return artifacts.answer

    # ==================== v4.0: Graph-Enhanced RAG ====================

    def graph_enhanced_answer(
        self,
        question: str,
        model_key: Optional[str] = None,
        return_context: bool = False,
        top_k: int = 5,
        use_query_rewrite: bool = True,
        graph_query_mode: str = "hybrid",
        return_artifacts: bool = False,
    ) -> Union[str, Tuple[str, List[str]], AnswerArtifacts]:
        """
        [v4.0] Graph-Enhanced RAG: 벡터 검색 + Knowledge Graph 결합

        LightRAG의 Dual-Level Retrieval 개념을 적용하여:
        1. Azure AI Search 하이브리드 검색 (기존 벡터+키워드)
        2. Knowledge Graph 맥락 정보 (엔티티/관계)
        3. 두 결과를 결합하여 더 풍부한 컨텍스트로 답변 생성

        Args:
            question: 사용자의 질문 문자열.
            model_key: 답변 생성에 사용할 특정 모델 키.
            return_context: True일 경우 답변과 함께 검색된 컨텍스트 리스트를 반환합니다.
            top_k: 검색할 문서의 개수 (기본값: 5).
            use_query_rewrite: Query Rewrite 사용 여부 (기본값: True).
            graph_query_mode: Graph 검색 모드 (local/global/hybrid/naive).
            return_artifacts: True일 경우 AnswerArtifacts 전체를 반환합니다.

        Returns:
            답변 문자열 또는 (답변, 컨텍스트 리스트) 튜플.
        """
        if return_artifacts and return_context:
            raise ValueError("return_artifacts 와 return_context 는 동시에 사용할 수 없습니다.")

        print(f"🔎 [Graph-Enhanced] Searching for: {question}")

        # === Part 1: 벡터 검색 (공통 로직) ===
        search_queries = self._prepare_search(question, use_query_rewrite)

        vector_results = self._vector_search(question, search_queries, top_k)

        # === Part 2: Knowledge Graph 검색 (v4.0 신규) ===
        graph_context = ""
        if self.graph_manager and graph_query_mode != "naive":
            try:
                from .graph_rag import QueryMode
                mode_map = {
                    "local": QueryMode.LOCAL,
                    "global": QueryMode.GLOBAL,
                    "hybrid": QueryMode.HYBRID,
                    "naive": QueryMode.NAIVE,
                }
                mode = mode_map.get(graph_query_mode, QueryMode.HYBRID)

                graph_result = self.graph_manager.query(
                    query_text=question,
                    mode=mode,
                    top_k=Config.GRAPH_TOP_K,
                )
                graph_context = graph_result.context_text
                if graph_context:
                    print(f"   📊 Graph context: {len(graph_result.entities)} entities, "
                          f"{len(graph_result.relationships)} relationships")

            except Exception as e:
                print(f"   ⚠️ Graph query failed: {e}")

        # === Part 3: 결합된 컨텍스트로 답변 생성 ===
        vector_contexts = self._format_contexts(vector_results)
        vector_context_str = "\n\n".join(vector_contexts)

        if not vector_context_str and not graph_context:
            print("   ⚠️ No relevant documentation found.")
            vector_context_str = "관련된 문서 내용을 찾을 수 없습니다."

        # Graph 컨텍스트가 있으면 추가
        combined_context = vector_context_str
        if graph_context:
            combined_context = (
                f"[문서 검색 결과]\n{vector_context_str}\n\n"
                f"[Knowledge Graph 분석]\n{graph_context}"
            )

        augmented_results = list(vector_results)
        if graph_context:
            augmented_results.insert(0, SearchResult(
                content=graph_context,
                source="knowledge_graph",
                score=1.0,
                metadata={"graph": True},
            ))

        artifacts = self._run_guardrailed_answer(
            question,
            augmented_results,
            model_key=model_key,
            system_prompt=_GRAPH_RAG_SYSTEM_PROMPT,
            search_queries=search_queries,
            graph_context_used=bool(graph_context),
        )

        if return_artifacts:
            return artifacts
        if return_context:
            return artifacts.answer, artifacts.contexts
        return artifacts.answer
