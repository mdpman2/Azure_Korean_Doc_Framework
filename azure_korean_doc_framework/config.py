import os
from dotenv import load_dotenv

# .env 파일 로드 (환경 변수 오버라이드 허용)
load_dotenv(override=True)

class Config:
    """
    프레임워크의 모든 설정을 관리하는 중앙 구성 클래스입니다.
    환경 변수(.env)에서 설정값을 읽어옵니다.

    [2026-02 v4.1 업데이트]
    - Contextual Retrieval (Anthropic 방식) — 청크별 LLM 맥락 생성
    - Hybrid Search: BM25 키워드 + Vector 유사성 + Semantic Ranking 결합
    - Contextual BM25 / Contextual Embeddings 지원

    [2026-03 v4.3]
    - GPT-5.4 기본 모델로 통일
    - `AZURE_OPENAI_*` 와 `OPEN_AI_*` 별칭을 함께 지원
    - 고성능 endpoint/key는 기본 endpoint/key와 동일 값 사용 가능

    [2026-04 v4.7 — EdgeQuake 참조 강화]
    - GRAPH_GLEANING_PASSES: Multi-Pass 엔티티 추출 횟수 (Gleaning)
    - GRAPH_MIX_WEIGHT: Mix Query Mode 그래프 가중치 (0.0~1.0)
    - GRAPH_INJECTION_FILE: 도메인 용어집/동의어 주입 파일 경로
    - GRAPH_QUERY_MODE에 mix/bypass 모드 추가

    [2026-04 v4.6]
    - doctor/status/session CLI가 그대로 참조하는 운영 설정 집합
    - Azure AI Search 필드명은 런타임 스키마 조회 결과로 자동 보정될 수 있음

    [2026-02 v4.0]
    - Graph RAG 지원 (LightRAG 기반 Knowledge Graph)
    - 구조화 엔티티 추출 (LangExtract 기반)
    - GPT-5.x 기본 모델 (Vision + Reasoning 통합)
    - API Version: 2024-12-01-preview
    - Structured Outputs / max_completion_tokens 지원
    - Document Intelligence API: 2024-11-30 (GA)
    """

    # =================================================================
    # Azure OpenAI 설정 (GPT-5.4, Claude 등 최신 모델 지원)
    # =================================================================

    # 기본 엔드포인트 (GPT-5.4 지원)
    OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY_5")
    OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPEN_AI_ENDPOINT_5")

    # API 버전 (환경 변수 별칭 우선, 기본값: 2024-12-01-preview)
    OPENAI_API_VERSION = (
        os.getenv("AZURE_OPENAI_API_VERSION")
        or os.getenv("AZURE_OPENAI_API_VER")
        or os.getenv("OPENAI_API_VER")
        or "2024-12-01-preview"
    )

    # 고성능 모델 전용 엔드포인트 (기본 endpoint/key와 동일 값 사용 가능)
    OPENAI_API_KEY_5 = os.getenv("OPEN_AI_KEY_5") or OPENAI_API_KEY
    OPENAI_ENDPOINT_5 = os.getenv("OPEN_AI_ENDPOINT_5") or OPENAI_ENDPOINT

    # =================================================================
    # 모델 배포 설정 (2026년 최신 모델)
    # NOTE: Azure AI Foundry에서 model-router를 통해 GPT-5.x 접근
    # =================================================================
    MODELS = {
        # GPT-5.x 시리즈 (model-router를 통해 최신 모델 자동 라우팅)
        "gpt-5.4": os.getenv("MODEL_DEPLOYMENT_GPT5_4", os.getenv("MODEL_DEPLOYMENT_GPT5_2", "model-router")),
        "gpt-5.2": os.getenv("MODEL_DEPLOYMENT_GPT5_2", os.getenv("MODEL_DEPLOYMENT_GPT5_4", "model-router")),
        "gpt-5.1": os.getenv("MODEL_DEPLOYMENT_GPT5_1", "model-router"),
        "gpt-5": os.getenv("MODEL_DEPLOYMENT_GPT5", "model-router"),
        "gpt-5-mini": os.getenv("MODEL_DEPLOYMENT_GPT5_MINI", "model-router"),

        # GPT-4.x 시리즈 (직접 배포)
        "gpt-4.1": os.getenv("MODEL_DEPLOYMENT_GPT4_1", "gpt-4.1"),

        # o-시리즈 추론 모델 (model-router 사용)
        "o3": os.getenv("MODEL_DEPLOYMENT_O3", "model-router"),
        "o4-mini": os.getenv("MODEL_DEPLOYMENT_O4_MINI", "model-router"),

        # Claude 모델 (model-router 사용)
        "claude-opus-4-5": os.getenv("MODEL_DEPLOYMENT_CLAUDE_OPUS", "model-router"),
        "claude-sonnet-4-5": os.getenv("MODEL_DEPLOYMENT_CLAUDE_SONNET", "model-router"),
    }

    # 고성능 엔드포인트를 사용할 모델 리스트 (frozenset으로 멤버쉽 검사 O(1))
    ADVANCED_MODELS = frozenset([
        "gpt-5.4", "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-5-mini",
        "o3", "o4-mini",
        "claude-opus-4-5", "claude-sonnet-4-5"
    ])

    # Reasoning 지원 모델 (reasoning_effort 파라미터 사용 가능)
    REASONING_MODELS = frozenset([
        "gpt-5.4", "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-5-mini",
        "o3", "o4-mini"
    ])

    # Structured Outputs 지원 모델
    STRUCTURED_OUTPUT_MODELS = frozenset([
        "gpt-5.4", "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-5-mini",
        "gpt-4.1", "o3", "o4-mini"
    ])

    # =================================================================
    # 기본 모델 설정
    # =================================================================

    # 기본 질문 답변 모델
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-5.4")

    # Vision/이미지 분석용 모델
    VISION_MODEL = os.getenv("VISION_MODEL", "gpt-5.4")

    # 문서 파싱용 모델
    PARSING_MODEL = os.getenv("PARSING_MODEL", "gpt-5.4")

    # =================================================================
    # 임베딩 설정
    # =================================================================

    # 임베딩 모델 배포명 (text-embedding-3-small 또는 text-embedding-3-large)
    EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

    # 임베딩 차원 (text-embedding-3-small: 1536, large: 3072)
    EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

    # Azure Document Intelligence 설정
    DI_KEY = os.getenv("AZURE_DI_KEY")
    DI_ENDPOINT = os.getenv("AZURE_DI_ENDPOINT")

    # Azure AI Search 설정
    # 기존 인덱스를 재사용할 수 있도록 필드명과 시맨틱 설정명을 모두 환경 변수로 매핑합니다.
    SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY") or os.getenv("AZURE_SEARCH_API_KEY")
    SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT") or os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
    # 현재 활성화된 인덱스명
    SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME") or os.getenv("SEARCH_INDEX_NAME", "")
    SEARCH_ID_FIELD = os.getenv("AZURE_SEARCH_ID_FIELD", "chunk_id")
    SEARCH_CONTENT_FIELD = os.getenv("AZURE_SEARCH_CONTENT_FIELD", "chunk")
    SEARCH_ORIGINAL_CONTENT_FIELD = os.getenv("AZURE_SEARCH_ORIGINAL_CONTENT_FIELD", "original_chunk")
    SEARCH_VECTOR_FIELD = os.getenv("AZURE_SEARCH_VECTOR_FIELD", "text_vector")
    SEARCH_TITLE_FIELD = os.getenv("AZURE_SEARCH_TITLE_FIELD", "title")
    SEARCH_PARENT_FIELD = os.getenv("AZURE_SEARCH_PARENT_FIELD", "parent_id")
    # 사용자에게 보여줄 출처 라벨. 외부 인덱스에서는 title 같은 사람이 읽기 쉬운 필드를 권장합니다.
    SEARCH_SOURCE_FIELD = os.getenv("AZURE_SEARCH_SOURCE_FIELD", "parent_id")
    SEARCH_CITATION_FIELD = os.getenv("AZURE_SEARCH_CITATION_FIELD", "citation")
    SEARCH_BOUNDING_BOX_FIELD = os.getenv("AZURE_SEARCH_BOUNDING_BOX_FIELD", "bounding_box_json")
    SEARCH_SOURCE_REGIONS_FIELD = os.getenv("AZURE_SEARCH_SOURCE_REGIONS_FIELD", "source_regions_json")
    SEARCH_SEMANTIC_CONFIG = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG", "my-semantic-config")

    # =================================================================
    # Graph RAG 설정 (v4.0 신규 - LightRAG 기반)
    # =================================================================

    # Graph RAG 활성화 여부
    GRAPH_RAG_ENABLED = os.getenv("GRAPH_RAG_ENABLED", "true").lower() == "true"

    # Knowledge Graph 저장 경로
    GRAPH_STORAGE_PATH = os.getenv("GRAPH_STORAGE_PATH", "output/knowledge_graph.json")

    # 엔티티 추출 배치 크기
    GRAPH_ENTITY_BATCH_SIZE = int(os.getenv("GRAPH_ENTITY_BATCH_SIZE", "5"))

    # 그래프 검색 모드 (local, global, hybrid, naive, mix, bypass)
    GRAPH_QUERY_MODE = os.getenv("GRAPH_QUERY_MODE", "hybrid")

    # 그래프 검색 top_k
    GRAPH_TOP_K = int(os.getenv("GRAPH_TOP_K", "10"))

    # =================================================================
    # [v4.7] EdgeQuake 강화 설정 (Gleaning, Normalization, Community, Mix, Injection)
    # =================================================================

    # Gleaning: Multi-Pass 추출 횟수 (0=비활성, 1=1회 추가, 2=2회 추가)
    # EdgeQuake 벤치마크에 따르면 1회 Gleaning으로 엔티티 15-25% 추가 포착
    GRAPH_GLEANING_PASSES = int(os.getenv("GRAPH_GLEANING_PASSES", "1"))

    # Mix Query Mode: 그래프 결과 가중치 (0.0~1.0, 기본 0.4)
    # 0.0 = 벡터 결과만, 0.5 = 반반, 1.0 = 그래프 결과만
    GRAPH_MIX_WEIGHT = float(os.getenv("GRAPH_MIX_WEIGHT", "0.4"))
    # [v4.7-fix] 범위 검증: 0.0 ~ 1.0
    if not (0.0 <= GRAPH_MIX_WEIGHT <= 1.0):
        print(f"⚠️ GRAPH_MIX_WEIGHT={GRAPH_MIX_WEIGHT}이 범위 밖(반드시 0.0~1.0)입니다. 기본값 0.4로 설정합니다.")
        GRAPH_MIX_WEIGHT = 0.4

    # Knowledge Injection 파일 경로 (선택사항)
    # 형식: 각 줄 "용어 (동의어1, 동의어2): 정의"
    GRAPH_INJECTION_FILE = os.getenv("GRAPH_INJECTION_FILE", "")

    # =================================================================
    # Contextual Retrieval 설정 (v4.1 신규 - Anthropic 방식)
    # 각 청크에 문서 전체 맥락을 LLM으로 생성하여 추가
    # BM25 키워드 검색 + 벡터 유사성 검색 결합 시 검색 실패율 49% 감소
    # =================================================================

    # Contextual Retrieval 활성화 여부
    CONTEXTUAL_RETRIEVAL_ENABLED = os.getenv("CONTEXTUAL_RETRIEVAL_ENABLED", "true").lower() == "true"

    # 맥락 생성에 사용할 모델 (비용 효율적인 모델 권장)
    CONTEXTUAL_RETRIEVAL_MODEL = os.getenv("CONTEXTUAL_RETRIEVAL_MODEL", "gpt-5.4")

    # 맥락 텍스트 최대 토큰 수 (50-150 토큰 권장)
    CONTEXTUAL_RETRIEVAL_MAX_TOKENS = int(os.getenv("CONTEXTUAL_RETRIEVAL_MAX_TOKENS", "150"))

    # 맥락 생성 배치 크기 (동시 처리할 청크 수)
    CONTEXTUAL_RETRIEVAL_BATCH_SIZE = int(os.getenv("CONTEXTUAL_RETRIEVAL_BATCH_SIZE", "5"))

    # Agent 질의 확장(Query Rewrite) 기본 활성화 여부
    QUERY_REWRITE_ENABLED = os.getenv("QUERY_REWRITE_ENABLED", "true").lower() == "true"

    # 답변 산출물에 운영 진단 정보 포함 여부
    ANSWER_DIAGNOSTICS_ENABLED = os.getenv("ANSWER_DIAGNOSTICS_ENABLED", "true").lower() == "true"

    # =================================================================
    # 구조화 추출 설정 (v4.0 신규 - LangExtract 기반)
    # =================================================================

    # 추출 패스 수 (Multi-Pass Extraction, 높을수록 Recall 향상)
    EXTRACTION_PASSES = int(os.getenv("EXTRACTION_PASSES", "1"))

    # 추출용 청크 최대 문자 수
    EXTRACTION_MAX_CHUNK_CHARS = int(os.getenv("EXTRACTION_MAX_CHUNK_CHARS", "3000"))

    # 병렬 추출 워커 수
    EXTRACTION_MAX_WORKERS = int(os.getenv("EXTRACTION_MAX_WORKERS", "4"))

    # =================================================================
    # Retrieval / Guardrails / Evaluation settings
    # =================================================================
    RETRIEVAL_GATE_ENABLED = os.getenv("RETRIEVAL_GATE_ENABLED", "true").lower() == "true"
    RETRIEVAL_GATE_MIN_TOP_SCORE = float(os.getenv("RETRIEVAL_GATE_MIN_TOP_SCORE", "0.15"))
    RETRIEVAL_GATE_MIN_DOC_COUNT = int(os.getenv("RETRIEVAL_GATE_MIN_DOC_COUNT", "1"))
    RETRIEVAL_GATE_MIN_DOC_SCORE = float(os.getenv("RETRIEVAL_GATE_MIN_DOC_SCORE", "0.05"))
    RETRIEVAL_GATE_SOFT_MODE = os.getenv("RETRIEVAL_GATE_SOFT_MODE", "true").lower() == "true"
    RETRIEVAL_GATE_NOT_FOUND_MESSAGE = os.getenv(
        "RETRIEVAL_GATE_NOT_FOUND_MESSAGE",
        "관련 문서를 충분히 찾지 못했습니다. 다른 키워드로 다시 질문해 주세요.",
    )

    EXACT_CITATION_ENABLED = os.getenv("EXACT_CITATION_ENABLED", "true").lower() == "true"
    NUMERIC_VERIFICATION_ENABLED = os.getenv("NUMERIC_VERIFICATION_ENABLED", "true").lower() == "true"
    PII_DETECTION_ENABLED = os.getenv("PII_DETECTION_ENABLED", "true").lower() == "true"
    INJECTION_DETECTION_ENABLED = os.getenv("INJECTION_DETECTION_ENABLED", "true").lower() == "true"
    FAITHFULNESS_ENABLED = os.getenv("FAITHFULNESS_ENABLED", "true").lower() == "true"
    FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.85"))
    HALLUCINATION_DETECTION_ENABLED = os.getenv("HALLUCINATION_DETECTION_ENABLED", "true").lower() == "true"
    HALLUCINATION_THRESHOLD = float(os.getenv("HALLUCINATION_THRESHOLD", "0.8"))

    EVALUATION_JUDGE_MODEL = os.getenv("EVALUATION_JUDGE_MODEL", "gpt-5.4")

    # =================================================================
    # [v5.0] 스트리밍 설정
    # =================================================================
    STREAMING_ENABLED = os.getenv("STREAMING_ENABLED", "true").lower() == "true"

    # =================================================================
    # [v5.0] 자동 압축 설정
    # =================================================================
    AUTO_COMPACT_ENABLED = os.getenv("AUTO_COMPACT_ENABLED", "true").lower() == "true"
    AUTO_COMPACT_MAX_CONTEXT_TOKENS = int(os.getenv("AUTO_COMPACT_MAX_CONTEXT_TOKENS", "120000"))
    AUTO_COMPACT_THRESHOLD_RATIO = float(os.getenv("AUTO_COMPACT_THRESHOLD_RATIO", "0.85"))

    # =================================================================
    # [v5.0] 웹 검색/수집 설정
    # =================================================================
    WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "false").lower() == "true"
    BING_API_KEY = os.getenv("BING_API_KEY", "")
    WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "3"))
    WEB_FETCH_MAX_CHARS = int(os.getenv("WEB_FETCH_MAX_CHARS", "8000"))

    # =================================================================
    # [v5.0] 에러 자동 복구 설정
    # =================================================================
    ERROR_RECOVERY_ENABLED = os.getenv("ERROR_RECOVERY_ENABLED", "true").lower() == "true"
    ERROR_RECOVERY_MAX_RETRIES = int(os.getenv("ERROR_RECOVERY_MAX_RETRIES", "3"))
    ERROR_RECOVERY_BASE_DELAY = float(os.getenv("ERROR_RECOVERY_BASE_DELAY", "1.0"))
    ERROR_RECOVERY_FALLBACK_MODELS = os.getenv(
        "ERROR_RECOVERY_FALLBACK_MODELS", "gpt-5.2,gpt-4.1"
    ).split(",")

    # =================================================================
    # [v5.0] 서브에이전트 위임 설정
    # =================================================================
    SUB_AGENT_ENABLED = os.getenv("SUB_AGENT_ENABLED", "true").lower() == "true"
    SUB_AGENT_MAX_WORKERS = int(os.getenv("SUB_AGENT_MAX_WORKERS", "3"))
    SUB_AGENT_TIMEOUT = int(os.getenv("SUB_AGENT_TIMEOUT", "60"))

    # =================================================================
    # [v5.0] Agent Routing (질문 유형별 모델 분기)
    # =================================================================
    AGENT_ROUTING_ENABLED = os.getenv("AGENT_ROUTING_ENABLED", "true").lower() == "true"
    # 질문 유형 → 모델 매핑 (JSON 형식 또는 기본값)
    AGENT_ROUTING_MAP = {
        "regulatory": os.getenv("AGENT_ROUTING_REGULATORY", "gpt-5.4"),
        "extraction": os.getenv("AGENT_ROUTING_EXTRACTION", "gpt-4.1"),
        "explanatory": os.getenv("AGENT_ROUTING_EXPLANATORY", "gpt-5.4"),
    }

    # =================================================================
    # [v5.0] Hook 시스템 설정
    # =================================================================
    HOOKS_ENABLED = os.getenv("HOOKS_ENABLED", "true").lower() == "true"

    # =================================================================
    # [v5.0] 병렬 도구 실행 설정
    # =================================================================
    PARALLEL_TOOL_ENABLED = os.getenv("PARALLEL_TOOL_ENABLED", "true").lower() == "true"
    PARALLEL_TOOL_MAX_WORKERS = int(os.getenv("PARALLEL_TOOL_MAX_WORKERS", "5"))

    @classmethod
    def get_openai_credentials(cls, prefer_advanced: bool = True):
        """환경 변수 별칭을 흡수한 OpenAI 인증 정보와 API 버전을 반환"""
        api_key = cls.OPENAI_API_KEY_5 if prefer_advanced else cls.OPENAI_API_KEY
        endpoint = cls.OPENAI_ENDPOINT_5 if prefer_advanced else cls.OPENAI_ENDPOINT
        api_key = api_key or cls.OPENAI_API_KEY or cls.OPENAI_API_KEY_5
        endpoint = endpoint or cls.OPENAI_ENDPOINT or cls.OPENAI_ENDPOINT_5
        return api_key, endpoint, cls.OPENAI_API_VERSION

    @classmethod
    def get_model_deployment(cls, model_name: str) -> str:
        """모델명에 대응하는 실제 deployment 이름 반환"""
        return cls.MODELS.get(model_name, model_name)

    @classmethod
    def validate(
        cls,
        *,
        require_openai: bool = True,
        require_search: bool = True,
        require_di: bool = True,
    ):
        """실행 모드에 맞는 필수 환경 변수가 설정되어 있는지 확인합니다."""
        missing = []
        if require_openai:
            api_key, endpoint, _ = cls.get_openai_credentials(prefer_advanced=True)
            if not api_key:
                missing.append("AZURE_OPENAI_API_KEY 또는 OPEN_AI_KEY_5")
            if not endpoint:
                missing.append("AZURE_OPENAI_ENDPOINT 또는 OPEN_AI_ENDPOINT_5")

        if require_di:
            if not cls.DI_KEY:
                missing.append("AZURE_DI_KEY")
            if not cls.DI_ENDPOINT:
                missing.append("AZURE_DI_ENDPOINT")

        if require_search:
            if not cls.SEARCH_KEY:
                missing.append("AZURE_SEARCH_KEY")
            if not cls.SEARCH_ENDPOINT:
                missing.append("AZURE_SEARCH_ENDPOINT")
            if not cls.SEARCH_INDEX_NAME:
                missing.append("AZURE_SEARCH_INDEX_NAME")

        if missing:
            error_msg = "\n".join([f"❌ 환경 변수 누락: {var}" for var in missing])
            raise EnvironmentError(f"\n필수 설정이 누락되었습니다. .env 파일을 확인해주세요:\n{error_msg}")

        print(f"✅ 환경 변수 설정 확인 완료. (현재 인덱스: '{cls.SEARCH_INDEX_NAME}')")
