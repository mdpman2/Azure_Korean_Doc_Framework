import os
from dotenv import load_dotenv

# .env 파일 로드 (환경 변수 오버라이드 허용)
load_dotenv(override=True)

class Config:
    """
    프레임워크의 모든 설정을 관리하는 중앙 구성 클래스입니다.
    환경 변수(.env)에서 설정값을 읽어옵니다.

    [2026-07 v4.0 업데이트]
    - Graph RAG 지원 (LightRAG 기반 Knowledge Graph)
    - 구조화 엔티티 추출 (LangExtract 기반)
    - GPT-5.2 기본 모델 (Vision + Reasoning 통합)
    - API Version: 2024-12-01-preview
    - Structured Outputs / max_completion_tokens 지원
    - Document Intelligence API: 2024-11-30 (GA)
    """

    # =================================================================
    # Azure OpenAI 설정 (GPT-5.2, Claude 등 최신 모델 지원)
    # =================================================================

    # 기본 엔드포인트 (GPT-5.2 지원)
    OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

    # API 버전 (환경 변수 우선, 기본값: 2024-12-01-preview)
    OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    # 고성능 모델 전용 엔드포인트 (GPT-5.2, Claude 등)
    OPENAI_API_KEY_5 = os.getenv("OPEN_AI_KEY_5")
    OPENAI_ENDPOINT_5 = os.getenv("OPEN_AI_ENDPOINT_5")

    # =================================================================
    # 모델 배포 설정 (2026년 최신 모델)
    # NOTE: Azure AI Foundry에서 model-router를 통해 GPT-5.x 접근
    # =================================================================
    MODELS = {
        # GPT-5.x 시리즈 (model-router를 통해 최신 모델 자동 라우팅)
        "gpt-5.2": os.getenv("MODEL_DEPLOYMENT_GPT5_2", "model-router"),
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
        "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-5-mini",
        "o3", "o4-mini",
        "claude-opus-4-5", "claude-sonnet-4-5"
    ])

    # Reasoning 지원 모델 (reasoning_effort 파라미터 사용 가능)
    REASONING_MODELS = frozenset([
        "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-5-mini",
        "o3", "o4-mini"
    ])

    # Structured Outputs 지원 모델
    STRUCTURED_OUTPUT_MODELS = frozenset([
        "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-5-mini",
        "gpt-4.1", "o3", "o4-mini"
    ])

    # =================================================================
    # 기본 모델 설정
    # =================================================================

    # 기본 질문 답변 모델 (GPT-5.2로 업그레이드)
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-5.2")

    # Vision/이미지 분석용 모델 (GPT-5.2 Vision 사용)
    VISION_MODEL = os.getenv("VISION_MODEL", "gpt-5.2")

    # 문서 파싱용 모델
    PARSING_MODEL = os.getenv("PARSING_MODEL", "gpt-5.2")

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
    SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
    SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
    # 현재 활성화된 인덱스명 (환경 변수보다 이 파일의 설정을 우선시하도록 수정)
    SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "")

    # =================================================================
    # Graph RAG 설정 (v4.0 신규 - LightRAG 기반)
    # =================================================================

    # Graph RAG 활성화 여부
    GRAPH_RAG_ENABLED = os.getenv("GRAPH_RAG_ENABLED", "true").lower() == "true"

    # Knowledge Graph 저장 경로
    GRAPH_STORAGE_PATH = os.getenv("GRAPH_STORAGE_PATH", "output/knowledge_graph.json")

    # 엔티티 추출 배치 크기
    GRAPH_ENTITY_BATCH_SIZE = int(os.getenv("GRAPH_ENTITY_BATCH_SIZE", "5"))

    # 그래프 검색 모드 (local, global, hybrid, naive)
    GRAPH_QUERY_MODE = os.getenv("GRAPH_QUERY_MODE", "hybrid")

    # 그래프 검색 top_k
    GRAPH_TOP_K = int(os.getenv("GRAPH_TOP_K", "10"))

    # =================================================================
    # 구조화 추출 설정 (v4.0 신규 - LangExtract 기반)
    # =================================================================

    # 추출 패스 수 (Multi-Pass Extraction, 높을수록 Recall 향상)
    EXTRACTION_PASSES = int(os.getenv("EXTRACTION_PASSES", "1"))

    # 추출용 청크 최대 문자 수
    EXTRACTION_MAX_CHUNK_CHARS = int(os.getenv("EXTRACTION_MAX_CHUNK_CHARS", "3000"))

    # 병렬 추출 워커 수
    EXTRACTION_MAX_WORKERS = int(os.getenv("EXTRACTION_MAX_WORKERS", "4"))

    @classmethod
    def validate(cls):
        """필수 환경 변수가 설정되어 있는지 확인합니다."""
        missing = []
        if not cls.OPENAI_API_KEY: missing.append("AZURE_OPENAI_API_KEY")
        if not cls.OPENAI_ENDPOINT: missing.append("AZURE_OPENAI_ENDPOINT")
        if not cls.DI_KEY: missing.append("AZURE_DI_KEY")
        if not cls.DI_ENDPOINT: missing.append("AZURE_DI_ENDPOINT")
        if not cls.SEARCH_KEY: missing.append("AZURE_SEARCH_KEY")
        if not cls.SEARCH_ENDPOINT: missing.append("AZURE_SEARCH_ENDPOINT")

        if missing:
            error_msg = "\n".join([f"❌ 환경 변수 누락: {var}" for var in missing])
            raise EnvironmentError(f"\n필수 설정이 누락되었습니다. .env 파일을 확인해주세요:\n{error_msg}")

        print(f"✅ 환경 변수 설정 확인 완료. (현재 인덱스: '{cls.SEARCH_INDEX_NAME}')")
