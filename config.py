import os
from dotenv import load_dotenv

# .env 파일 로드 (환경 변수 오버라이드 허용)
load_dotenv(override=True)

class Config:
    """
    프레임워크의 모든 설정을 관리하는 중앙 구성 클래스입니다.
    환경 변수(.env)에서 설정값을 읽어옵니다.
    """

    # Azure OpenAI (표준 모델용 - GPT-4 등)
    OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    # Azure OpenAI (고성능 모델 전용 엔드포인트 - GPT-5, Claude 등)
    OPENAI_API_KEY_5 = os.getenv("OPEN_AI_KEY_5")
    OPENAI_ENDPOINT_5 = os.getenv("OPEN_AI_ENDPOINT_5")

    # 모델 배포 설정 (추천 값: model-router 사용 시 유연한 모델 교체 가능)
    MODELS = {
        "gpt-4.1": os.getenv("MODEL_DEPLOYMENT_GPT4_1", "gpt-4.1"),
        "gpt-5.2": os.getenv("MODEL_DEPLOYMENT_GPT5_2", "model-router"),
        "claude-opus-4-5": os.getenv("MODEL_DEPLOYMENT_CLAUDE_OPUS", "model-router"),
        "claude-sonnet-4-5": os.getenv("MODEL_DEPLOYMENT_CLAUDE_SONNET", "model-router"),
    }

    # 고성능 엔드포인트를 사용할 모델 리스트
    ADVANCED_MODELS = ["gpt-5.2", "claude-opus-4-5", "claude-sonnet-4-5"]

    # 기본 질문 답변 모델
    DEFAULT_MODEL = "gpt-4.1"

    # 임베딩 모델 배포명
    EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

    # Contextual Retrieval용 GPT-4o 배포명 (기본값: gpt-4.1 - 배포가 없는 경우 대비)
    GPT_4O_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT", os.getenv("MODEL_DEPLOYMENT_GPT4_1", "gpt-4.1"))

    # Azure Document Intelligence 설정
    DI_KEY = os.getenv("AZURE_DI_KEY")
    DI_ENDPOINT = os.getenv("AZURE_DI_ENDPOINT")

    # Azure AI Search 설정
    SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
    SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
    # 현재 활성화된 인덱스명 (환경 변수보다 이 파일의 설정을 우선시하도록 수정)
    SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "")

    @classmethod
    def validate(cls):
        """필수 환경 변수가 설정되어 있는지 확인합니다."""
        required_vars = [
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_DI_KEY",
            "DI_ENDPOINT", # .env에는 AZURE_DI_ENDPOINT로 되어 있을 수 있으므로 체크 필요
            "AZURE_SEARCH_KEY",
            "AZURE_SEARCH_ENDPOINT"
        ]

        # 실제 로직에서는 .env의 변수명과 Config 클래스의 속성 이름을 매칭해서 체크
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
