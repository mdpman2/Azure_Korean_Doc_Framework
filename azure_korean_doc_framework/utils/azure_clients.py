"""
Azure 서비스 클라이언트 팩토리 모듈

Azure OpenAI, Document Intelligence, AI Search 클라이언트를
생성하고 캐싱하여 불필요한 인스턴스 재생성을 방지합니다.
"""

from openai import AzureOpenAI
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from ..config import Config

class AzureClientFactory:
    """
    Azure 서비스 클라이언트 생성을 담당하는 팩토리 클래스
    클라이언트 캐싱을 통해 불필요한 인스턴스 생성을 방지하고 성능을 최적화합니다.
    """
    _cache = {}

    @classmethod
    def _get_from_cache(cls, key, creator_fn):
        """내부 캐시에서 클라이언트를 가져오거나 없으면 새로 생성하여 캐싱합니다."""
        if key not in cls._cache:
            cls._cache[key] = creator_fn()
        return cls._cache[key]

    @staticmethod
    def get_openai_client(is_advanced=False):
        """
        Azure OpenAI 클라이언트를 반환합니다.

        NOTE: 현재 환경에서는 OPENAI_API_KEY_5/ENDPOINT_5가 기본 엔드포인트와 동일하므로
              모든 요청에 고성능 엔드포인트를 사용합니다.
        """
        # 단일 클라이언트로 통합 (캐시 키 고정)
        cache_key = "openai_unified"

        def create_client():
            # 고성능 엔드포인트 우선, 없으면 기본 엔드포인트 사용
            api_key = Config.OPENAI_API_KEY_5 or Config.OPENAI_API_KEY
            endpoint = Config.OPENAI_ENDPOINT_5 or Config.OPENAI_ENDPOINT

            if not api_key or not endpoint:
                raise ValueError("Azure OpenAI API 키 또는 엔드포인트가 설정되지 않았습니다.")

            return AzureOpenAI(
                api_key=api_key,
                api_version=Config.OPENAI_API_VERSION,
                azure_endpoint=endpoint
            )

        return AzureClientFactory._get_from_cache(cache_key, create_client)

    @staticmethod
    def get_di_client():
        """Azure Document Intelligence 클라이언트를 반환합니다."""
        return AzureClientFactory._get_from_cache("di", lambda: DocumentIntelligenceClient(
            endpoint=Config.DI_ENDPOINT,
            credential=AzureKeyCredential(Config.DI_KEY)
        ))

    @staticmethod
    def get_search_client(index_name=None):
        """공유 및 검색 작업을 위한 Azure AI Search 클라이언트를 반환합니다."""
        name = index_name or Config.SEARCH_INDEX_NAME
        return AzureClientFactory._get_from_cache(f"search_{name}", lambda: SearchClient(
            endpoint=Config.SEARCH_ENDPOINT,
            index_name=name,
            credential=AzureKeyCredential(Config.SEARCH_KEY)
        ))

    @staticmethod
    def get_search_index_client():
        """인덱스 관리(생성, 수정)를 위한 Azure AI Search 인덱스 클라이언트를 반환합니다."""
        return AzureClientFactory._get_from_cache("search_index", lambda: SearchIndexClient(
            endpoint=Config.SEARCH_ENDPOINT,
            credential=AzureKeyCredential(Config.SEARCH_KEY)
        ))
