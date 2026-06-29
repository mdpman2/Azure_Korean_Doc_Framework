"""Azure 서비스 클라이언트 팩토리 모듈.

공용 Azure SDK 클라이언트를 지연 생성하고 캐싱합니다.
Azure OpenAI는 동기/비동기 클라이언트를 분리 캐싱해 반복 생성 비용을 줄입니다.
"""

from openai import AsyncAzureOpenAI, AzureOpenAI
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from ..config import Config

class AzureClientFactory:
    """
    Azure 서비스 클라이언트 생성을 담당하는 팩토리 클래스.

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
    def _build_openai_client(async_client: bool, is_advanced: bool):
        api_key, endpoint, api_version = Config.get_openai_credentials(prefer_advanced=is_advanced)
        if not api_key or not endpoint:
            raise ValueError("Azure OpenAI API 키 또는 엔드포인트가 설정되지 않았습니다.")

        client_class = AsyncAzureOpenAI if async_client else AzureOpenAI
        return client_class(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )

    @staticmethod
    def get_openai_client(is_advanced=False):
        """동기 Azure OpenAI 클라이언트를 반환합니다."""
        cache_key = f"openai_sync_{'advanced' if is_advanced else 'standard'}"
        return AzureClientFactory._get_from_cache(
            cache_key,
            lambda: AzureClientFactory._build_openai_client(async_client=False, is_advanced=is_advanced),
        )

    @staticmethod
    def get_async_openai_client(is_advanced=False):
        """비동기 Azure OpenAI 클라이언트를 반환합니다."""
        cache_key = f"openai_async_{'advanced' if is_advanced else 'standard'}"
        return AzureClientFactory._get_from_cache(
            cache_key,
            lambda: AzureClientFactory._build_openai_client(async_client=True, is_advanced=is_advanced),
        )

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
