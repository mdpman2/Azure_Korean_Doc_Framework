import hashlib
from typing import List, Dict, Any, Optional
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch
)
from ..utils.azure_clients import AzureClientFactory
from ..config import Config
from langchain_openai import AzureOpenAIEmbeddings # Moved here from inside __init__

class VectorStore:
    """
    Azure AI Search 기반의 벡터 저장소 관리 클래스.

    인덱스 생성, 문서 벡터화(임베딩), 검색 및 증분 업데이트 관리를 담당합니다.
    """

    def __init__(self, index_name: Optional[str] = None):
        """
        VectorStore 인스턴스를 초기화합니다.

        Args:
            index_name: 사용할 AI Search 인덱스명. 생략 시 Config.SEARCH_INDEX_NAME 사용.
        """
        self.index_name = index_name or Config.SEARCH_INDEX_NAME
        self.index_client = AzureClientFactory.get_search_index_client()
        self.search_client = AzureClientFactory.get_search_client(self.index_name)

        # 임베딩 클라이언트 설정 (LangChain AzureOpenAIEmbeddings 사용)
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=Config.EMBEDDING_DEPLOYMENT,
            openai_api_version=Config.OPENAI_API_VERSION,
            azure_endpoint=Config.OPENAI_ENDPOINT,
            api_key=Config.OPENAI_API_KEY
        )

        # 인덱스가 없으면 자동 생성
        self.create_index_if_not_exists()

        # 인덱스 초기화 및 필드/시맨틱 설정 자동 보정
        self._ensure_incremental_fields()

    def create_index_if_not_exists(self, vector_dim: int = 1536) -> None:
        """
        AI Search 인덱스가 존재하지 않으면 생성합니다.
        벡터 검색 및 시맨틱 랭킹(Semantic Ranking) 설정을 포함합니다.

        Args:
            vector_dim: 벡터 필드의 차원 수 (기본값: 1536 - text-embedding-ada-002 기준).
        """
        try:
            self.index_client.get_index(self.index_name)
            print(f"✅ 인덱스 존재: '{self.index_name}'")
        except Exception: # Changed from bare except
            print(f"🛠️ 인덱스 생성 중: '{self.index_name}'...")

            fields = [
                SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True),
                SimpleField(name="parent_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="last_modified", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="content_hash", type=SearchFieldDataType.String, filterable=True),
                SearchField(name="chunk", type=SearchFieldDataType.String, searchable=True, analyzer_name="ko.microsoft"),
                SearchField(name="title", type=SearchFieldDataType.String, searchable=True),
                SearchField(name="text_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            searchable=True, vector_search_dimensions=vector_dim, vector_search_profile_name="my-vector-profile"),
            ]

            # 벡터 검색 알고리즘 및 프로필 설정
            vector_search = VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name="my-hnsw")],
                profiles=[VectorSearchProfile(name="my-vector-profile", algorithm_configuration_name="my-hnsw")]
            )

            # 시맨틱 검색 설정 (한국어 검색 품질 향상)
            semantic_search = SemanticSearch(
                configurations=[
                    SemanticConfiguration(
                        name="my-semantic-config",
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=None,
                            content_fields=[SemanticField(field_name="chunk")]
                        )
                    )
                ]
            )

            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search,
                semantic_search=semantic_search
            )
            self.index_client.create_index(index)
            print(f"✅ 인덱스 생성 완료: '{self.index_name}'")


    def _ensure_incremental_fields(self) -> None:
        """
        기존 인덱스 스키마에 증분 업데이트용 필드 및 시맨틱 검색 설정이 없으면 동적으로 추가합니다.
        """
        try:
            index = self.index_client.get_index(self.index_name)
            field_names = [f.name for f in index.fields]
            updated = False

            if "last_modified" not in field_names:
                print(f"🛠️ 'last_modified' 필드 추가 중: {self.index_name}")
                index.fields.append(SimpleField(name="last_modified", type=SearchFieldDataType.String, filterable=True))
                updated = True

            if "content_hash" not in field_names:
                print(f"🛠️ 'content_hash' 필드 추가 중: {self.index_name}")
                index.fields.append(SimpleField(name="content_hash", type=SearchFieldDataType.String, filterable=True))
                updated = True

            if "parent_id" in field_names:
                # parent_id를 facetable로 변경 시도 (기본적으로 교체는 인덱스 재생성 필요할 수 있음)
                pass

            # 시맨틱 검색 설정 확인 및 추가
            if not index.semantic_search or not any(c.name == "my-semantic-config" for c in index.semantic_search.configurations):
                print(f"🛠️ 'my-semantic-config' 시맨틱 검색 설정 추가 중: {self.index_name}")
                index.semantic_search = SemanticSearch(
                    configurations=[
                        SemanticConfiguration(
                            name="my-semantic-config",
                            prioritized_fields=SemanticPrioritizedFields(
                                title_field=None,
                                content_fields=[SemanticField(field_name="chunk")]
                            )
                        )
                    ]
                )
                updated = True

            if updated:
                self.index_client.create_or_update_index(index)
                print("✅ 인덱스 스키마 및 설정 업데이트 완료.")
        except Exception as e:
            print(f"⚠️ 인덱스 스키마 업데이트 확인 실패 (무시 가능): {e}") # Changed error message

    def upload_documents(self, chunks: List[Any]) -> None:
        """
        문서 청크를 벡터화하여 AI Search에 업로드합니다.
        배치 임베딩을 통해 API 호출 횟수를 최적화하며, 고유 ID 생성을 통해 충돌을 방지합니다.

        Args:
            chunks: 업로드할 LangChain Document 객체 리스트.
        """
        if not chunks:
            return

        print(f"📡 {len(chunks)}개 청크 배치 임베딩 및 업로드 중... 인덱스: '{self.index_name}'")

        # 1. 문서 텍스트 추출 및 한꺼번에 임베딩 (성능 최적화 핵심)
        texts = [chunk.page_content for chunk in chunks]
        vectors = self.embeddings.embed_documents(texts)

        # 2. AI Search 업로드용 데이터 구성
        documents = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            # 파일명과 인덱스를 조합하여 고유 ID 생성 (중복 방지)
            parent_id_str = str(chunk.metadata.get("source", "unknown"))
            # Encode parent_id to handle non-ascii chars safely in hash
            parent_hash = hashlib.md5(parent_id_str.encode('utf-8')).hexdigest()[:10]

            documents.append({
                "chunk_id": f"c_{parent_hash}_{i}", # Short unique prefix
                "chunk": chunk.page_content,
                "title": chunk.metadata.get("Header 1", "No Title"),
                "parent_id": parent_id_str,
                "last_modified": str(chunk.metadata.get("last_modified", "")),
                "content_hash": str(chunk.metadata.get("content_hash", "")),
                "text_vector": vector
            })

        # 3. 50개 단위로 나누어 업로드 (배치 처리)
        for j in range(0, len(documents), 50):
            batch = documents[j:j+50]
            self.search_client.upload_documents(batch)

        print(f"✅ {len(documents)}개 업로드 완료.")

    def is_file_up_to_date(self, file_name: str, file_mod_time: float, file_hash: Optional[str] = None) -> bool:
        """
        해당 파일이 이미 최신 상태로 인덱싱되어 있는지 확인합니다.
        해시(내용 검사)를 우선적으로 확인하고, 보조적으로 수정 시간을 확인합니다.

        Args:
            file_name: 확인 대상 파일명.
            file_mod_time: 파일의 마지막 수정 시간.
            file_hash: 파일의 SHA256 해시값 (옵션).

        Returns:
            최신 상태이면 True, 아니면 False.
        """
        try:
            results = self.search_client.search(
                search_text="*",
                filter=f"parent_id eq '{file_name}'",
                select=["last_modified", "content_hash"],
                top=1
            )
            for r in results:
                # 1. 파일 내용 해시 비교 (가장 정확한 방법)
                if file_hash:
                    stored_hash = r.get("content_hash")
                    if stored_hash == file_hash:
                        print(f"⏭️ 중복 스킵: '{file_name}' (내용이 동일함)")
                        return True

                # 2. 수정 시간 비교
                stored_time = r.get("last_modified")
                if stored_time and float(stored_time) >= file_mod_time:
                    print(f"⏭️ 중복 스킵: '{file_name}' (날짜가 최신임)")
                    return True
            return False
        except Exception:
            return False

    def delete_documents_by_parent_id(self, parent_id: str) -> None:
        """
        특정 파일(parent_id)에 연관된 모든 청크 데이터를 삭제합니다 (업데이트 전 처리용).

        Args:
            parent_id: 삭제할 부모 문서 ID(파일명).
        """
        try:
            results = self.search_client.search(
                search_text="*",
                filter=f"parent_id eq '{parent_id}'",
                select=["chunk_id"]
            )

            ids_to_delete = [{"chunk_id": r["chunk_id"]} for r in results]
            if ids_to_delete:
                print(f"🗑️ 기존 데이터 삭제 중: '{parent_id}' ({len(ids_to_delete)}개 청크)")
                for i in range(0, len(ids_to_delete), 100):
                    batch = ids_to_delete[i:i+100]
                    self.search_client.delete_documents(batch)
                print("✅ 삭제 완료.")
        except Exception as e:
            print(f"⚠️ 데이터 삭제 오류: {e}")
