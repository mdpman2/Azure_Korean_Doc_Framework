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

class VectorStore:
    """
    Azure AI Search 기반의 벡터 저장소 관리 클래스
    인덱스 생성, 문서 벡터화(임베딩), 검색 및 증분 업데이트 관리를 담당합니다.
    """
    def __init__(self, index_name=None):
        self.index_name = index_name or Config.SEARCH_INDEX_NAME
        self.index_client = AzureClientFactory.get_search_index_client()
        self.search_client = AzureClientFactory.get_search_client(self.index_name)

        # 임베딩 클라이언트 설정 (LangChain AzureOpenAIEmbeddings 사용)
        from langchain_openai import AzureOpenAIEmbeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=Config.EMBEDDING_DEPLOYMENT,
            openai_api_version=Config.OPENAI_API_VERSION,
            azure_endpoint=Config.OPENAI_ENDPOINT,
            api_key=Config.OPENAI_API_KEY
        )

    def create_index_if_not_exists(self, vector_dim=1536):
        """
        AI Search 인덱스가 존재하지 않으면 생성합니다.
        벡터 검색 및 시맨틱 랭킹(Semantic Ranking) 설정을 포함합니다.
        """
        try:
            self.index_client.get_index(self.index_name)
            print(f"✅ 인덱스 존재: '{self.index_name}'")
        except:
            print(f"🛠️ 인덱스 생성 중: '{self.index_name}'...")

            fields = [
                SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True),
                SimpleField(name="parent_id", type=SearchFieldDataType.String, filterable=True),
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

        # 기존 인덱스가 있더라도 필요한 필드(증분 업데이트용)가 누락되었는지 확인
        self._ensure_incremental_fields()

    def _ensure_incremental_fields(self):
        """기존 인덱스 스키마에 증분 업데이트용 필드(last_modified, content_hash)가 없으면 동적으로 추가합니다."""
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

            if updated:
                self.index_client.create_or_update_index(index)
                print("✅ 인덱스 스키마 업데이트 완료.")
        except Exception as e:
            print(f"⚠️ 인덱스 스키마 업데이트 실패: {e}")

    def upload_documents(self, chunks):
        """
        문서 청크를 벡터화하여 AI Search에 업로드합니다.
        배치 임베딩을 통해 API 호출 횟수를 최적화합니다.
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
            documents.append({
                "chunk_id": f"chunk_{i}",
                "chunk": chunk.page_content,
                "title": chunk.metadata.get("Header 1", "No Title"),
                "parent_id": str(chunk.metadata.get("source", "unknown")),
                "last_modified": str(chunk.metadata.get("last_modified", "")),
                "content_hash": str(chunk.metadata.get("content_hash", "")),
                "text_vector": vector
            })

        # 3. 50개 단위로 나누어 업로드
        for j in range(0, len(documents), 50):
            batch = documents[j:j+50]
            self.search_client.upload_documents(batch)

        print(f"✅ {len(documents)}개 문서 업로드 완료.")

    def is_file_up_to_date(self, file_name, file_mod_time, file_hash=None):
        """
        해당 파일이 이미 최신 상태로 인덱싱되어 있는지 확인합니다.
        해시(내용 검사)를 우선적으로 확인하고, 보조적으로 수정 시간을 확인합니다.
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

    def delete_documents_by_parent_id(self, parent_id):
        """특정 파일(parent_id)에 연관된 모든 청크 데이터를 삭제합니다 (업데이트 전 처리용)."""
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
