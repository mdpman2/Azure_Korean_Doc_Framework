import hashlib
import json
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
from ..utils.search_schema import apply_search_runtime_mapping
from ..config import Config
from .schema import Document


class VectorStore:
    """
    Azure AI Search 기반의 벡터 저장소 관리 클래스.

    인덱스 생성, 문서 벡터화(임베딩), 검색 및 증분 업데이트 관리를 담당합니다.
    기본 스키마를 생성할 수 있으며, 기존 인덱스도 Config.SEARCH_* 필드 매핑으로 재사용할 수 있습니다.

    [v4.1 업데이트 - Contextual Retrieval]
    - SEARCH_CONTENT_FIELD: 맥락 포함 텍스트 (BM25 키워드 + 벡터 임베딩 대상)
    - SEARCH_ORIGINAL_CONTENT_FIELD: 원본 텍스트 (답변 생성 시 사용)
    - SEARCH_VECTOR_FIELD: 맥락 포함 텍스트 기반 Contextual Embeddings
    """

    def __init__(self, index_name: Optional[str] = None):
        """
        VectorStore 인스턴스를 초기화합니다.

        Args:
            index_name: 사용할 AI Search 인덱스명. 생략 시 Config.SEARCH_INDEX_NAME 사용.
        """
        self.index_name = index_name or Config.SEARCH_INDEX_NAME
        apply_search_runtime_mapping(self.index_name)
        self.index_client = AzureClientFactory.get_search_index_client()
        self.search_client = AzureClientFactory.get_search_client(self.index_name)

        # 임베딩 클라이언트 설정 (openai native SDK 사용)
        self.openai_client = AzureClientFactory.get_openai_client()

        # 인덱스가 없으면 자동 생성
        self.create_index_if_not_exists()

        # 인덱스 초기화 및 필드/시맨틱 설정 자동 보정
        self._ensure_incremental_fields()

    def _semantic_configuration(self) -> SemanticConfiguration:
        return SemanticConfiguration(
            name=Config.SEARCH_SEMANTIC_CONFIG,
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name=Config.SEARCH_TITLE_FIELD),
                content_fields=[SemanticField(field_name=Config.SEARCH_CONTENT_FIELD)],
            ),
        )

    def _build_document_id(self, parent_id: str, index: int) -> str:
        parent_hash = hashlib.md5(parent_id.encode('utf-8')).hexdigest()[:10]
        return f"c_{parent_hash}_{index}"

    def _base_document_fields(self, vector_dim: int):
        return [
            SimpleField(name=Config.SEARCH_ID_FIELD, type=SearchFieldDataType.String, key=True),
            SimpleField(name=Config.SEARCH_PARENT_FIELD, type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="last_modified", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="content_hash", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name=Config.SEARCH_CITATION_FIELD, type=SearchFieldDataType.String),
            SimpleField(name=Config.SEARCH_BOUNDING_BOX_FIELD, type=SearchFieldDataType.String),
            SimpleField(name=Config.SEARCH_SOURCE_REGIONS_FIELD, type=SearchFieldDataType.String),
            SearchField(name=Config.SEARCH_CONTENT_FIELD, type=SearchFieldDataType.String, searchable=True, analyzer_name="ko.microsoft"),
            SearchField(name=Config.SEARCH_ORIGINAL_CONTENT_FIELD, type=SearchFieldDataType.String, searchable=True),
            SearchField(name=Config.SEARCH_TITLE_FIELD, type=SearchFieldDataType.String, searchable=True),
            SearchField(
                name=Config.SEARCH_VECTOR_FIELD,
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=vector_dim,
                vector_search_profile_name="my-vector-profile",
            ),
        ]

    @staticmethod
    def _batched(items: List[Any], batch_size: int):
        for start in range(0, len(items), batch_size):
            yield items[start:start + batch_size]

    @staticmethod
    def _json_dumps(value: Any) -> str:
        if value in (None, "", [], {}):
            return ""
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _build_citation_value(chunk: Document, fallback_source: str) -> str:
        page_numbers = chunk.metadata.get("page_numbers") or []
        if not page_numbers and chunk.metadata.get("page_number") is not None:
            page_numbers = [chunk.metadata.get("page_number")]

        page_part = ""
        if page_numbers:
            if len(page_numbers) == 1:
                page_part = f"p.{page_numbers[0]}"
            else:
                page_part = "pp." + ",".join(str(page) for page in page_numbers)

        bbox = chunk.metadata.get("bounding_box") or {}
        bbox_part = ""
        if bbox:
            bbox_part = (
                "bbox: "
                f"{bbox.get('left', 0):.2f},"
                f"{bbox.get('top', 0):.2f},"
                f"{bbox.get('right', 0):.2f},"
                f"{bbox.get('bottom', 0):.2f}"
            )

        suffix = " | ".join(part for part in [page_part, bbox_part] if part)
        return f"{fallback_source} | {suffix}" if suffix else fallback_source

    def create_index_if_not_exists(self, vector_dim: int = None) -> None:
        """
        AI Search 인덱스가 존재하지 않으면 생성합니다.
        벡터 검색, 시맨틱 랭킹(Semantic Ranking), Contextual Retrieval 필드를 포함합니다.

        인덱스 필드 구성 (v4.1):
        - Config.SEARCH_CONTENT_FIELD: 맥락 포함된 텍스트 (BM25 키워드 검색 + ko.microsoft 분석기)
        - Config.SEARCH_ORIGINAL_CONTENT_FIELD: 원본 텍스트 (답변 생성용, 맥락 제외)
        - Config.SEARCH_VECTOR_FIELD: 맥락 포함된 텍스트의 Contextual Embeddings

        Args:
            vector_dim: 벡터 필드의 차원 수. 기본값: Config.EMBEDDING_DIMENSIONS
                        (text-embedding-3-large + dimensions=2048 권장)
        """
        vector_dim = vector_dim or Config.EMBEDDING_DIMENSIONS
        try:
            self.index_client.get_index(self.index_name)
            print(f"✅ 인덱스 존재: '{self.index_name}'")
        except Exception: # Changed from bare except
            print(f"🛠️ 인덱스 생성 중: '{self.index_name}'...")

            fields = self._base_document_fields(vector_dim)

            # 벡터 검색 알고리즘 및 프로필 설정
            vector_search = VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name="my-hnsw")],
                profiles=[VectorSearchProfile(name="my-vector-profile", algorithm_configuration_name="my-hnsw")]
            )

            # 시맨틱 검색 설정 (한국어 검색 품질 향상)
            semantic_search = SemanticSearch(configurations=[self._semantic_configuration()])

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
        기존 인덱스 스키마에 증분 업데이트용 필드, 시맨틱 검색 설정,
        그리고 원본 텍스트/부모 문서 추적 필드가 없으면 동적으로 추가합니다.
        """
        try:
            index = self.index_client.get_index(self.index_name)
            field_names = [f.name for f in index.fields]
            updated = False

            if Config.SEARCH_PARENT_FIELD not in field_names:
                print(f"🛠️ '{Config.SEARCH_PARENT_FIELD}' 필드 추가 중: {self.index_name}")
                index.fields.append(
                    SimpleField(name=Config.SEARCH_PARENT_FIELD, type=SearchFieldDataType.String, filterable=True, facetable=True)
                )
                updated = True

            if "last_modified" not in field_names:
                print(f"🛠️ 'last_modified' 필드 추가 중: {self.index_name}")
                index.fields.append(SimpleField(name="last_modified", type=SearchFieldDataType.String, filterable=True))
                updated = True

            if "content_hash" not in field_names:
                print(f"🛠️ 'content_hash' 필드 추가 중: {self.index_name}")
                index.fields.append(SimpleField(name="content_hash", type=SearchFieldDataType.String, filterable=True))
                updated = True

            if Config.SEARCH_CITATION_FIELD not in field_names:
                print(f"🛠️ '{Config.SEARCH_CITATION_FIELD}' 필드 추가 중: {self.index_name}")
                index.fields.append(SimpleField(name=Config.SEARCH_CITATION_FIELD, type=SearchFieldDataType.String))
                updated = True

            if Config.SEARCH_BOUNDING_BOX_FIELD not in field_names:
                print(f"🛠️ '{Config.SEARCH_BOUNDING_BOX_FIELD}' 필드 추가 중: {self.index_name}")
                index.fields.append(SimpleField(name=Config.SEARCH_BOUNDING_BOX_FIELD, type=SearchFieldDataType.String))
                updated = True

            if Config.SEARCH_SOURCE_REGIONS_FIELD not in field_names:
                print(f"🛠️ '{Config.SEARCH_SOURCE_REGIONS_FIELD}' 필드 추가 중: {self.index_name}")
                index.fields.append(SimpleField(name=Config.SEARCH_SOURCE_REGIONS_FIELD, type=SearchFieldDataType.String))
                updated = True

            # v4.1: 원본 텍스트 보존 필드 동적 추가
            if Config.SEARCH_ORIGINAL_CONTENT_FIELD not in field_names:
                print(f"🛠️ '{Config.SEARCH_ORIGINAL_CONTENT_FIELD}' 필드 추가 중: {self.index_name}")
                index.fields.append(
                    SearchField(name=Config.SEARCH_ORIGINAL_CONTENT_FIELD, type=SearchFieldDataType.String, searchable=True)
                )
                updated = True

            # [v5.1-fix] 제목/소스 필드가 기존 인덱스에 없을 경우 동적 추가
            if Config.SEARCH_TITLE_FIELD not in field_names:
                print(f"🛠️ '{Config.SEARCH_TITLE_FIELD}' 필드 추가 중: {self.index_name}")
                index.fields.append(
                    SearchField(name=Config.SEARCH_TITLE_FIELD, type=SearchFieldDataType.String, searchable=True)
                )
                updated = True

            if Config.SEARCH_SOURCE_FIELD and Config.SEARCH_SOURCE_FIELD not in field_names:
                print(f"🛠️ '{Config.SEARCH_SOURCE_FIELD}' 필드 추가 중: {self.index_name}")
                index.fields.append(
                    SearchField(name=Config.SEARCH_SOURCE_FIELD, type=SearchFieldDataType.String, searchable=True, filterable=True)
                )
                updated = True

            # 시맨틱 검색 설정 확인 및 추가
            if not index.semantic_search or not any(
                c.name == Config.SEARCH_SEMANTIC_CONFIG for c in index.semantic_search.configurations
            ):
                print(f"🛠️ '{Config.SEARCH_SEMANTIC_CONFIG}' 시맨틱 검색 설정 추가 중: {self.index_name}")
                index.semantic_search = SemanticSearch(configurations=[self._semantic_configuration()])
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

        [v4.1 Contextual Retrieval]
        - Config.SEARCH_CONTENT_FIELD: 맥락 포함 텍스트 (BM25 키워드 + Contextual BM25)
        - Config.SEARCH_ORIGINAL_CONTENT_FIELD: 원본 텍스트 (맥락 제외, 답변 생성용)
        - Config.SEARCH_VECTOR_FIELD: 맥락 포함 텍스트 기반 Contextual Embeddings

        Args:
            chunks: 업로드할 Document 객체 리스트.
        """
        if not chunks:
            return

        print(f"📡 {len(chunks)}개 청크 배치 임베딩 및 업로드 중... 인덱스: '{self.index_name}'")

        # 1. 문서 텍스트 추출 (Contextual Retrieval: 맥락이 포함된 텍스트로 임베딩)
        # page_content에는 맥락이 이미 포함되어 있음 (Contextual Embeddings + Contextual BM25)
        texts = [chunk.page_content for chunk in chunks]

        # 배치 임베딩 (한 번에 2048개까지 처리 가능하나, API 안정성과 속도 균형점: 500개 단위)
        vectors = []
        batch_size = 500
        for batch_texts in self._batched(texts, batch_size):
            response = self.openai_client.embeddings.create(
                input=batch_texts,
                model=Config.EMBEDDING_DEPLOYMENT
            )
            vectors.extend([data.embedding for data in response.data])

        # 2. AI Search 업로드용 데이터 구성
        documents = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            # 파일명과 인덱스를 조합하여 고유 ID 생성 (중복 방지)
            parent_id_str = str(chunk.metadata.get("source", "unknown"))
            title = str(chunk.metadata.get("Header 1") or parent_id_str or "No Title")
            document = {
                Config.SEARCH_ID_FIELD: self._build_document_id(parent_id_str, i),
                Config.SEARCH_CONTENT_FIELD: chunk.page_content,
                Config.SEARCH_ORIGINAL_CONTENT_FIELD: chunk.metadata.get("original_chunk", chunk.page_content),
                Config.SEARCH_TITLE_FIELD: title,
                Config.SEARCH_PARENT_FIELD: parent_id_str,
                "last_modified": str(chunk.metadata.get("last_modified", "")),
                "content_hash": str(chunk.metadata.get("content_hash", "")),
                Config.SEARCH_CITATION_FIELD: self._build_citation_value(chunk, title),
                Config.SEARCH_BOUNDING_BOX_FIELD: self._json_dumps(chunk.metadata.get("bounding_box")),
                Config.SEARCH_SOURCE_REGIONS_FIELD: self._json_dumps(chunk.metadata.get("source_regions")),
                Config.SEARCH_VECTOR_FIELD: vector,
            }
            if Config.SEARCH_SOURCE_FIELD not in document:
                document[Config.SEARCH_SOURCE_FIELD] = title
            documents.append(document)

        # 3. 문서 업로드 (배치 처리)
        failed_count = 0
        for batch_idx, batch in enumerate(self._batched(documents, 50)):
            try:
                result = self.search_client.upload_documents(batch)
                errors = [r for r in result if not r.succeeded]
                if errors:
                    failed_count += len(errors)
                    print(f"   ⚠️ 배치 {batch_idx + 1}: {len(errors)}개 문서 업로드 실패")
            except Exception as e:
                failed_count += len(batch)
                print(f"   ❌ 배치 {batch_idx + 1} 업로드 오류: {e}")

        success_count = len(documents) - failed_count
        if failed_count:
            print(f"⚠️ {success_count}/{len(documents)}개 업로드 완료 ({failed_count}개 실패)")
        else:
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
            # filter-only 검색 최적화: search_text 없이 필터만 사용
            results = self.search_client.search(
                search_text=None,
                filter=f"{Config.SEARCH_PARENT_FIELD} eq '{file_name}'",
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
        특정 파일(parent_id)에 연관된 모든 청크 데이터를 삭제합니다.
        실제 필터링에는 Config.SEARCH_PARENT_FIELD를 사용합니다.

        Args:
            parent_id: 삭제할 부모 문서 ID(파일명).
        """
        try:
            # filter-only 검색 최적화: search_text 없이 필터만 사용
            results = self.search_client.search(
                search_text=None,
                filter=f"{Config.SEARCH_PARENT_FIELD} eq '{parent_id}'",
                select=[Config.SEARCH_ID_FIELD]
            )

            ids_to_delete = [{Config.SEARCH_ID_FIELD: r[Config.SEARCH_ID_FIELD]} for r in results]
            if ids_to_delete:
                print(f"🗑️ 기존 데이터 삭제 중: '{parent_id}' ({len(ids_to_delete)}개 청크)")
                for batch in self._batched(ids_to_delete, 100):
                    self.search_client.delete_documents(batch)
                print("✅ 삭제 완료.")
        except Exception as e:
            print(f"⚠️ 데이터 삭제 오류: {e}")
