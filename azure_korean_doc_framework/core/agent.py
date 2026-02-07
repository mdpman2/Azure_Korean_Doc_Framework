import json
from typing import List, Tuple, Optional, Union
from azure.search.documents.models import VectorizedQuery
from .multi_model_manager import MultiModelManager
from ..utils.azure_clients import AzureClientFactory
from ..config import Config

# 공통 RAG 시스템 프롬프트 (answer_question / graph_enhanced_answer 공유)
_RAG_SYSTEM_PROMPT = (
    "당신은 문서 분석 및 Q&A 전문가입니다. "
    "주어진 [Context] 내용을 바탕으로 사용자의 [Question]에 한국어로 친절하고 정확하게 답변하세요. "
    "\n\n### 답변 규칙:"
    "\n1. 답변 시 반드시 해당 정보의 **출처(파일명)**를 언급하세요. (예: '...입니다 [출처: 파일명.pdf]')"
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

    Azure AI Search의 Hybrid Search + Semantic Ranking을 활용하여 문맥을 찾고,
    GPT-5.2를 통해 지능적인 답변을 생성합니다.

    [2026-02 v4.0 업데이트]
    - Graph-Enhanced RAG (LightRAG 기반 Knowledge Graph 연동)
    - 구조화 엔티티 추출 결과 활용 (LangExtract 기반)
    - GPT-5.2 기본 모델 사용
    - Query Rewrite 지원 (시맨틱 쿼리 확장)
    - 향상된 Semantic Ranking (L2 reranking)
    - Dual-Mode 검색: Vector + Graph 하이브리드
    """

    def __init__(self, model_key: Optional[str] = None, graph_manager=None):
        """
        KoreanDocAgent를 초기화합니다.

        Args:
            model_key: 답변 생성 시 기본으로 사용할 모델 키 (Config.MODELS에 정의된 키).
                      기본값: Config.DEFAULT_MODEL (gpt-5.2)
            graph_manager: KnowledgeGraphManager 인스턴스 (Graph RAG 사용 시)
        """
        self.model_manager = MultiModelManager(default_model=model_key or Config.DEFAULT_MODEL)
        self.search_client = AzureClientFactory.get_search_client()

        # 임베딩 클라이언트 (벡터 검색용) - 기본 엔드포인트 사용 (text-embedding-3-small)
        self.embedding_client = AzureClientFactory.get_openai_client(is_advanced=False)

        # LLM 클라이언트 (Query Rewrite용) - 고성능 엔드포인트
        self.llm_client = AzureClientFactory.get_openai_client(is_advanced=True)

        # Query Rewrite 활성화 여부
        self.enable_query_rewrite = True

        # [v4.0] Graph RAG 매니저 (LightRAG 기반)
        self.graph_manager = graph_manager

    def _rewrite_query(self, question: str) -> List[str]:
        """
        GPT-5.2를 사용하여 쿼리를 의미적으로 확장합니다.
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
                model=Config.MODELS.get("gpt-5.2", "gpt-5.2"),
                messages=[{"role": "user", "content": rewrite_prompt}],
                temperature=0.3,
                max_completion_tokens=200
            )

            result = response.choices[0].message.content.strip()
            # JSON 배열 파싱
            if result.startswith("["):
                queries = json.loads(result)
                return queries[:3] if queries else [question]
            return [question]

        except Exception as e:
            print(f"   ⚠️ Query rewrite failed, using original: {e}")
            return [question]

    # ==================== 공통 벡터 검색 로직 ====================

    def _vector_search(
        self,
        question: str,
        search_queries: List[str],
        top_k: int = 5,
    ) -> List[str]:
        """
        Azure AI Search 하이브리드 검색 (벡터 + 키워드 + 시맨틱 랭킹)

        Args:
            question: 원본 질문 (임베딩용)
            search_queries: 검색 쿼리 리스트 (Query Rewrite 결과 포함)
            top_k: 검색할 문서 수

        Returns:
            검색된 컨텍스트 리스트
        """
        embedding_response = self.embedding_client.embeddings.create(
            input=[question],
            model=Config.EMBEDDING_DEPLOYMENT
        )
        query_vector = embedding_response.data[0].embedding
        vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=50, fields="text_vector")

        all_contexts = []
        seen_contexts: set = set()

        try:
            for search_query in search_queries:
                results = self.search_client.search(
                    search_text=search_query,
                    vector_queries=[vector_query],
                    select=["chunk", "parent_id"],
                    query_type="semantic",
                    semantic_configuration_name="my-semantic-config",
                    top=top_k
                )

                for r in results:
                    content = r.get('chunk') or r.get('content') or ""
                    source = r.get('parent_id') or "알 수 없는 출처"

                    context_entry = f"[출처: {source}]\n{content}"
                    if content and context_entry not in seen_contexts:
                        seen_contexts.add(context_entry)
                        all_contexts.append(context_entry)

        except Exception as e:
            print(f"   ❌ Search failed: {e}")

        return all_contexts[:top_k * 2]

    def answer_question(
        self,
        question: str,
        model_key: Optional[str] = None,
        return_context: bool = False,
        top_k: int = 5,
        use_query_rewrite: bool = True
    ) -> Union[str, Tuple[str, List[str]]]:
        """
        사용자의 질문에 대해 검색 증강 생성(RAG)을 수행합니다.

        1. Query Rewrite (선택적): 질문을 의미적으로 확장
        2. AI Search에서 하이브리드 검색(벡터+키워드) 및 시맨틱 랭킹 수행
        3. 검색된 문맥(Context)을 바탕으로 GPT-5.2로 답변 생성 (출처 정보 포함)

        Args:
            question: 사용자의 질문 문자열.
            model_key: 답변 생성에 사용할 특정 모델 키.
            return_context: True일 경우 답변과 함께 검색된 컨텍스트 리스트를 반환합니다.
            top_k: 검색할 문서의 개수 (기본값: 5).
            use_query_rewrite: Query Rewrite 사용 여부 (기본값: True).

        Returns:
            답변 문자열 또는 (답변, 컨텍스트 리스트) 튜플.
        """
        print(f"🔎 Searching for: {question} (top_k={top_k})")

        # 0. Query Rewrite (선택적)
        search_queries = [question]
        if use_query_rewrite and self.enable_query_rewrite:
            search_queries = self._rewrite_query(question)
            if len(search_queries) > 1:
                print(f"   📝 Query expanded to {len(search_queries)} variants")

        # 1. 벡터 검색 (공통 로직)
        contexts = self._vector_search(question, search_queries, top_k)
        context_str = "\n\n".join(contexts)

        if not context_str:
            print("   ⚠️ No relevant documentation found.")
            context_str = "관련된 문서 내용을 찾을 수 없습니다."

        user_prompt = f"[Context]\n{context_str}\n\n[Question]\n{question}"

        # LLM 호출을 통한 답변 생성
        answer = self.model_manager.get_completion(
            prompt=user_prompt,
            model_key=model_key,
            system_message=_RAG_SYSTEM_PROMPT
        )

        if return_context:
            return answer, contexts
        return answer

    # ==================== v4.0: Graph-Enhanced RAG ====================

    def graph_enhanced_answer(
        self,
        question: str,
        model_key: Optional[str] = None,
        return_context: bool = False,
        top_k: int = 5,
        use_query_rewrite: bool = True,
        graph_query_mode: str = "hybrid",
    ) -> Union[str, Tuple[str, List[str]]]:
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

        Returns:
            답변 문자열 또는 (답변, 컨텍스트 리스트) 튜플.
        """
        print(f"🔎 [Graph-Enhanced] Searching for: {question}")

        # === Part 1: 벡터 검색 (공통 로직) ===
        search_queries = [question]
        if use_query_rewrite and self.enable_query_rewrite:
            search_queries = self._rewrite_query(question)

        vector_contexts = self._vector_search(question, search_queries, top_k)

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

        user_prompt = f"[Context]\n{combined_context}\n\n[Question]\n{question}"

        answer = self.model_manager.get_completion(
            prompt=user_prompt,
            model_key=model_key,
            system_message=_GRAPH_RAG_SYSTEM_PROMPT
        )

        if return_context:
            return answer, vector_contexts
        return answer
