from typing import List, Tuple, Optional, Union, Any, Dict
from azure.search.documents.models import VectorizedQuery
from .multi_model_manager import MultiModelManager
from ..utils.azure_clients import AzureClientFactory
from ..config import Config

class KoreanDocAgent:
    """
    한국어 문서 분석 및 Q&A 전문가 검색 에이전트.

    Azure AI Search의 Hybrid Search + Semantic Ranking을 활용하여 문맥을 찾고,
    GPT-5.2를 통해 지능적인 답변을 생성합니다.

    [2026-01 업데이트]
    - GPT-5.2 기본 모델 사용
    - Query Rewrite 지원 (시맨틱 쿼리 확장)
    - 향상된 Semantic Ranking (L2 reranking)
    - Agentic Retrieval 패턴 지원 준비
    """

    def __init__(self, model_key: Optional[str] = None):
        """
        KoreanDocAgent를 초기화합니다.

        Args:
            model_key: 답변 생성 시 기본으로 사용할 모델 키 (Config.MODELS에 정의된 키).
                      기본값: Config.DEFAULT_MODEL (gpt-5.2)
        """
        self.model_manager = MultiModelManager(default_model=model_key or Config.DEFAULT_MODEL)
        self.search_client = AzureClientFactory.get_search_client()

        # 임베딩 클라이언트 (벡터 검색용) - 기본 엔드포인트 사용 (text-embedding-3-small)
        self.embedding_client = AzureClientFactory.get_openai_client(is_advanced=False)

        # LLM 클라이언트 (Query Rewrite용) - 고성능 엔드포인트
        self.llm_client = AzureClientFactory.get_openai_client(is_advanced=True)

        # Query Rewrite 활성화 여부
        self.enable_query_rewrite = True

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

            import json
            result = response.choices[0].message.content.strip()
            # JSON 배열 파싱
            if result.startswith("["):
                queries = json.loads(result)
                return queries[:3] if queries else [question]
            return [question]

        except Exception as e:
            print(f"   ⚠️ Query rewrite failed, using original: {e}")
            return [question]

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

        # 1. 질문 임베딩 생성 (벡터 검색용) - 원본 질문 사용
        embedding_response = self.embedding_client.embeddings.create(
            input=[question],
            model=Config.EMBEDDING_DEPLOYMENT
        )
        query_vector = embedding_response.data[0].embedding
        vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=50, fields="text_vector")

        # 2. 하이브리드 검색 및 시맨틱 랭킹 수행 (모든 쿼리 변형에 대해)
        all_contexts = []
        all_sources = set()

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

                    # 중복 제거
                    context_entry = f"[출처: {source}]\n{content}"
                    if content and context_entry not in all_contexts:
                        all_contexts.append(context_entry)
                        all_sources.add(source)

        except Exception as e:
            print(f"   ❌ Search failed: {e}")
            all_contexts = []

        # 상위 top_k * 2개만 유지 (중복 제거 후)
        contexts = all_contexts[:top_k * 2]
        context_str = "\n\n".join(contexts)

        if not context_str:
            print("   ⚠️ No relevant documentation found.")
            context_str = "관련된 문서 내용을 찾을 수 없습니다."

        system_prompt = (
            "당신은 문서 분석 및 Q&A 전문가입니다. "
            "주어진 [Context] 내용을 바탕으로 사용자의 [Question]에 한국어로 친절하고 정확하게 답변하세요. "
            "\n\n### 답변 규칙:"
            "\n1. 답변 시 반드시 해당 정보의 **출처(파일명)**를 언급하세요. (예: '...입니다 [출처: 파일명.pdf]')"
            "\n2. 여러 문서에서 정보를 취합한 경우, 각각의 출처를 밝히세요."
            "\n3. 추출된 정보가 부족하면 아는 범위 내에서 최선을 다해 답변하되, 정보가 전혀 없다면 솔직하게 모른다고 답하세요."
        )

        user_prompt = f"[Context]\n{context_str}\n\n[Question]\n{question}"

        # LLM 호출을 통한 답변 생성
        answer = self.model_manager.get_completion(
            prompt=user_prompt,
            model_key=model_key,
            system_message=system_prompt
        )

        if return_context:
            return answer, contexts
        return answer
