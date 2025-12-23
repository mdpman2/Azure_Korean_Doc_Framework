from typing import List, Tuple, Optional, Union, Any
from azure.search.documents.models import VectorizedQuery
from langchain_openai import AzureOpenAIEmbeddings
from .multi_model_manager import MultiModelManager
from ..utils.azure_clients import AzureClientFactory
from ..config import Config

class KoreanDocAgent:
    """
    한국어 문서 분석 및 Q&A 전문가 검색 에이전트.

    Azure AI Search의 Hybrid Search를 활용하여 문맥을 찾고,
    Azure OpenAI 모델들을 통해 지능적인 답변을 생성합니다.
    """

    def __init__(self, model_key: Optional[str] = None):
        """
        KoreanDocAgent를 초기화합니다.

        Args:
            model_key: 답변 생성 시 기본으로 사용할 모델 키 (Config.MODELS에 정의된 키).
        """
        self.model_manager = MultiModelManager(default_model=model_key)
        self.search_client = AzureClientFactory.get_search_client()

        # 임베딩 클라이언트 (벡터 검색용)
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=Config.EMBEDDING_DEPLOYMENT,
            openai_api_version=Config.OPENAI_API_VERSION,
            azure_endpoint=Config.OPENAI_ENDPOINT,
            api_key=Config.OPENAI_API_KEY
        )

    def answer_question(
        self,
        question: str,
        model_key: Optional[str] = None,
        return_context: bool = False,
        top_k: int = 5
    ) -> Union[str, Tuple[str, List[str]]]:
        """
        사용자의 질문에 대해 검색 증강 생성(RAG)을 수행합니다.

        1. AI Search에서 하이브리드 검색(벡터+키워드) 및 시맨틱 랭킹 수행
        2. 검색된 문맥(Context)을 바탕으로 답변 생성 (출처 정보 포함)

        Args:
            question: 사용자의 질문 문자열.
            model_key: 답변 생성에 사용할 특정 모델 키.
            return_context: True일 경우 답변과 함께 검색된 컨텍스트 리스트를 반환합니다.
            top_k: 검색할 문서의 개수 (기본값: 5).

        Returns:
            답변 문자열 또는 (답변, 컨텍스트 리스트) 튜플.
        """
        print(f"🔎 Searching for: {question} (top_k={top_k})")

        # 1. 질문 임베딩 생성 (벡터 검색용)
        query_vector = self.embeddings.embed_query(question)
        vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=50, fields="text_vector")

        # 2. 하이브리드 검색 및 시맨틱 랭킹 수행
        try:
            results = self.search_client.search(
                search_text=question,
                vector_queries=[vector_query],
                select=["chunk", "parent_id"],
                query_type="semantic",
                semantic_configuration_name="my-semantic-config",
                top=top_k
            )

            contexts = []
            sources = set()

            for r in results:
                content = r.get('chunk') or r.get('content') or ""
                source = r.get('parent_id') or "알 수 없는 출처"

                if content:
                    # 컨텍스트에 출처 정보 명시적으로 삽입
                    contexts.append(f"[출처: {source}]\n{content}")
                    sources.add(source)

        except Exception as e:
            print(f"   ❌ Search failed: {e}")
            contexts = []

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
