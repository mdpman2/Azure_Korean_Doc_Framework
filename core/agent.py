from .multi_model_manager import MultiModelManager
from ..utils.azure_clients import AzureClientFactory
from ..config import Config

class KoreanDocAgent:
    def __init__(self, model_key=None):
        self.model_manager = MultiModelManager(default_model=model_key)
        self.search_client = AzureClientFactory.get_search_client()

    def answer_question(self, question, model_key=None):
        """
        1. AI Search에서 관련 문서 검색
        2. 검색된 문맥(Context)을 바탕으로 답변 생성
        """
        print(f"🔎 Searching for: {question}")

        # 실제 환경에서는 벡터 검색을 수행해야 함
        # 여기서는 단순 텍스트 검색 예시로 대체 (Hybrid Search 설정 필요)
        # 검색 수행 (select 제거하여 모든 필드 가져오거나, 가능한 필드 자동 감지)
        results = self.search_client.search(
            search_text=question,
            top=3
        )

        contexts = []
        for r in results:
            # content 필드 우선 시도, 없으면 text나 chunk 등 유사 필드 시도
            content = r.get('content') or r.get('text') or r.get('chunk') or ""
            if content:
                contexts.append(content)

        context = "\n".join(contexts)

        if not context:
            print("⚠️ No relevant documentation found.")
            context = "관련된 문서 내용을 찾을 수 없습니다."

        system_prompt = (
            "당신은 문서 분석 및 Q&A 전문가입니다. "
            "주어진 [Context] 내용을 바탕으로 사용자의 [Question]에 한국어로 친절하고 정확하게 답변하세요. "
            "추출된 정보가 부족하면 아는 범위 내에서 최선을 다해 답변하되, 정보가 전혀 없다면 솔직하게 모른다고 답하세요."
        )

        user_prompt = f"[Context]\n{context}\n\n[Question]\n{question}"

        answer = self.model_manager.get_completion(
            prompt=user_prompt,
            model_key=model_key,
            system_message=system_prompt
        )

        return answer
