from ..config import Config
from ..utils.azure_clients import AzureClientFactory

class MultiModelManager:
    """
    GPT-4, GPT-5, Claude 등 다양한 모델에 대한 API 호출을 통합 관리하는 클래스입니다.
    모델 키에 따라 적절한 Azure OpenAI 엔드포인트 및 배포판으로 요청을 라우팅합니다.
    """
    def __init__(self, default_model=None):
        self.default_model = default_model or Config.DEFAULT_MODEL

    def get_completion(self, prompt, model_key=None, system_message="You are a helpful assistant.", temperature=0.7):
        """
        요청된 모델을 사용하여 텍스트 생성을 수행합니다.
        """
        key = model_key or self.default_model
        model_name = Config.MODELS.get(key)

        # 고성능 모델(Advanced) 여부 확인 (Config.ADVANCED_MODELS 기준)
        is_advanced = key in getattr(Config, "ADVANCED_MODELS", [])

        # 해당 그룹(일반/고성능)에 맞는 최적화된 클라이언트 획득 (캐시 활용)
        client = AzureClientFactory.get_openai_client(is_advanced=is_advanced)

        if not model_name:
            print(f"⚠️ 모델 키 '{model_key}'를 찾을 수 없어 기본 모델 '{self.default_model}'을 사용합니다.")
            model_name = Config.MODELS.get(self.default_model)

        print(f"🤖 LLM 호출 중: {key} (배포명: {model_name}, 고성능모드: {is_advanced})")

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ LLM 호출 중 오류 발생: {e}"
