from typing import Optional, Dict, Any
from ..config import Config
from ..utils.azure_clients import AzureClientFactory

class MultiModelManager:
    """
    GPT-5.2, GPT-4.1, Claude 등 다양한 모델에 대한 API 호출을 통합 관리하는 클래스입니다.
    모델 키에 따라 적절한 Azure OpenAI 엔드포인트 및 배포판으로 요청을 라우팅합니다.

    [2026-01 업데이트]
    - GPT-5.2 기본 모델 지원
    - Structured Outputs 지원
    - max_completion_tokens 파라미터 사용 (GPT-5.x)
    - reasoning_effort 파라미터 지원 (추론 모델)
    """
    def __init__(self, default_model: Optional[str] = None):
        self.default_model = default_model or Config.DEFAULT_MODEL

    def get_completion(
        self,
        prompt: str,
        model_key: Optional[str] = None,
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        reasoning_effort: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        요청된 모델을 사용하여 텍스트 생성을 수행합니다.

        Args:
            prompt: 사용자 프롬프트
            model_key: 사용할 모델 키 (Config.MODELS에 정의)
            system_message: 시스템 메시지
            temperature: 생성 온도 (0.0 ~ 1.0)
            max_tokens: 최대 토큰 수
            reasoning_effort: 추론 강도 ('low', 'medium', 'high') - GPT-5.x, o3, o4-mini 전용
            response_format: Structured Outputs용 응답 형식 (예: {"type": "json_schema", ...})

        Returns:
            생성된 텍스트
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
            key = self.default_model

        # GPT-5.x 여부 확인 (max_completion_tokens 사용 모델)
        # o-시리즈도 GPT-5.x와 동일한 API 파라미터 사용
        is_gpt5_series = key.startswith("gpt-5") or key.startswith("o3") or key.startswith("o4")
        is_reasoning_model = key in Config.REASONING_MODELS

        print(f"🤖 LLM 호출 중: {key} (배포명: {model_name}, 고성능: {is_advanced}, GPT-5.x: {is_gpt5_series})")

        try:
            # 기본 파라미터 구성
            completion_params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature
            }

            # GPT-5.x/o-시리즈: max_completion_tokens 사용, 그 외: max_tokens
            if is_gpt5_series:
                completion_params["max_completion_tokens"] = max_tokens
            else:
                completion_params["max_tokens"] = max_tokens

            # Reasoning effort (GPT-5.x, o3, o4-mini 전용)
            if reasoning_effort and is_reasoning_model:
                completion_params["reasoning_effort"] = reasoning_effort
                print(f"   🧠 Reasoning effort: {reasoning_effort}")

            # Structured Outputs (response_format)
            if response_format:
                completion_params["response_format"] = response_format
                print(f"   📋 Structured Output enabled")

            response = client.chat.completions.create(**completion_params)
            return response.choices[0].message.content

        except Exception as e:
            return f"❌ LLM 호출 중 오류 발생: {e}"

    def get_structured_completion(
        self,
        prompt: str,
        json_schema: Dict[str, Any],
        model_key: Optional[str] = None,
        system_message: str = "You are a helpful assistant that responds in JSON.",
        temperature: float = 0.0
    ) -> str:
        """
        Structured Outputs를 사용하여 JSON 형식의 응답을 생성합니다.

        Args:
            prompt: 사용자 프롬프트
            json_schema: JSON 스키마 정의
            model_key: 사용할 모델 키
            system_message: 시스템 메시지
            temperature: 생성 온도

        Returns:
            JSON 형식의 응답 문자열
        """
        response_format = {
            "type": "json_schema",
            "json_schema": json_schema
        }

        return self.get_completion(
            prompt=prompt,
            model_key=model_key,
            system_message=system_message,
            temperature=temperature,
            response_format=response_format
        )
