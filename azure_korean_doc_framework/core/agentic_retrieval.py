"""
Agentic Retrieval 모듈 (v6.0 — Azure AI Search Knowledge Base 통합)

Azure AI Search의 Knowledge Base / Agentic Retrieval API(2025-11-01-preview+)를 활용하여
쿼리 분해 → 병렬 검색 → 시맨틱 랭킹 → 답변 합성을 자동화합니다.

주요 기능:
- Knowledge Base 생성/관리
- 자동 쿼리 분해 및 병렬 검색 (Agentic Retrieval)
- Answer Synthesis 또는 Extractive Data 모드 지원
- Retrieval Reasoning Effort 제어 (minimal/low/medium)
- 다중 Knowledge Source 라우팅

참조: https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-overview
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import Config
from ..utils.azure_clients import AzureClientFactory


@dataclass
class AgenticRetrievalResult:
    """Agentic Retrieval 응답 결과"""
    answer: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    query_plan: List[str] = field(default_factory=list)
    reasoning_effort: str = "low"
    output_mode: str = "extractive_data"
    raw_response: Optional[Dict[str, Any]] = None


class AgenticRetrievalManager:
    """
    Azure AI Search Knowledge Base를 활용한 Agentic Retrieval 관리자.

    Knowledge Base는 쿼리를 자동 분해하고, 병렬 검색 후 시맨틱 랭킹을 거쳐
    답변을 합성하거나 원시 검색 결과를 반환합니다.

    [v6.0 신규]
    - answer_synthesis 모드: LLM이 검색 결과 기반으로 답변 직접 생성
    - extractive_data 모드: 원시 검색 결과를 반환하여 다운스트림 처리
    - reasoning_effort: minimal(최소 비용), low(기본), medium(고품질)
    """

    def __init__(
        self,
        kb_name: Optional[str] = None,
        output_mode: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ):
        self.kb_name = kb_name or Config.AGENTIC_KB_NAME
        self.output_mode = output_mode or Config.AGENTIC_OUTPUT_MODE
        self.reasoning_effort = reasoning_effort or Config.AGENTIC_REASONING_EFFORT

        # Azure AI Search 인덱스 클라이언트
        self.index_client = AzureClientFactory.get_search_index_client()
        self.search_client = AzureClientFactory.get_search_client()

        print(f"🤖 AgenticRetrievalManager 초기화 (KB: {self.kb_name}, "
              f"모드: {self.output_mode}, reasoning: {self.reasoning_effort})")

    def retrieve(
        self,
        question: str,
        max_docs: int = 10,
        reasoning_effort: Optional[str] = None,
    ) -> AgenticRetrievalResult:
        """
        Knowledge Base를 통한 Agentic Retrieval을 수행합니다.

        Knowledge Base가 자동으로:
        1. 복잡한 쿼리를 서브쿼리로 분해
        2. 각 서브쿼리를 병렬 실행
        3. 시맨틱 랭킹으로 결과 재순위
        4. (answer_synthesis 모드 시) LLM으로 답변 합성

        Args:
            question: 사용자 질문
            max_docs: 최대 반환 문서 수
            reasoning_effort: 추론 강도 오버라이드

        Returns:
            AgenticRetrievalResult
        """
        effort = reasoning_effort or self.reasoning_effort

        try:
            # Knowledge Base retrieve API 호출
            # NOTE: azure-search-documents SDK의 Knowledge Base API가 GA되면
            # 아래 REST 호출을 SDK 메서드로 교체
            result = self._call_retrieve_api(question, max_docs, effort)
            return result

        except Exception as e:
            print(f"   ⚠️ Agentic Retrieval 실패: {e}")
            return AgenticRetrievalResult(
                answer=f"Agentic Retrieval 호출 실패: {e}",
                reasoning_effort=effort,
                output_mode=self.output_mode,
            )

    def _call_retrieve_api(
        self,
        question: str,
        max_docs: int,
        reasoning_effort: str,
    ) -> AgenticRetrievalResult:
        """
        Azure AI Search Knowledge Base Retrieve API를 호출합니다.

        SDK에서 아직 Preview인 경우 REST fallback을 사용합니다.
        """
        import requests

        endpoint = Config.SEARCH_ENDPOINT.rstrip("/")
        api_key = Config.SEARCH_KEY
        api_version = "2025-11-01-preview"

        url = f"{endpoint}/knowledgebases/{self.kb_name}/retrieve?api-version={api_version}"

        headers = {
            "Content-Type": "application/json",
            "api-key": api_key,
        }

        body = {
            "query": question,
            "retrievalReasoningEffort": {"kind": reasoning_effort},
        }

        if self.output_mode == "answer_synthesis":
            body["outputMode"] = "answerSynthesis"

        response = requests.post(url, headers=headers, json=body, timeout=60)
        response.raise_for_status()

        data = response.json()
        return self._parse_retrieve_response(data, reasoning_effort)

    def _parse_retrieve_response(
        self,
        data: Dict[str, Any],
        reasoning_effort: str,
    ) -> AgenticRetrievalResult:
        """Retrieve API 응답을 파싱합니다."""
        answer = ""
        citations = []
        query_plan = []

        # Answer Synthesis 모드
        if "answer" in data:
            answer = data["answer"].get("text", "")
            citations = data["answer"].get("citations", [])

        # Extractive Data 모드
        if "results" in data:
            for result in data.get("results", []):
                citations.append({
                    "content": result.get("content", ""),
                    "source": result.get("source", ""),
                    "score": result.get("score", 0.0),
                })

        # 쿼리 플랜
        if "activity" in data:
            for step in data["activity"].get("steps", []):
                query_plan.append(step.get("description", ""))

        return AgenticRetrievalResult(
            answer=answer,
            citations=citations,
            query_plan=query_plan,
            reasoning_effort=reasoning_effort,
            output_mode=self.output_mode,
            raw_response=data,
        )

    def create_knowledge_base(
        self,
        name: Optional[str] = None,
        knowledge_sources: Optional[List[str]] = None,
        description: str = "",
        aoai_endpoint: Optional[str] = None,
        aoai_deployment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Knowledge Base를 생성합니다.

        Args:
            name: KB 이름 (기본값: Config.AGENTIC_KB_NAME)
            knowledge_sources: 연결할 Knowledge Source 이름 리스트
            description: KB 설명 (LLM이 쿼리 계획에 활용)
            aoai_endpoint: Azure OpenAI 엔드포인트
            aoai_deployment: Azure OpenAI 배포명

        Returns:
            생성된 Knowledge Base 정의
        """
        import requests

        kb_name = name or self.kb_name
        endpoint = Config.SEARCH_ENDPOINT.rstrip("/")
        api_key = Config.SEARCH_KEY
        api_version = "2025-11-01-preview"

        url = f"{endpoint}/knowledgebases/{kb_name}?api-version={api_version}"

        headers = {
            "Content-Type": "application/json",
            "api-key": api_key,
        }

        body = {
            "name": kb_name,
            "description": description or f"Korean Document RAG Knowledge Base",
            "knowledgeSources": [
                {"name": ks} for ks in (knowledge_sources or [Config.SEARCH_INDEX_NAME])
            ],
            "retrievalReasoningEffort": {"kind": self.reasoning_effort},
        }

        if Config.AGENTIC_RETRIEVAL_INSTRUCTIONS:
            body["retrievalInstructions"] = Config.AGENTIC_RETRIEVAL_INSTRUCTIONS

        if Config.AGENTIC_ANSWER_INSTRUCTIONS:
            body["answerInstructions"] = Config.AGENTIC_ANSWER_INSTRUCTIONS

        if self.output_mode == "answer_synthesis":
            body["outputMode"] = "answerSynthesis"

        # LLM 모델 연결
        oai_endpoint = aoai_endpoint or Config.OPENAI_ENDPOINT_5 or Config.OPENAI_ENDPOINT
        oai_deployment = aoai_deployment or Config.MODELS.get(Config.DEFAULT_MODEL, "model-router")
        if oai_endpoint:
            body["models"] = [{
                "azureOpenAIParameters": {
                    "resourceUri": oai_endpoint,
                    "deploymentName": oai_deployment,
                    "modelName": oai_deployment,
                }
            }]

        response = requests.put(url, headers=headers, json=body, timeout=30)
        response.raise_for_status()

        result = response.json()
        print(f"✅ Knowledge Base '{kb_name}' 생성/업데이트 완료")
        return result
