"""
RAGAS 기반 RAG 품질 평가 모듈 (v5.1 — LightRAG RAGAS 통합 참조)

LightRAG의 RAGAS 통합을 참조하여 구현한 표준 RAG 메트릭 평가 시스템입니다.
검색 단계와 생성 단계를 분리 평가하여 정확히 어디서 정확도가 떨어지는지 진단합니다.

평가 메트릭:
1. Context Precision: 검색된 문서 중 관련 문서 비율
2. Context Recall: 정답에 필요한 정보가 검색 결과에 포함된 비율
3. Faithfulness: 답변이 검색 결과에 근거하는 정도
4. Answer Relevancy: 답변이 질문에 얼마나 관련되는지
5. Answer Correctness: 답변이 정답과 일치하는 정도

참조: https://github.com/HKUDS/LightRAG — RAGAS 통합
참조: https://docs.ragas.io/en/stable/
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core.multi_model_manager import MultiModelManager


@dataclass
class RAGASMetrics:
    """RAGAS 평가 결과"""
    context_precision: float = 0.0
    context_recall: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    answer_correctness: float = 0.0
    overall_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "answer_correctness": round(self.answer_correctness, 4),
            "overall_score": round(self.overall_score, 4),
            "details": self.details,
        }


@dataclass
class RAGASBatchResult:
    """배치 평가 결과"""
    items: List[Dict[str, Any]] = field(default_factory=list)
    average_metrics: Optional[RAGASMetrics] = None
    total_items: int = 0
    evaluated_items: int = 0


_CONTEXT_PRECISION_PROMPT = """당신은 RAG 시스템의 검색 품질을 평가하는 전문가입니다.

주어진 질문과 정답을 기준으로, 검색된 각 문서가 정답에 필요한 정보를 포함하는지 판단하세요.

[질문]
{question}

[정답]
{ground_truth}

[검색된 문서들]
{contexts}

각 문서에 대해 "relevant" (정답에 필요한 정보를 포함) 또는 "irrelevant" (불필요)로 판정하세요.

응답 형식 (JSON):
{{
  "judgments": [
    {{"doc_index": 0, "verdict": "relevant", "reason": "이유"}},
    {{"doc_index": 1, "verdict": "irrelevant", "reason": "이유"}}
  ]
}}"""

_CONTEXT_RECALL_PROMPT = """주어진 정답의 각 핵심 구문/사실이 검색된 문서들에 포함되어 있는지 판단하세요.

[정답]
{ground_truth}

[검색된 문서들]
{contexts}

정답의 각 핵심 사실에 대해 판정하세요.

응답 형식 (JSON):
{{
  "facts": [
    {{"fact": "핵심 사실1", "found_in_context": true, "doc_index": 0}},
    {{"fact": "핵심 사실2", "found_in_context": false, "doc_index": null}}
  ]
}}"""

_FAITHFULNESS_PROMPT = """답변의 각 주장이 검색된 문서에 근거하는지 판단하세요.

[답변]
{answer}

[검색된 문서들]
{contexts}

답변의 각 주장에 대해 근거 여부를 판정하세요.

응답 형식 (JSON):
{{
  "claims": [
    {{"claim": "주장1", "supported": true, "doc_index": 0}},
    {{"claim": "주장2", "supported": false, "doc_index": null}}
  ]
}}"""

_ANSWER_RELEVANCY_PROMPT = """답변이 질문에 얼마나 관련되는지 평가하세요.

[질문]
{question}

[답변]
{answer}

0.0 (완전 무관) ~ 1.0 (완벽히 관련) 사이 점수와 이유를 반환하세요.

응답 형식 (JSON):
{{
  "score": 0.85,
  "reason": "평가 이유"
}}"""

_ANSWER_CORRECTNESS_PROMPT = """답변이 정답과 얼마나 일치하는지 평가하세요.

[질문]
{question}

[정답]
{ground_truth}

[모델 답변]
{answer}

0.0 (완전 불일치) ~ 1.0 (완벽 일치) 사이 점수와 이유를 반환하세요.
부분 일치도 인정합니다. 핵심 정보가 모두 포함되면 0.8 이상입니다.

응답 형식 (JSON):
{{
  "score": 0.9,
  "reason": "평가 이유",
  "missing_info": ["누락 정보1"],
  "extra_info": ["추가 정보1"]
}}"""


class RAGASEvaluator:
    """
    RAGAS 스타일 RAG 품질 평가기

    LightRAG의 RAGAS 통합을 참조하여, LLM-as-Judge 방식으로
    검색/생성 단계별 품질을 측정합니다.

    Args:
        model_manager: MultiModelManager 인스턴스
        judge_model: 평가용 모델 키 (기본: gpt-5.4)
    """

    def __init__(
        self,
        model_manager: MultiModelManager,
        judge_model: str = "gpt-5.4",
    ):
        self.model_manager = model_manager
        self.judge_model = judge_model

    def _call_judge(self, prompt: str) -> Dict[str, Any]:
        """LLM 판정을 호출하고 JSON을 파싱합니다."""
        response = self.model_manager.get_completion_with_retry(
            prompt=prompt,
            model_key=self.judge_model,
            system_message="당신은 RAG 시스템 품질 평가 전문가입니다. 항상 유효한 JSON으로만 응답하세요.",
        )
        # JSON 추출
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            text = "\n".join(json_lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # JSON 블록 추출 시도
            import re
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {}

    def evaluate_context_precision(
        self,
        question: str,
        ground_truth: str,
        contexts: List[str],
    ) -> float:
        """Context Precision: 검색 결과 중 관련 문서 비율"""
        if not contexts:
            return 0.0

        ctx_text = "\n".join(f"[문서 {i}]\n{c[:500]}" for i, c in enumerate(contexts))
        prompt = _CONTEXT_PRECISION_PROMPT.format(
            question=question, ground_truth=ground_truth, contexts=ctx_text,
        )

        result = self._call_judge(prompt)
        judgments = result.get("judgments", [])
        if not judgments:
            return 0.0

        relevant = sum(1 for j in judgments if j.get("verdict") == "relevant")
        return relevant / len(judgments)

    def evaluate_context_recall(
        self,
        ground_truth: str,
        contexts: List[str],
    ) -> float:
        """Context Recall: 정답 사실이 검색 결과에 포함된 비율"""
        if not contexts or not ground_truth:
            return 0.0

        ctx_text = "\n".join(f"[문서 {i}]\n{c[:500]}" for i, c in enumerate(contexts))
        prompt = _CONTEXT_RECALL_PROMPT.format(
            ground_truth=ground_truth, contexts=ctx_text,
        )

        result = self._call_judge(prompt)
        facts = result.get("facts", [])
        if not facts:
            return 0.0

        found = sum(1 for f in facts if f.get("found_in_context"))
        return found / len(facts)

    def evaluate_faithfulness(
        self,
        answer: str,
        contexts: List[str],
    ) -> float:
        """Faithfulness: 답변이 검색 결과에 근거하는 정도"""
        if not answer or not contexts:
            return 0.0

        ctx_text = "\n".join(f"[문서 {i}]\n{c[:500]}" for i, c in enumerate(contexts))
        prompt = _FAITHFULNESS_PROMPT.format(answer=answer, contexts=ctx_text)

        result = self._call_judge(prompt)
        claims = result.get("claims", [])
        if not claims:
            return 0.0

        supported = sum(1 for c in claims if c.get("supported"))
        return supported / len(claims)

    def evaluate_answer_relevancy(
        self,
        question: str,
        answer: str,
    ) -> float:
        """Answer Relevancy: 답변이 질문에 관련되는 정도"""
        if not answer:
            return 0.0

        prompt = _ANSWER_RELEVANCY_PROMPT.format(question=question, answer=answer)
        result = self._call_judge(prompt)

        score = result.get("score", 0.0)
        try:
            return max(0.0, min(1.0, float(score)))
        except (ValueError, TypeError):
            return 0.0

    def evaluate_answer_correctness(
        self,
        question: str,
        ground_truth: str,
        answer: str,
    ) -> Tuple[float, Dict[str, Any]]:
        """Answer Correctness: 답변과 정답의 일치도"""
        if not answer or not ground_truth:
            return 0.0, {}

        prompt = _ANSWER_CORRECTNESS_PROMPT.format(
            question=question, ground_truth=ground_truth, answer=answer,
        )
        result = self._call_judge(prompt)

        score = result.get("score", 0.0)
        try:
            score = max(0.0, min(1.0, float(score)))
        except (ValueError, TypeError):
            score = 0.0

        detail = {
            "reason": result.get("reason", ""),
            "missing_info": result.get("missing_info", []),
            "extra_info": result.get("extra_info", []),
        }
        return score, detail

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = "",
    ) -> RAGASMetrics:
        """
        전체 RAGAS 메트릭을 한 번에 평가합니다.

        Args:
            question: 사용자 질문
            answer: 모델 답변
            contexts: 검색된 컨텍스트 리스트
            ground_truth: 정답 (없으면 correctness/precision/recall 건너뜀)

        Returns:
            RAGASMetrics: 모든 메트릭 포함
        """
        metrics = RAGASMetrics()
        details = {}

        # 항상 평가 가능한 메트릭
        metrics.faithfulness = self.evaluate_faithfulness(answer, contexts)
        metrics.answer_relevancy = self.evaluate_answer_relevancy(question, answer)

        # 정답이 있을 때만 평가
        if ground_truth:
            metrics.context_precision = self.evaluate_context_precision(
                question, ground_truth, contexts,
            )
            metrics.context_recall = self.evaluate_context_recall(
                ground_truth, contexts,
            )
            correctness, correctness_detail = self.evaluate_answer_correctness(
                question, ground_truth, answer,
            )
            metrics.answer_correctness = correctness
            details["correctness_detail"] = correctness_detail

            # 전체 점수: 5개 메트릭 가중 평균
            weights = {
                "context_precision": 0.15,
                "context_recall": 0.15,
                "faithfulness": 0.30,
                "answer_relevancy": 0.15,
                "answer_correctness": 0.25,
            }
            metrics.overall_score = (
                metrics.context_precision * weights["context_precision"]
                + metrics.context_recall * weights["context_recall"]
                + metrics.faithfulness * weights["faithfulness"]
                + metrics.answer_relevancy * weights["answer_relevancy"]
                + metrics.answer_correctness * weights["answer_correctness"]
            )
        else:
            # 정답 없이 2개 메트릭만으로 점수 산출
            metrics.overall_score = (
                metrics.faithfulness * 0.6
                + metrics.answer_relevancy * 0.4
            )

        metrics.details = details
        return metrics

    def evaluate_batch(
        self,
        items: List[Dict[str, Any]],
        verbose: bool = True,
    ) -> RAGASBatchResult:
        """
        배치 평가를 실행합니다.

        각 item은 다음 키를 포함해야 합니다:
        - "question": 질문
        - "answer": 모델 답변
        - "contexts": 검색 컨텍스트 리스트
        - "ground_truth" (선택): 정답

        Args:
            items: 평가 항목 리스트
            verbose: 진행 상황 출력 여부

        Returns:
            RAGASBatchResult: 배치 평가 결과
        """
        batch_result = RAGASBatchResult(total_items=len(items))
        all_metrics = []

        for i, item in enumerate(items):
            question = item.get("question", "")
            answer = item.get("answer", "")
            contexts = item.get("contexts", [])
            ground_truth = item.get("ground_truth", "")

            if verbose:
                print(f"   📊 [{i+1}/{len(items)}] 평가 중: {question[:50]}...")

            try:
                metrics = self.evaluate(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=ground_truth,
                )
                all_metrics.append(metrics)

                batch_result.items.append({
                    "index": i,
                    "question": question[:100],
                    "metrics": metrics.to_dict(),
                })
                batch_result.evaluated_items += 1

                if verbose:
                    print(f"      Overall: {metrics.overall_score:.3f} | "
                          f"Faith: {metrics.faithfulness:.3f} | "
                          f"Rel: {metrics.answer_relevancy:.3f}")

            except Exception as e:
                if verbose:
                    print(f"      ⚠️ 평가 실패: {e}")
                batch_result.items.append({
                    "index": i,
                    "question": question[:100],
                    "error": str(e),
                })

        # 평균 메트릭 계산
        if all_metrics:
            avg = RAGASMetrics(
                context_precision=sum(m.context_precision for m in all_metrics) / len(all_metrics),
                context_recall=sum(m.context_recall for m in all_metrics) / len(all_metrics),
                faithfulness=sum(m.faithfulness for m in all_metrics) / len(all_metrics),
                answer_relevancy=sum(m.answer_relevancy for m in all_metrics) / len(all_metrics),
                answer_correctness=sum(m.answer_correctness for m in all_metrics) / len(all_metrics),
                overall_score=sum(m.overall_score for m in all_metrics) / len(all_metrics),
            )
            batch_result.average_metrics = avg

        return batch_result
