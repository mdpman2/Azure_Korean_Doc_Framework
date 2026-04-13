"""
서브에이전트 위임 시스템.

Claude Code의 AgentTool / 서브에이전트 개념을 RAG 프레임워크에 적용.
복잡한 질문을 분해하여 서브에이전트가 병렬로 조사 → 결과를 종합합니다.

[v5.0 신규]
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..config import Config
from .multi_model_manager import MultiModelManager


@dataclass
class SubTask:
    """서브에이전트에 위임할 하위 작업."""
    task_id: str
    question: str
    context_hint: str = ""
    model_key: Optional[str] = None


@dataclass
class SubTaskResult:
    """서브에이전트 실행 결과."""
    task_id: str
    question: str
    answer: str
    elapsed_ms: float = 0.0
    model_key: Optional[str] = None
    error: Optional[str] = None


@dataclass
class DelegationResult:
    """전체 위임 결과."""
    original_question: str
    sub_results: List[SubTaskResult] = field(default_factory=list)
    synthesized_answer: str = ""
    total_elapsed_ms: float = 0.0
    was_decomposed: bool = False


class QuestionDecomposer:
    """
    복잡한 질문을 하위 질문으로 분해합니다.

    예시:
        "A 회사의 인사제도와 B 회사의 인사제도를 비교해주세요"
        → ["A 회사의 인사제도는?", "B 회사의 인사제도는?"]
    """

    def __init__(self, model_manager: MultiModelManager):
        self.model_manager = model_manager

    def should_decompose(self, question: str) -> bool:
        """질문이 분해가 필요한지 판단합니다."""
        # 비교 질문, 다중 항목 질문, 복합 요청 패턴 감지
        decompose_patterns = [
            "비교", "차이", "대비", "versus", "vs",
            "각각", "모두", "전부",
            "그리고", "또한", "및",
            "첫째.*둘째", "1).*2)",
        ]
        question_lower = question.lower()
        import re
        for pattern in decompose_patterns:
            if re.search(pattern, question_lower):
                return True

        # 질문이 충분히 복잡한지 (토큰 수 기준)
        return len(question) > 100 and ("?" in question or "?" in question)

    def decompose(self, question: str, max_sub_questions: int = 3) -> List[str]:
        """질문을 하위 질문 리스트로 분해합니다."""
        prompt = (
            f"다음 복합 질문을 독립적으로 검색 가능한 하위 질문들로 분해하세요.\n"
            f"최대 {max_sub_questions}개, 각 질문은 자체적으로 완결된 형태여야 합니다.\n"
            f"분해가 불필요하면 원본 질문만 반환하세요.\n"
            f"JSON 배열로 반환: [\"질문1\", \"질문2\", ...]\n\n"
            f"원본 질문: {question}"
        )
        try:
            import json
            result = self.model_manager.get_completion(
                prompt=prompt,
                system_message="질문을 검색 최적화된 하위 질문으로 분해하는 전문가.",
                temperature=0.0,
                max_tokens=500,
            )
            result = result.strip()
            # JSON 배열 추출
            if "[" in result:
                json_str = result[result.index("["):result.rindex("]") + 1]
                sub_questions = json.loads(json_str)
                if isinstance(sub_questions, list) and len(sub_questions) > 1:
                    return sub_questions[:max_sub_questions]
        except Exception:
            pass
        return [question]


class SubAgentManager:
    """
    서브에이전트 위임 관리자.

    복잡한 질문을 감지 → 분해 → 병렬 검색/답변 → 결과 종합.

    사용 예시:
        manager = SubAgentManager(answer_fn=agent.answer_question)

        # 자동 분해 및 종합
        result = manager.delegate("A 인사제도와 B 인사제도를 비교해주세요")
        print(result.synthesized_answer)
    """

    def __init__(
        self,
        answer_fn: Callable[..., str],
        model_manager: Optional[MultiModelManager] = None,
        max_workers: int = 3,
        timeout: int = 60,
    ):
        """
        Args:
            answer_fn: 하위 질문에 대한 답변 함수 (question: str, ...) -> str
            model_manager: LLM 매니저 (분해/종합에 사용)
            max_workers: 병렬 서브에이전트 수
            timeout: 서브에이전트 타임아웃(초)
        """
        self.answer_fn = answer_fn
        self.model_manager = model_manager or MultiModelManager()
        self.max_workers = max_workers
        self.timeout = timeout
        self.decomposer = QuestionDecomposer(self.model_manager)

    def delegate(
        self,
        question: str,
        model_key: Optional[str] = None,
        force_decompose: bool = False,
        **answer_kwargs,
    ) -> DelegationResult:
        """
        질문을 분석하고 필요 시 서브에이전트로 위임합니다.

        Args:
            question: 원본 질문
            model_key: 서브에이전트가 사용할 모델 키
            force_decompose: 강제 분해 여부
            **answer_kwargs: answer_fn에 전달할 추가 인자
        """
        total_start = time.perf_counter()

        # 분해 필요 여부 판단
        if not force_decompose and not self.decomposer.should_decompose(question):
            return DelegationResult(
                original_question=question,
                was_decomposed=False,
            )

        # 질문 분해
        sub_questions = self.decomposer.decompose(question)
        if len(sub_questions) <= 1:
            return DelegationResult(
                original_question=question,
                was_decomposed=False,
            )

        print(f"   🔀 질문 분해: {len(sub_questions)}개 서브태스크")

        # 서브태스크 생성
        tasks = [
            SubTask(
                task_id=f"sub_{i}",
                question=sq,
                model_key=model_key,
            )
            for i, sq in enumerate(sub_questions)
        ]

        # 병렬 실행
        sub_results = self._execute_parallel(tasks, **answer_kwargs)

        # 결과 종합
        synthesized = self._synthesize_results(question, sub_results, model_key)

        total_elapsed = (time.perf_counter() - total_start) * 1000

        return DelegationResult(
            original_question=question,
            sub_results=sub_results,
            synthesized_answer=synthesized,
            total_elapsed_ms=total_elapsed,
            was_decomposed=True,
        )

    def _execute_parallel(self, tasks: List[SubTask], **answer_kwargs) -> List[SubTaskResult]:
        """서브태스크를 병렬 실행합니다."""
        results: List[SubTaskResult] = []

        def run_task(task: SubTask) -> SubTaskResult:
            start = time.perf_counter()
            try:
                # 서브에이전트 실행 — 기본 answer_fn 호출
                answer = self.answer_fn(
                    question=task.question,
                    model_key=task.model_key,
                    use_query_rewrite=True,
                    **answer_kwargs,
                )
                elapsed = (time.perf_counter() - start) * 1000
                return SubTaskResult(
                    task_id=task.task_id,
                    question=task.question,
                    answer=answer if isinstance(answer, str) else str(answer),
                    elapsed_ms=elapsed,
                    model_key=task.model_key,
                )
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                return SubTaskResult(
                    task_id=task.task_id,
                    question=task.question,
                    answer="",
                    elapsed_ms=elapsed,
                    model_key=task.model_key,
                    error=str(e),
                )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(run_task, task): task for task in tasks}
            for future in as_completed(futures, timeout=self.timeout):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    task = futures[future]
                    results.append(SubTaskResult(
                        task_id=task.task_id,
                        question=task.question,
                        answer="",
                        error=str(e),
                    ))

        # task_id 순 정렬
        results.sort(key=lambda r: r.task_id)
        return results

    def _synthesize_results(
        self,
        original_question: str,
        sub_results: List[SubTaskResult],
        model_key: Optional[str] = None,
    ) -> str:
        """서브에이전트 결과를 종합하여 최종 답변을 생성합니다."""
        valid_results = [r for r in sub_results if r.answer and not r.error]

        if not valid_results:
            return "서브에이전트의 결과를 얻지 못했습니다."

        if len(valid_results) == 1:
            return valid_results[0].answer

        sub_answers = "\n\n".join(
            f"[하위 질문 {i + 1}: {r.question}]\n{r.answer}"
            for i, r in enumerate(valid_results)
        )

        synthesis_prompt = (
            f"다음은 원본 질문에 대한 하위 질문별 답변입니다.\n"
            f"이를 종합하여 원본 질문에 대한 하나의 완성된 답변을 작성하세요.\n"
            f"각 하위 답변의 출처 정보를 보존하세요.\n\n"
            f"원본 질문: {original_question}\n\n"
            f"하위 답변:\n{sub_answers}"
        )

        return self.model_manager.get_completion(
            prompt=synthesis_prompt,
            model_key=model_key,
            system_message="여러 정보를 종합하여 하나의 완성된 답변을 작성하는 전문가.",
            temperature=0.3,
        )
