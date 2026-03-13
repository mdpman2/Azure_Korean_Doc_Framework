#!/usr/bin/env python
"""Batch quality evaluation runner for the Azure Korean document RAG pipeline.

Supports JSON/TSV datasets, runs live answers through KoreanDocAgent, and writes
per-question scores plus a summary JSON file.
"""

import argparse
import json
import os
from statistics import mean
from typing import List, Dict, Any

from azure_korean_doc_framework.config import Config
from azure_korean_doc_framework.core.agent import KoreanDocAgent
from azure_korean_doc_framework.core.multi_model_manager import MultiModelManager


JUDGE_SYSTEM_PROMPT = (
    "당신은 RAG 시스템 답변 품질 평가관입니다. "
    "질문, 정답, 모델 답변을 비교하여 정확성과 충실도를 0~100으로 평가하세요."
)


def evaluate_item(manager: MultiModelManager, question: str, ground_truth: str, answer: str) -> Dict[str, Any]:
    prompt = (
        "다음 형식으로만 평가 결과를 반환하세요.\n"
        "score: 0~100\n"
        "reason: 한 줄 요약\n\n"
        f"[질문]\n{question}\n\n"
        f"[정답]\n{ground_truth}\n\n"
        f"[모델 답변]\n{answer}"
    )
    response = manager.get_completion(
        prompt=prompt,
        model_key=Config.EVALUATION_JUDGE_MODEL,
        system_message=JUDGE_SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=200,
    )

    score = 0.0
    reason = response.strip()
    for line in response.splitlines():
        line = line.strip()
        if line.lower().startswith("score"):
            try:
                score = float(line.split(":", 1)[-1].strip())
            except ValueError:
                score = 0.0
        elif line.lower().startswith("reason"):
            reason = line.split(":", 1)[-1].strip()
    return {"score": score, "reason": reason}


def load_dataset(dataset_path: str) -> List[Dict[str, str]]:
    with open(dataset_path, "r", encoding="utf-8") as handle:
        if dataset_path.lower().endswith(".json"):
            data = json.load(handle)
            return data["items"] if isinstance(data, dict) and "items" in data else data

        items = []
        for line in handle:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            question, answer = line.split("\t", 1)
            items.append({"question": question, "ground_truth": answer})
        return items


def main():
    parser = argparse.ArgumentParser(description="Run RAG quality evaluation against a dataset.")
    parser.add_argument("--dataset", required=True, help="JSON or TSV dataset path")
    parser.add_argument("--output", default="output/evaluation_results.json", help="Where to write results")
    parser.add_argument("--model", default=Config.DEFAULT_MODEL, help="Answer generation model")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    Config.validate()
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dataset = load_dataset(args.dataset)
    agent = KoreanDocAgent(model_key=args.model)
    judge = MultiModelManager(default_model=Config.EVALUATION_JUDGE_MODEL)

    results = []
    for item in dataset:
        question = item["question"]
        ground_truth = item["ground_truth"]
        answer, contexts = agent.answer_question(question, model_key=args.model, return_context=True, top_k=args.top_k)
        evaluation = evaluate_item(judge, question, ground_truth, answer)
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "contexts": contexts,
            "score": evaluation["score"],
            "reason": evaluation["reason"],
        })
        print(f"- {question[:50]}... -> {evaluation['score']:.1f}")

    summary = {
        "items": results,
        "average_score": mean([item["score"] for item in results]) if results else 0.0,
        "count": len(results),
        "answer_model": args.model,
        "judge_model": Config.EVALUATION_JUDGE_MODEL,
    }

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"\n✅ 평가 완료: 평균 {summary['average_score']:.1f} / 100")
    print(f"📁 결과 저장: {args.output}")


if __name__ == "__main__":
    main()