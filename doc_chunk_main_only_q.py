import os
import hashlib
import json
from azure_korean_doc_framework.parsing.parser import HybridDocumentParser
from azure_korean_doc_framework.parsing.chunker import KoreanSemanticChunker
from azure_korean_doc_framework.core.vector_store import VectorStore
from azure_korean_doc_framework.core.agent import KoreanDocAgent
from azure_korean_doc_framework.config import Config


def main():
    # 0. 환경 변수 체크
    try:
        Config.validate()
    except Exception as e:
        print(e)
        return

    # 1. 구성 요소 초기화
    parser = HybridDocumentParser()
    chunker = KoreanSemanticChunker()
    vector_store = VectorStore()

    # 3. 멀티 모델 Q&A 테스트 단계
    agent = KoreanDocAgent()
    question = "겨울철 눈건강"

    # 테스트할 모델 목록 (Config.MODELS에 정의된 키값)
    models_to_test = ["gpt-5.2"]

    print("\n--- [2단계: 멀티 모델 Q&A 테스트] ---")
    print(f"질문: {question}")

    for model in models_to_test:
        print(f"\n--- 모델: {model} ---")
        answer, contexts = agent.answer_question(question, model_key=model, return_context=True)
        print(f"답변:\n{answer}")

        print("\n🔎 [검색 결과 Top 5]")
        for idx, ctx in enumerate(contexts):
            print(f"--- Document {idx+1} ---")
            print(ctx)
            print("-----------------------")

if __name__ == "__main__":
    main()
