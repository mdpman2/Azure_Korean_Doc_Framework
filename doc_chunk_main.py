import os
import argparse
import glob
import hashlib
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from azure_korean_doc_framework.parsing.parser import HybridDocumentParser
from azure_korean_doc_framework.parsing.chunker import KoreanSemanticChunker
from azure_korean_doc_framework.core.vector_store import VectorStore
from azure_korean_doc_framework.core.agent import KoreanDocAgent
from azure_korean_doc_framework.config import Config
from azure_korean_doc_framework.utils.logger import ChunkLogger

def calculate_file_hash(file_path: str) -> str:
    """파일의 SHA256 해시를 계산하여 내용 변경 여부를 정확히 판단합니다."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def process_single_file(
    file_path: str,
    parser: HybridDocumentParser,
    chunker: KoreanSemanticChunker,
    vector_store: VectorStore
) -> str:
    """
    단일 파일을 파싱, 청킹, 로깅 및 업로드합니다.
    (병렬 처리를 위한 단위 함수)
    """
    filename = os.path.basename(file_path)
    try:
        # 1. 변경 감지
        file_mod_time = os.path.getmtime(file_path)
        file_hash = calculate_file_hash(file_path)

        if vector_store.is_file_up_to_date(filename, file_mod_time, file_hash=file_hash):
             return f"⏩ [SKIPPED] {filename} (최신 상태)"

        # 2. 파싱 및 청킹
        print(f"🔄 [START] {filename}: 파일 변경 감지. 처리를 시작합니다...")
        vector_store.delete_documents_by_parent_id(filename)

        parsed_segments = parser.parse(file_path)

        extra_meta = {
            "source": filename,
            "last_modified": file_mod_time,
            "content_hash": file_hash
        }

        chunks = chunker.chunk(parsed_segments, filename=filename, extra_metadata=extra_meta)

        # 3. JSON 로깅 (ChunkLogger 사용)
        ChunkLogger.save_chunks_to_json(chunks, filename)

        # 4. 벡터 저장소 업로드
        vector_store.upload_documents(chunks)
        return f"✅ [SUCCESS] {filename}: {len(chunks)}개 청크 업로드 완료"

    except Exception as e:
        return f"❌ [ERROR] {filename}: {str(e)}"

def process_documents(
    target_path: str,
    parser: HybridDocumentParser,
    chunker: KoreanSemanticChunker,
    vector_store: VectorStore,
    max_workers: int = 3
):
    """
    지정된 경로의 문서를 병렬로 처리합니다.
    """
    if not os.path.exists(target_path):
        print(f"\nℹ️ 문서 수집 생략: '{target_path}'를 찾을 수 없습니다.")
        return

    # 인덱스는 VectorStore 초기화 시 자동으로 생성됨 (create_index_if_not_exists)

    if os.path.isdir(target_path):
        print(f"\n--- [1단계: 문서 수집 - {target_path} 디렉토리 (병렬 모드)] ---")
        files_to_process = [os.path.join(target_path, f) for f in os.listdir(target_path) if f.lower().endswith('.pdf')]
    else:
        print(f"\n--- [1단계: 문서 수집 - {target_path} 파일] ---")
        files_to_process = [target_path] if target_path.lower().endswith('.pdf') else []

    if not files_to_process:
        print(f"ℹ️ 처리할 PDF 파일이 없습니다. (대상: {target_path})")
        return

    print(f"🚀 총 {len(files_to_process)}개의 파일을 처리합니다. (병렬 작업 수: {max_workers})")

    # ThreadPoolExecutor를 사용한 병렬 처리
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_single_file, f, parser, chunker, vector_store): f for f in files_to_process}
        for future in as_completed(future_to_file):
            res = future.result()
            print(f"   > {res}")
            results.append(res)

    print(f"\n✅ 수집 완료 요약: 총 {len(files_to_process)}개 파일 중 {len([r for r in results if 'SUCCESS' in r])}개 성공, {len([r for r in results if 'SKIPPED' in r])}개 건너뜀")

def perform_qa_test(question: str, models: List[str]):
    """멀티 모델 Q&A 테스트를 수행합니다."""
    agent = KoreanDocAgent()

    print("\n--- [2단계: 멀티 모델 Q&A 테스트] ---")
    print(f"질문: {question}")

    for model in models:
        print(f"\n--- 모델: {model} ---")
        answer = agent.answer_question(question, model_key=model, top_k=5)
        print(f"답변:\n{answer}")

def main():
    print("🌟 Azure Korean Document Understanding & Retrieval Framework 🌟")

    # 명령줄 인자 파싱
    arg_parser = argparse.ArgumentParser(
        description="Azure Korean Document Understanding & Retrieval Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 단일 파일 ingest
  python doc_chunk_main.py --path "RAG_TEST_DATA/sample.pdf"

  # 디렉토리 내 모든 PDF ingest
  python doc_chunk_main.py --path "RAG_TEST_DATA"

  # ingest만 수행 (Q&A 테스트 생략)
  python doc_chunk_main.py --path "RAG_TEST_DATA" --skip-qa

  # 특정 질문으로 Q&A 테스트
  python doc_chunk_main.py --question "질문 내용"
        """
    )
    arg_parser.add_argument(
        "-p", "--path",
        type=str,
        help="Ingest할 파일 또는 디렉토리 경로 (여러 개 지정 가능)",
        action="append",
        default=[]
    )
    arg_parser.add_argument(
        "-q", "--question",
        type=str,
        default="바이오주 주가 급락에 따른 셀트리온의 주가 변동률과, 현대차, 삼성전자, 신한지주의 상승률, 그리고 POSCO와 LG화학의 하락률을 각각 말해주세요.",
        help="Q&A 테스트에 사용할 질문"
    )
    arg_parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="Q&A 테스트를 건너뜁니다"
    )
    arg_parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="문서 Ingest를 건너뜁니다 (Q&A만 수행)"
    )
    arg_parser.add_argument(
        "-w", "--workers",
        type=int,
        default=3,
        help="병렬 처리 작업 수 (기본값: 3)"
    )
    arg_parser.add_argument(
        "-m", "--model",
        type=str,
        default="gpt-5.2",
        help="Q&A에 사용할 모델 (기본값: gpt-5.2)"
    )

    args = arg_parser.parse_args()

    # 0. 환경 변수 체크
    try:
        Config.validate()
    except Exception as e:
        print(e)
        return

    # 1. 구성 요소 초기화
    doc_parser = HybridDocumentParser()
    chunker = KoreanSemanticChunker()
    vector_store = VectorStore()

    # 2. 문서 수집 (Ingestion)
    if not args.skip_ingest:
        # 경로가 지정되지 않은 경우 기본 경로 사용
        target_paths = args.path if args.path else [r"RAG_TEST_DATA"]

        for target_path in target_paths:
            # glob 패턴 처리 (예: RAG_TEST_DATA/*.pdf)
            if "*" in target_path or "?" in target_path:
                matched_paths = glob.glob(target_path)
                for matched_path in matched_paths:
                    if os.path.exists(matched_path):
                        process_documents(matched_path, doc_parser, chunker, vector_store, max_workers=args.workers)
            elif os.path.exists(target_path):
                process_documents(target_path, doc_parser, chunker, vector_store, max_workers=args.workers)
            else:
                print(f"⚠️ 경로를 찾을 수 없습니다: {target_path}")

    # 3. Q&A 테스트
    if not args.skip_qa:
        models_to_test = [args.model]
        perform_qa_test(args.question, models_to_test)

if __name__ == "__main__":
    main()
