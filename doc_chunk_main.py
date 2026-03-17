"""
azure_korean_doc_framework v4.1 통합 CLI 실행 스크립트

문서 파싱 → 청킹 → Contextual Retrieval (맥락 추가) → 인덱싱 → Q&A 테스트를 통합 실행합니다.
변경 감지 인덱싱, 병렬 처리, Graph RAG, 엔티티 추출, Hybrid Search 등을 지원합니다.

[v4.1 업데이트]
- Contextual Retrieval: 청킹 시 모든 청크에 LLM 맥락 자동 추가 (Anthropic 방식)
- Hybrid Search: BM25 키워드 + Vector 유사성 + Semantic Ranking 결합
- 원본/맥락 분리 저장: 검색은 맥락 포함, 답변은 원본 사용

Usage:
  python doc_chunk_main.py --path "RAG_TEST_DATA"
  python doc_chunk_main.py --path "data" --graph-rag --extract-entities
    python doc_chunk_main.py --skip-ingest -q "질문" --model gpt-5.4
"""

import os
import json
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

# v4.0: Graph RAG & 구조화 추출
try:
    from azure_korean_doc_framework.core.graph_rag import KnowledgeGraphManager, QueryMode
    HAS_GRAPH_RAG = True
except ImportError:
    HAS_GRAPH_RAG = False
    print("⚠️ Graph RAG 비활성화 (networkx 패키지 필요: pip install networkx)")

try:
    from azure_korean_doc_framework.parsing.entity_extractor import StructuredEntityExtractor
    HAS_ENTITY_EXTRACTOR = True
except ImportError:
    HAS_ENTITY_EXTRACTOR = False


SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_ROOT)


def _build_document_key(file_path: str) -> str:
    """동일 파일명 충돌을 피하기 위해 워크스페이스 기준 상대 경로를 문서 키로 사용합니다."""
    absolute_path = os.path.abspath(file_path)
    try:
        relative_path = os.path.relpath(absolute_path, start=WORKSPACE_ROOT)
    except ValueError:
        relative_path = absolute_path
    return relative_path.replace("\\", "/")

def calculate_file_hash(file_path: str) -> str:
    """파일의 SHA256 해시를 계산하여 내용 변경 여부를 정확히 판단합니다."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def process_single_file(
    file_path: str,
    document_key: str,
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

        if vector_store.is_file_up_to_date(document_key, file_mod_time, file_hash=file_hash):
             return f"⏩ [SKIPPED] {filename} (최신 상태)"

        # 2. 파싱 및 청킹
        print(f"🔄 [START] {filename}: 파일 변경 감지. 처리를 시작합니다... (key={document_key})")
        vector_store.delete_documents_by_parent_id(document_key)

        parsed_segments = parser.parse(file_path)

        extra_meta = {
            "source": document_key,
            "source_name": filename,
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
        future_to_file = {
            executor.submit(
                process_single_file,
                f,
                _build_document_key(f),
                parser,
                chunker,
                vector_store,
            ): f
            for f in files_to_process
        }
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
        default=Config.DEFAULT_MODEL,
        help=f"Q&A에 사용할 모델 (기본값: {Config.DEFAULT_MODEL})"
    )
    # v4.0: Graph RAG 옵션
    arg_parser.add_argument(
        "--graph-rag",
        action="store_true",
        help="[v4.0] Graph RAG 활성화 (LightRAG 기반 Knowledge Graph 구축 및 검색)"
    )
    arg_parser.add_argument(
        "--graph-mode",
        type=str,
        default="hybrid",
        choices=["local", "global", "hybrid", "naive"],
        help="[v4.0] Graph 검색 모드 (기본값: hybrid)"
    )
    arg_parser.add_argument(
        "--extract-entities",
        action="store_true",
        help="[v4.0] LangExtract 기반 구조화 엔티티 추출 수행"
    )
    arg_parser.add_argument(
        "--graph-save",
        type=str,
        default="output/knowledge_graph.json",
        help="[v4.0] Knowledge Graph 저장 경로"
    )

    args = arg_parser.parse_args()

    will_ingest = not args.skip_ingest
    will_run_qa = not args.skip_qa
    will_extract_entities = args.extract_entities and HAS_ENTITY_EXTRACTOR
    will_build_graph = args.graph_rag and HAS_GRAPH_RAG and will_ingest

    # 0. 실행 모드별 환경 변수 체크
    try:
        Config.validate(
            require_openai=will_ingest or will_run_qa or will_extract_entities or will_build_graph,
            require_search=will_ingest or will_run_qa,
            require_di=will_ingest,
        )
    except Exception as e:
        print(e)
        return

    # 1. 구성 요소 초기화
    doc_parser = None
    chunker = None
    vector_store = None
    if will_ingest:
        doc_parser = HybridDocumentParser()
        chunker = KoreanSemanticChunker()
        vector_store = VectorStore()

    # 2. 문서 수집 (Ingestion)
    if will_ingest:
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

    # 2.5 [v4.0] Graph RAG 구축 (--graph-rag 옵션)
    graph_manager = None
    if args.graph_rag and HAS_GRAPH_RAG:
        print("\n--- [Graph RAG: Knowledge Graph 구축] ---")
        graph_manager = KnowledgeGraphManager(model_key=args.model)

        # 기존 그래프 로드 시도
        graph_path = args.graph_save
        if os.path.exists(graph_path):
            graph_manager.load_graph(graph_path)
            print(f"📁 기존 Knowledge Graph 로드 완료")

        # 새로운 청크가 있으면 그래프 구축
        if will_ingest:
            chunk_files = glob.glob("output/*_chunks.json")
            if chunk_files:
                all_chunk_texts = []
                for cf in chunk_files:
                    with open(cf, 'r', encoding='utf-8') as f:
                        chunks_data = json.load(f)
                    for c in chunks_data:
                        if not c.get('metadata', {}).get('is_table_data'):
                            all_chunk_texts.append({"page_content": c.get('page_content', '')})

                if all_chunk_texts:
                    print(f"🔍 {len(all_chunk_texts)}개 텍스트 청크에서 엔티티/관계 추출 중...")
                    graph_manager.extract_from_chunks(
                        all_chunk_texts,
                        batch_size=Config.GRAPH_ENTITY_BATCH_SIZE,
                    )
                    # 그래프 저장
                    os.makedirs(os.path.dirname(graph_path) or '.', exist_ok=True)
                    graph_manager.save_graph(graph_path)

        stats = graph_manager.get_stats()
        print(f"📊 Knowledge Graph 통계: 노드 {stats['nodes']}개, 엣지 {stats['edges']}개")
        if stats.get('entity_types'):
            for et, count in stats['entity_types'].items():
                print(f"   - {et}: {count}개")

    # 2.6 [v4.0] 구조화 엔티티 추출 (--extract-entities 옵션)
    if args.extract_entities and HAS_ENTITY_EXTRACTOR:
        print("\n--- [v4.0: 구조화 엔티티 추출 (LangExtract 기반)] ---")
        extractor = StructuredEntityExtractor(
            model_key=args.model,
            extraction_passes=Config.EXTRACTION_PASSES,
            max_chunk_chars=Config.EXTRACTION_MAX_CHUNK_CHARS,
            max_workers=Config.EXTRACTION_MAX_WORKERS,
        )

        chunk_files = glob.glob("output/*_chunks.json")
        for cf in chunk_files:
            with open(cf, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)

            texts = [c.get('page_content', '') for c in chunks_data if c.get('page_content')]
            full_text = "\n\n".join(texts[:20])  # 상위 20개 청크만

            result = extractor.extract(full_text)
            output_path = cf.replace('_chunks.json', '_entities.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(extractor.extractions_to_dict(result), f, ensure_ascii=False, indent=2)
            print(f"   ✅ {os.path.basename(cf)}: {len(result.extractions)}개 엔티티 추출 → {output_path}")

    # 3. Q&A 테스트
    if will_run_qa:
        models_to_test = [args.model]

        # v4.0: Graph RAG가 활성화되면 graph_enhanced_answer 사용
        if graph_manager and graph_manager.graph.number_of_nodes() > 0:
            print("\n--- [2단계: Graph-Enhanced Q&A 테스트] ---")
            agent = KoreanDocAgent(graph_manager=graph_manager)
            print(f"질문: {args.question}")
            for model in models_to_test:
                print(f"\n--- 모델: {model} (Graph-Enhanced, mode={args.graph_mode}) ---")
                answer = agent.graph_enhanced_answer(
                    args.question,
                    model_key=model,
                    top_k=5,
                    graph_query_mode=args.graph_mode,
                )
                print(f"답변:\n{answer}")
        else:
            perform_qa_test(args.question, models_to_test)

if __name__ == "__main__":
    main()
