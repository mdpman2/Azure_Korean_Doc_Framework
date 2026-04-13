"""
azure_korean_doc_framework v4.7 통합 CLI 실행 스크립트

문서 파싱 → 청킹 → Contextual Retrieval → 인덱싱 → Q&A 테스트를 하나의 CLI로 실행합니다.
기존 ingestion/Q&A 흐름에 더해 doctor/status, JSON 출력, 세션 저장/복원,
런타임 Azure AI Search 스키마 자동 보정까지 포함한 운영용 진입점입니다.

주요 기능:
- Contextual Retrieval + Hybrid Search + Guardrails 통합 실행
- Graph RAG / 구조화 엔티티 추출 선택 실행
- `--doctor`, `--status`, `--output-format json` 운영 점검 명령
- `--save-session`, `--resume-session` 세션 저장/복원

[v4.7] EdgeQuake 참조 강화:
- Gleaning (Multi-Pass 추출): GRAPH_GLEANING_PASSES 설정으로 추가 패스 횟수 지정
- Mix Query Mode: GRAPH_MIX_WEIGHT로 벡터+그래프 가중 결합
- Knowledge Injection: GRAPH_INJECTION_FILE로 도메인 용어집 자동 로딩

Usage:
    python doc_chunk_main.py --path "RAG_TEST_DATA"
    python doc_chunk_main.py --path "data" --graph-rag --extract-entities
    python doc_chunk_main.py --path "data" --graph-rag --graph-mode mix
    python doc_chunk_main.py --skip-ingest -q "질문" --model gpt-5.4
    python doc_chunk_main.py --doctor --output-format json
"""

import os
import json
import argparse
import glob
import hashlib
import io
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from azure_korean_doc_framework.parsing.parser import HybridDocumentParser
from azure_korean_doc_framework.parsing.chunker import KoreanSemanticChunker
from azure_korean_doc_framework.core.vector_store import VectorStore
from azure_korean_doc_framework.core.agent import KoreanDocAgent
from azure_korean_doc_framework.config import Config
from azure_korean_doc_framework.utils.azure_clients import AzureClientFactory
from azure_korean_doc_framework.utils.logger import ChunkLogger
from azure_korean_doc_framework.utils.search_schema import apply_search_runtime_mapping, get_search_runtime_mapping

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
DEFAULT_QUESTION = "바이오주 주가 급락에 따른 셀트리온의 주가 변동률과, 현대차, 삼성전자, 신한지주의 상승률, 그리고 POSCO와 LG화학의 하락률을 각각 말해주세요."
DEFAULT_TOP_K = 5


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _resolve_output_dir(base_dir: Optional[str] = None) -> str:
    root = base_dir or SCRIPT_ROOT
    return os.path.join(root, "output")


def _resolve_session_dir(base_dir: Optional[str] = None) -> str:
    return os.path.join(_resolve_output_dir(base_dir), "sessions")


def _latest_session_pointer(base_dir: Optional[str] = None) -> str:
    return os.path.join(_resolve_session_dir(base_dir), "latest_session.txt")


def _ensure_directory(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _write_json_file(path: str, payload: Dict[str, Any]) -> None:
    _ensure_directory(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_serializable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _generate_session_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"session-{timestamp}-{uuid.uuid4().hex[:8]}"


def _store_latest_session_id(session_id: str, base_dir: Optional[str] = None) -> None:
    pointer_path = _latest_session_pointer(base_dir)
    _ensure_directory(os.path.dirname(pointer_path) or ".")
    with open(pointer_path, "w", encoding="utf-8") as f:
        f.write(session_id)


def _get_latest_session_id(base_dir: Optional[str] = None) -> Optional[str]:
    pointer_path = _latest_session_pointer(base_dir)
    if not os.path.exists(pointer_path):
        return None
    with open(pointer_path, "r", encoding="utf-8") as f:
        value = f.read().strip()
    return value or None


def load_session_record(session_ref: str, base_dir: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    """Load a saved session by explicit id/path or by the `latest` pointer."""
    session_dir = _resolve_session_dir(base_dir)
    resolved_ref = session_ref
    if session_ref == "latest":
        resolved_ref = _get_latest_session_id(base_dir) or ""
    if not resolved_ref:
        raise FileNotFoundError("복원할 세션이 없습니다. 먼저 --save-session 으로 세션을 저장하세요.")

    session_path = resolved_ref
    if not session_path.endswith(".json"):
        # [v4.7-fix] 경로 순회 방지: 세션 ID에 디렉토리 구분자 차단
        safe_id = os.path.basename(resolved_ref)
        session_path = os.path.join(session_dir, f"{safe_id}.json")

    if not os.path.exists(session_path):
        raise FileNotFoundError(f"세션 파일을 찾을 수 없습니다: {session_path}")
    return _read_json_file(session_path), session_path


def save_session_record(
    *,
    request_payload: Dict[str, Any],
    response_payload: Dict[str, Any],
    base_dir: Optional[str] = None,
    session_id: Optional[str] = None,
    existing_session: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], str]:
    """Append the current Q&A run to a persistent session file under output/sessions."""
    now = _utc_now_iso()
    session = dict(existing_session or {})
    session_identifier = session.get("session_id") or session_id or _generate_session_id()
    session.setdefault("session_id", session_identifier)
    session.setdefault("created_at", now)
    session["updated_at"] = now
    session.setdefault("runs", [])

    run_payload = {
        "run_id": f"run-{uuid.uuid4().hex[:8]}",
        "timestamp": now,
        "request": _to_serializable(request_payload),
        "response": _to_serializable(response_payload),
    }
    session["runs"].append(run_payload)
    session["run_count"] = len(session["runs"])
    session["last_request"] = _to_serializable(request_payload)
    session["last_response"] = _to_serializable(response_payload)

    session_dir = _ensure_directory(_resolve_session_dir(base_dir))
    session_path = os.path.join(session_dir, f"{session_identifier}.json")
    _write_json_file(session_path, session)
    _store_latest_session_id(session_identifier, base_dir)
    return session, session_path


def build_doctor_report(
    *,
    require_openai: bool,
    require_search: bool,
    require_di: bool,
    base_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a preflight report for env vars, Azure clients, output paths, and runtime index mapping."""
    checks: List[Dict[str, Any]] = []
    resolved_mapping = get_search_runtime_mapping(refresh=True)

    def add_check(name: str, passed: bool, detail: str, required: bool = True) -> None:
        checks.append({
            "name": name,
            "passed": passed,
            "detail": detail,
            "required": required,
        })

    api_key, endpoint, api_version = Config.get_openai_credentials(prefer_advanced=True)
    add_check("openai_api_key", bool(api_key), "configured" if api_key else "missing", require_openai)
    add_check("openai_endpoint", bool(endpoint), endpoint or "missing", require_openai)
    add_check("openai_api_version", bool(api_version), api_version, require_openai)
    add_check("search_key", bool(Config.SEARCH_KEY), "configured" if Config.SEARCH_KEY else "missing", require_search)
    add_check("search_endpoint", bool(Config.SEARCH_ENDPOINT), Config.SEARCH_ENDPOINT or "missing", require_search)
    add_check("search_index_name", bool(Config.SEARCH_INDEX_NAME), Config.SEARCH_INDEX_NAME or "missing", require_search)
    add_check("document_intelligence_key", bool(Config.DI_KEY), "configured" if Config.DI_KEY else "missing", require_di)
    add_check("document_intelligence_endpoint", bool(Config.DI_ENDPOINT), Config.DI_ENDPOINT or "missing", require_di)

    try:
        Config.validate(
            require_openai=require_openai,
            require_search=require_search,
            require_di=require_di,
        )
        add_check("config_validate", True, "required configuration present")
    except Exception as exc:
        add_check("config_validate", False, str(exc).strip())

    output_dir = _ensure_directory(_resolve_output_dir(base_dir))
    session_dir = _ensure_directory(_resolve_session_dir(base_dir))
    add_check("output_directory", os.path.isdir(output_dir), output_dir)
    add_check("session_directory", os.path.isdir(session_dir), session_dir)

    latest_session_id = _get_latest_session_id(base_dir)
    add_check(
        "latest_session_pointer",
        True,
        latest_session_id or "not created yet",
        required=False,
    )

    if require_openai and api_key and endpoint:
        try:
            AzureClientFactory.get_openai_client(is_advanced=False)
            add_check("openai_client_factory", True, "standard Azure OpenAI client ready")
        except Exception as exc:
            add_check("openai_client_factory", False, str(exc))

    if require_search and Config.SEARCH_KEY and Config.SEARCH_ENDPOINT and Config.SEARCH_INDEX_NAME:
        try:
            apply_search_runtime_mapping(refresh=True)
            AzureClientFactory.get_search_client()
            add_check("search_client_factory", True, f"index={Config.SEARCH_INDEX_NAME}")
        except Exception as exc:
            add_check("search_client_factory", False, str(exc))

    if require_di and Config.DI_KEY and Config.DI_ENDPOINT:
        try:
            AzureClientFactory.get_di_client()
            add_check("document_intelligence_client_factory", True, "document intelligence client ready")
        except Exception as exc:
            add_check("document_intelligence_client_factory", False, str(exc))

    add_check("graph_rag_dependency", HAS_GRAPH_RAG, "networkx available" if HAS_GRAPH_RAG else "networkx not installed", required=False)
    add_check(
        "entity_extractor_dependency",
        HAS_ENTITY_EXTRACTOR,
        "structured entity extractor available" if HAS_ENTITY_EXTRACTOR else "entity extractor dependency unavailable",
        required=False,
    )

    ok = all(item["passed"] for item in checks if item["required"])
    return {
        "ok": ok,
        "checked_at": _utc_now_iso(),
        "requirements": {
            "openai": require_openai,
            "search": require_search,
            "document_intelligence": require_di,
        },
        "search_runtime_mapping": resolved_mapping,
        "checks": checks,
    }


def build_status_report(base_dir: Optional[str] = None) -> Dict[str, Any]:
    """Summarize current workspace artifacts, session history, feature flags, and runtime search mapping."""
    resolved_mapping = get_search_runtime_mapping(refresh=True)
    apply_search_runtime_mapping(refresh=True)
    output_dir = _resolve_output_dir(base_dir)
    session_dir = _resolve_session_dir(base_dir)
    chunk_files = sorted(glob.glob(os.path.join(output_dir, "*_chunks.json")))
    entity_files = sorted(glob.glob(os.path.join(output_dir, "*_entities.json")))
    graph_path = Config.GRAPH_STORAGE_PATH
    if not os.path.isabs(graph_path):
        graph_path = os.path.join(SCRIPT_ROOT, graph_path)

    latest_session_summary = None
    latest_session_id = _get_latest_session_id(base_dir)
    if latest_session_id:
        try:
            session, session_path = load_session_record(latest_session_id, base_dir=base_dir)
            last_request = session.get("last_request", {})
            latest_session_summary = {
                "session_id": session.get("session_id"),
                "path": session_path,
                "updated_at": session.get("updated_at"),
                "run_count": session.get("run_count", 0),
                "last_question": last_request.get("question"),
                "last_mode": last_request.get("qa_mode"),
            }
        except Exception as exc:
            latest_session_summary = {
                "session_id": latest_session_id,
                "error": str(exc),
            }

    session_files = []
    if os.path.isdir(session_dir):
        session_files = sorted(
            [path for path in glob.glob(os.path.join(session_dir, "*.json")) if os.path.basename(path) != "latest_session.txt"],
            key=os.path.getmtime,
            reverse=True,
        )

    return {
        "generated_at": _utc_now_iso(),
        "workspace_root": WORKSPACE_ROOT,
        "script_root": SCRIPT_ROOT,
        "output_dir": output_dir,
        "search_index_name": Config.SEARCH_INDEX_NAME,
        "default_model": Config.DEFAULT_MODEL,
        "features": {
            "graph_rag_enabled": Config.GRAPH_RAG_ENABLED,
            "contextual_retrieval_enabled": Config.CONTEXTUAL_RETRIEVAL_ENABLED,
            "query_rewrite_enabled": Config.QUERY_REWRITE_ENABLED,
            "diagnostics_enabled": Config.ANSWER_DIAGNOSTICS_ENABLED,
            "retrieval_gate_enabled": Config.RETRIEVAL_GATE_ENABLED,
        },
        "artifacts": {
            "chunk_file_count": len(chunk_files),
            "entity_file_count": len(entity_files),
            "knowledge_graph_path": graph_path,
            "knowledge_graph_exists": os.path.exists(graph_path),
        },
        "sessions": {
            "session_dir": session_dir,
            "session_count": len(session_files),
            "latest": latest_session_summary,
        },
        "environment": {
            "openai_configured": bool(Config.get_openai_credentials(prefer_advanced=True)[0] and Config.get_openai_credentials(prefer_advanced=True)[1]),
            "search_configured": bool(Config.SEARCH_KEY and Config.SEARCH_ENDPOINT and Config.SEARCH_INDEX_NAME),
            "document_intelligence_configured": bool(Config.DI_KEY and Config.DI_ENDPOINT),
        },
        "search_runtime_mapping": resolved_mapping,
    }


def _print_doctor_report(report: Dict[str, Any]) -> None:
    print("\n--- [Doctor / Preflight] ---")
    for check in report.get("checks", []):
        icon = "✅" if check.get("passed") else ("ℹ️" if not check.get("required") else "❌")
        requirement_label = "required" if check.get("required") else "optional"
        print(f"{icon} {check.get('name')} [{requirement_label}] - {check.get('detail')}")
    print(f"\nDoctor 결과: {'정상' if report.get('ok') else '주의 필요'}")


def _print_status_report(report: Dict[str, Any]) -> None:
    print("\n--- [Status] ---")
    print(f"Workspace: {report.get('workspace_root')}")
    print(f"Index: {report.get('search_index_name') or '(unset)'}")
    print(f"Default model: {report.get('default_model')}")
    artifacts = report.get("artifacts", {})
    print(f"Chunk files: {artifacts.get('chunk_file_count', 0)} | Entity files: {artifacts.get('entity_file_count', 0)}")
    print(
        f"Knowledge graph: {artifacts.get('knowledge_graph_path')} "
        f"({'exists' if artifacts.get('knowledge_graph_exists') else 'missing'})"
    )
    sessions = report.get("sessions", {})
    print(f"Session count: {sessions.get('session_count', 0)}")
    latest = sessions.get("latest")
    if latest:
        print(
            f"Latest session: {latest.get('session_id')} | runs={latest.get('run_count', 0)} "
            f"| updated={latest.get('updated_at')}"
        )
        if latest.get("last_question"):
            print(f"Last question: {latest.get('last_question')}")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser shared by production runs and regression tests."""
    arg_parser = argparse.ArgumentParser(
        description="Azure Korean Document Understanding & Retrieval Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 단일 파일 ingest
  python doc_chunk_main.py --path "RAG_TEST_DATA/sample.pdf"

  # Q&A 결과를 JSON 으로 출력
  python doc_chunk_main.py -q "질문 내용" --skip-ingest --output-format json

  # 세션 저장 및 복원
  python doc_chunk_main.py -q "질문 내용" --skip-ingest --save-session
  python doc_chunk_main.py --resume-session latest --skip-ingest --output-format json

  # Doctor / Status
  python doc_chunk_main.py --doctor
  python doc_chunk_main.py --status --output-format json
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
        default=None,
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
    arg_parser.add_argument(
        "--doctor",
        action="store_true",
        help="실행 전 필수 설정과 런타임 상태를 점검합니다"
    )
    arg_parser.add_argument(
        "--status",
        action="store_true",
        help="현재 인덱스, 산출물, 세션 상태를 출력합니다"
    )
    arg_parser.add_argument(
        "--output-format",
        type=str,
        default="text",
        choices=["text", "json"],
        help="출력 형식 (text/json)"
    )
    arg_parser.add_argument(
        "--save-session",
        action="store_true",
        help="Q&A 실행 결과를 output/sessions 아래에 저장합니다"
    )
    arg_parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="세션 저장 시 사용할 고정 세션 ID"
    )
    arg_parser.add_argument(
        "--resume-session",
        type=str,
        default=None,
        help="이전 세션을 복원합니다. latest 또는 세션 ID 사용 가능"
    )
    return arg_parser


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

        # 4. 벡터 저장소: 기존 문서 삭제 후 업로드
        vector_store.delete_documents_by_parent_id(document_key)
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
        return {
            "target_path": target_path,
            "file_count": 0,
            "success_count": 0,
            "skipped_count": 0,
            "error_count": 0,
            "results": [],
        }

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
            try:
                res = future.result()
            except Exception as e:
                failed_file = future_to_file[future]
                res = f"❌ [ERROR] {failed_file}: {e}"
            print(f"   > {res}")
            results.append(res)

    success_count = len([r for r in results if 'SUCCESS' in r])
    skipped_count = len([r for r in results if 'SKIPPED' in r])
    error_count = len([r for r in results if 'ERROR' in r])
    print(f"\n✅ 수집 완료 요약: 총 {len(files_to_process)}개 파일 중 {success_count}개 성공, {skipped_count}개 건너뜀")
    return {
        "target_path": target_path,
        "file_count": len(files_to_process),
        "success_count": success_count,
        "skipped_count": skipped_count,
        "error_count": error_count,
        "results": results,
    }

def perform_qa_test(question: str, models: List[str]):
    """멀티 모델 Q&A 테스트를 수행합니다."""
    agent = KoreanDocAgent()

    print("\n--- [2단계: 멀티 모델 Q&A 테스트] ---")
    print(f"질문: {question}")

    for model in models:
        print(f"\n--- 모델: {model} ---")
        answer = agent.answer_question(question, model_key=model, top_k=5)
        print(f"답변:\n{answer}")


def _resolve_effective_question(args, resumed_session: Optional[Dict[str, Any]]) -> str:
    if args.question:
        return args.question
    if resumed_session:
        return resumed_session.get("last_request", {}).get("question") or DEFAULT_QUESTION
    return DEFAULT_QUESTION


def _resolve_effective_model(args, resumed_session: Optional[Dict[str, Any]]) -> str:
    if resumed_session and args.model == Config.DEFAULT_MODEL:
        return resumed_session.get("last_request", {}).get("model") or args.model
    return args.model


def _build_session_request_payload(args, question: str, model: str, qa_mode: str) -> Dict[str, Any]:
    return {
        "question": question,
        "model": model,
        "qa_mode": qa_mode,
        "graph_rag": bool(args.graph_rag),
        "graph_mode": args.graph_mode,
        "skip_ingest": bool(args.skip_ingest),
        "skip_qa": bool(args.skip_qa),
        "extract_entities": bool(args.extract_entities),
        "workers": args.workers,
        "paths": args.path,
    }


def _execute_cli(args) -> Dict[str, Any]:
    """Execute the requested CLI mode and return a serializable payload for text or JSON rendering."""
    payload: Dict[str, Any] = {
        "command": "run",
        "generated_at": _utc_now_iso(),
    }

    will_ingest = not args.skip_ingest
    will_run_qa = not args.skip_qa
    will_extract_entities = args.extract_entities and HAS_ENTITY_EXTRACTOR
    will_build_graph = args.graph_rag and HAS_GRAPH_RAG and will_ingest

    if args.doctor:
        payload["command"] = "doctor"
        doctor_report = build_doctor_report(
            require_openai=will_ingest or will_run_qa or will_extract_entities or will_build_graph,
            require_search=will_ingest or will_run_qa,
            require_di=will_ingest,
        )
        payload["doctor"] = doctor_report
        _print_doctor_report(doctor_report)
        return payload

    if args.status:
        payload["command"] = "status"
        status_report = build_status_report()
        payload["status"] = status_report
        _print_status_report(status_report)
        return payload

    resumed_session = None
    resumed_session_path = None
    if args.resume_session:
        resumed_session, resumed_session_path = load_session_record(args.resume_session)
        print(f"📂 세션 복원: {resumed_session.get('session_id')} ({resumed_session_path})")
        payload["resumed_session"] = {
            "session_id": resumed_session.get("session_id"),
            "path": resumed_session_path,
            "run_count": resumed_session.get("run_count", 0),
        }

    effective_question = _resolve_effective_question(args, resumed_session)
    effective_model = _resolve_effective_model(args, resumed_session)

    try:
        Config.validate(
            require_openai=will_ingest or will_run_qa or will_extract_entities or will_build_graph,
            require_search=will_ingest or will_run_qa,
            require_di=will_ingest,
        )
    except Exception as e:
        payload["error"] = str(e)
        print(e)
        return payload

    doc_parser = None
    chunker = None
    vector_store = None
    if will_ingest:
        doc_parser = HybridDocumentParser()
        chunker = KoreanSemanticChunker()
        vector_store = VectorStore()

    ingest_summaries: List[Dict[str, Any]] = []
    if will_ingest:
        target_paths = args.path if args.path else [r"RAG_TEST_DATA"]
        for target_path in target_paths:
            if "*" in target_path or "?" in target_path:
                matched_paths = glob.glob(target_path)
                for matched_path in matched_paths:
                    if os.path.exists(matched_path):
                        ingest_summaries.append(
                            process_documents(matched_path, doc_parser, chunker, vector_store, max_workers=args.workers)
                        )
            elif os.path.exists(target_path):
                ingest_summaries.append(
                    process_documents(target_path, doc_parser, chunker, vector_store, max_workers=args.workers)
                )
            else:
                print(f"⚠️ 경로를 찾을 수 없습니다: {target_path}")
                ingest_summaries.append({
                    "target_path": target_path,
                    "file_count": 0,
                    "success_count": 0,
                    "skipped_count": 0,
                    "error_count": 1,
                    "results": [f"❌ [ERROR] path not found: {target_path}"],
                })
    payload["ingest"] = ingest_summaries

    graph_manager = None
    graph_summary: Dict[str, Any] = {"enabled": False}
    if args.graph_rag and HAS_GRAPH_RAG:
        print("\n--- [Graph RAG: Knowledge Graph 구축] ---")
        # [v5.1-fix] llm_cache를 전달하여 엔티티 추출 캐시 활용
        _llm_cache = None
        if Config.LLM_CACHE_ENABLED:
            from azure_korean_doc_framework.core.llm_cache import LLMResponseCache
            _llm_cache = LLMResponseCache(
                cache_dir=Config.LLM_CACHE_DIR,
                max_memory_entries=Config.LLM_CACHE_MAX_MEMORY,
                default_ttl=Config.LLM_CACHE_TTL,
                enabled=True,
            )
        graph_manager = KnowledgeGraphManager(
            model_key=effective_model,
            gleaning_passes=Config.GRAPH_GLEANING_PASSES,
            mix_graph_weight=Config.GRAPH_MIX_WEIGHT,
            llm_cache=_llm_cache,
        )
        graph_path = args.graph_save
        if os.path.exists(graph_path):
            graph_manager.load_graph(graph_path)
            print("📁 기존 Knowledge Graph 로드 완료")

        # [v4.7] Knowledge Injection 파일 로드 (EdgeQuake 참조)
        injection_file = Config.GRAPH_INJECTION_FILE
        if injection_file and os.path.exists(injection_file):
            with open(injection_file, "r", encoding="utf-8") as f:
                injection_text = f.read()
            injected = graph_manager.inject_from_text(injection_text)
            print(f"💉 Knowledge Injection: {injected}개 용어 주입 ({injection_file})")

        extracted_chunk_count = 0
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

                extracted_chunk_count = len(all_chunk_texts)
                if all_chunk_texts:
                    print(f"🔍 {len(all_chunk_texts)}개 텍스트 청크에서 엔티티/관계 추출 중...")
                    graph_manager.extract_from_chunks(
                        all_chunk_texts,
                        batch_size=Config.GRAPH_ENTITY_BATCH_SIZE,
                    )
                    os.makedirs(os.path.dirname(graph_path) or '.', exist_ok=True)
                    graph_manager.save_graph(graph_path)

        stats = graph_manager.get_stats()
        print(f"📊 Knowledge Graph 통계: 노드 {stats['nodes']}개, 엣지 {stats['edges']}개, "
              f"커뮤니티 {stats.get('communities', 0)}개, 주입 {stats.get('injections', 0)}개")
        if stats.get('entity_types'):
            for et, count in stats['entity_types'].items():
                print(f"   - {et}: {count}개")
        graph_summary = {
            "enabled": True,
            "path": graph_path,
            "stats": stats,
            "extracted_chunk_count": extracted_chunk_count,
        }
    payload["graph"] = graph_summary

    entity_summary: List[Dict[str, Any]] = []
    if args.extract_entities and HAS_ENTITY_EXTRACTOR:
        print("\n--- [v4.0: 구조화 엔티티 추출 (LangExtract 기반)] ---")
        extractor = StructuredEntityExtractor(
            model_key=effective_model,
            extraction_passes=Config.EXTRACTION_PASSES,
            max_chunk_chars=Config.EXTRACTION_MAX_CHUNK_CHARS,
            max_workers=Config.EXTRACTION_MAX_WORKERS,
        )

        chunk_files = glob.glob("output/*_chunks.json")
        for cf in chunk_files:
            with open(cf, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)

            texts = [c.get('page_content', '') for c in chunks_data if c.get('page_content')]
            full_text = "\n\n".join(texts[:20])

            result = extractor.extract(full_text)
            output_path = cf.replace('_chunks.json', '_entities.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(extractor.extractions_to_dict(result), f, ensure_ascii=False, indent=2)
            print(f"   ✅ {os.path.basename(cf)}: {len(result.extractions)}개 엔티티 추출 → {output_path}")
            entity_summary.append({
                "chunk_file": cf,
                "output_path": output_path,
                "extraction_count": len(result.extractions),
            })
    payload["entity_extraction"] = entity_summary

    qa_payload = None
    if will_run_qa:
        if graph_manager and graph_manager.graph.number_of_nodes() > 0:
            print("\n--- [2단계: Graph-Enhanced Q&A 테스트] ---")
            agent = KoreanDocAgent(graph_manager=graph_manager)
            print(f"질문: {effective_question}")
            print(f"\n--- 모델: {effective_model} (Graph-Enhanced, mode={args.graph_mode}) ---")
            artifacts = agent.graph_enhanced_answer(
                effective_question,
                model_key=effective_model,
                top_k=DEFAULT_TOP_K,
                graph_query_mode=args.graph_mode,
                return_artifacts=True,
            )
            print(f"답변:\n{artifacts.answer}")
            qa_mode = "graph_enhanced"
        else:
            print("\n--- [2단계: 멀티 모델 Q&A 테스트] ---")
            print(f"질문: {effective_question}")
            print(f"\n--- 모델: {effective_model} ---")
            agent = KoreanDocAgent()
            artifacts = agent.answer_question(
                effective_question,
                model_key=effective_model,
                top_k=DEFAULT_TOP_K,
                return_artifacts=True,
            )
            print(f"답변:\n{artifacts.answer}")
            qa_mode = "standard"

        qa_payload = {
            "question": effective_question,
            "model": effective_model,
            "mode": qa_mode,
            "artifacts": _to_serializable(artifacts),
        }

        if args.save_session or args.resume_session:
            request_payload = _build_session_request_payload(args, effective_question, effective_model, qa_mode)
            response_payload = {
                "answer": artifacts.answer,
                "gate_reason": artifacts.gate_reason,
                "diagnostics": _to_serializable(artifacts.diagnostics),
                "steps": _to_serializable(artifacts.steps),
                "search_results": _to_serializable(artifacts.search_results),
            }
            session, session_path = save_session_record(
                request_payload=request_payload,
                response_payload=response_payload,
                session_id=args.session_id,
                existing_session=resumed_session,
            )
            print(f"💾 세션 저장 완료: {session_path}")
            qa_payload["session"] = {
                "session_id": session.get("session_id"),
                "path": session_path,
                "run_count": session.get("run_count", 0),
            }
    payload["qa"] = qa_payload
    return payload

def main():
    arg_parser = _build_arg_parser()
    args = arg_parser.parse_args()

    if args.output_format == "json":
        stdout_buffer = io.StringIO()
        with redirect_stdout(stdout_buffer):
            payload = _execute_cli(args)
        payload["logs"] = [line for line in stdout_buffer.getvalue().splitlines() if line.strip()]
        print(json.dumps(_to_serializable(payload), ensure_ascii=False, indent=2))
        return

    print("🌟 Azure Korean Document Understanding & Retrieval Framework 🌟")
    _execute_cli(args)

if __name__ == "__main__":
    main()
