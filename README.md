# Azure Korean Document Understanding & Retrieval Framework

이 프로젝트는 Azure AI Services (Document Intelligence, OpenAI GPT-5.4)를 활용하여 한국어 문서를 정밀하게 분석하고, RAG (Retrieval-Augmented Generation) 시스템을 위한 최적의 청킹(Chunking) 데이터를 생성 및 관리하는 프레임워크입니다.

> **📢 2026-03 문서 정리**: Azure AI Search 필드 매핑, 출처 표기, 시맨틱 설정명이 현재 코드와 실제 검증 결과에 맞게 갱신되었습니다.

> **📢 최신 기능 요약**
> - `v4.1`: Contextual Retrieval + Hybrid Search
> - `v4.2`: Guardrails + Evidence Extraction + Quality Evaluation
> - `v4.3`: 외부 Azure AI Search 인덱스 재사용, 출처 복원, 라이브 재색인 검증
> - `v4.4`: Agent diagnostics + Query Rewrite feature toggle

## 🆕 최근 주요 업데이트

| 버전 | 핵심 추가 내용 | 운영 관점 효과 |
|------|----------------|----------------|
| **v4.1** | Contextual Retrieval, Contextual BM25, Contextual Embeddings, Hybrid Search | 검색 실패율 감소, 원본 텍스트 기반 답변 |
| **v4.2** | Retrieval Gate, Question Classification, Evidence Extraction, Numeric Verification, PII/Injection/Faithfulness/Hallucination Guardrails, Batch Evaluation | 저품질 검색 차단, 규정형 답변 안정화, 운영 안전장치 강화 |
| **v4.3** | Azure AI Search 필드 매핑, semantic config 매핑, evidence 답변 출처 복원, 기존 인덱스 재사용, 라이브 재색인 검증 | 외부 인덱스 호환성 확보, 실제 운영 인덱스에 안전하게 연결 |
| **v4.4** | AnswerArtifacts diagnostics, Query Rewrite 토글, mode-aware validation, 문서 키 안정화 | 운영 디버깅 단순화, 실행 모드 제약 완화, 동일 파일명 충돌 방지 |

## 🌟 핵심 기능

### 🛡️ Guardrails + Evaluation (v4.2 신규)

검색이 된다고 바로 답변하지 않고, 검색 품질과 답변 안전성을 점검한 뒤 응답하는 운영형 RAG 파이프라인입니다.

- **Retrieval Quality Gate**: 검색 점수와 문서 수가 기준 미달이면 답변 생성을 차단하거나 soft-fail 처리
- **Question Classification**: extraction / regulatory / explanatory 유형을 구분해 처리 경로 최적화
- **Evidence Extraction + Exact Citation**: 규정형 질문에 대해 근거 문장을 먼저 추출하고, 최종 답변에 출처를 복원
- **Numeric Verification**: 답변의 숫자, 기간, 횟수가 실제 검색 문맥에 존재하는지 검증
- **PII Masking**: 이메일, 휴대전화, 주민등록번호 등 민감정보 자동 마스킹
- **Prompt Injection Detection**: 명시적 패턴과 LLM 판정으로 위험 질의 차단
- **Faithfulness / Hallucination Check**: 생성 답변의 충실도와 비근거 주장 여부 후검증
- **Batch Quality Evaluation**: JSON/TSV 데이터셋을 사용해 평균 점수와 질문별 평가 사유를 저장

### 🧠 Contextual Retrieval (v4.1 신규 - Anthropic 방식)

[Anthropic Contextual Retrieval](https://docs.anthropic.com/en/docs/build-with-claude/retrieval) 방식을 참조하여 구현한 **맥락 추가 + 하이브리드 검색**:

- **Contextual Embeddings**: 각 청크에 LLM으로 문서 맥락을 생성하여 임베딩 시 맥락 포함 → 벡터 검색 정확도 35% 향상
- **Contextual BM25**: 맥락이 포함된 텍스트로 BM25 키워드 검색 → 정확한 용어/엔티티 매칭 향상
- **Hybrid Search (BM25 + Vector + Semantic)**: 3단계 결합으로 검색 실패율 49% 감소
- **원본/맥락 분리 저장**: 검색은 맥락 포함 텍스트, 답변 생성은 원본 텍스트 사용
- **전체 전략 적용**: Legal, Hierarchical, Tabular, Fallback 모든 전략에 맥락 추가

```
┌─────────────────────────────────────────────────────────────────┐
│                  Contextual Retrieval Flow                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [인덱싱 시]                                                      │
│  문서 → 청크 분할 → LLM 맥락 생성 → [맥락: ...] + 원본              │
│                          │                                       │
│                    ┌─────┴─────┐                                │
│                    │  chunk    │──→ BM25 Index (키워드 검색)     │
│                    │ (맥락+원본)│──→ Embedding  (벡터 검색)     │
│                    └───────────┘                                │
│                                                                 │
│  [검색 시]                                                      │
│  질문 → Query Rewrite → ┬─ BM25 키워드 검색 (Contextual BM25) │
│                        ├─ Vector 유사성 검색 (Contextual Embed) │
│                        └─ RRF 결합 → Semantic Ranker → 답변     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 📊 Graph RAG (v4.0 - LightRAG 기반)

[LightRAG](https://github.com/HKUDS/LightRAG)의 핵심 아키텍처를 참조하여 구현한 경량 Knowledge Graph 기반 RAG:

- **엔티티/관계 자동 추출**: GPT-5.4를 활용하여 문서 청크에서 엔티티와 관계를 자동 추출
- **Dual-Level Retrieval**: Local (엔티티 중심) + Global (관계/주제 중심) 검색
- **NetworkX 인메모리 그래프**: 빠른 탐색과 서브그래프 추출
- **하이브리드 검색**: Azure AI Search 벡터 검색 + Knowledge Graph 컨텍스트 결합
- **14개 한국어 엔티티 타입**: 인물, 조직, 장소, 날짜, 법률, 정책, 기관, 사건, 기술, 제품, 금액, 지표, 문서, 개념

```
┌─────────────────────────────────────────────────────────────────┐
│                     Graph-Enhanced RAG Flow                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  질문 → [Query Rewrite] → ┬── Azure AI Search (벡터+키워드)     │
│                           ├── Knowledge Graph (Dual-Level)       │
│                           └── ⇒ 결합 컨텍스트 → GPT-5.4 답변   │
│                                                                  │
│  Knowledge Graph:                                                │
│    엔티티 ──[관계]──▶ 엔티티                                     │
│    (인물)  (소속)      (조직)                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 📋 구조화 엔티티 추출 (v4.0 신규 - LangExtract 기반)

[LangExtract](https://github.com/google/langextract)의 핵심 개념을 참조하여 구현한 한국어 문서 구조화 추출:

- **Few-Shot 기반 추출**: 사용자 정의 예시로 LLM 추출 패턴 가이드
- **Multi-Pass Extraction**: 다중 패스로 Recall 향상 (놓친 엔티티 재추출)
- **Source Grounding**: 추출된 엔티티의 원문 위치 추적 (char_interval)
- **한국어 Unicode 토크나이저**: CJK/한글 문자 정확한 위치 매핑
- **병렬 처리**: ThreadPoolExecutor 기반 고속 추출

### 📄 Context-Rich Rolling Window 청킹

단순한 텍스트 분할이 아닌, 문서의 **구조(Hierarchy)**와 **문맥(Context)**을 보존하는 전략:

- **구조적 하이브리드 파싱**: Azure Document Intelligence Layout 모델(`prebuilt-layout`) v4.0 GA
- **계층적 문맥 주입**: 모든 청크에 Breadcrumb(상위 목차 경로) 자동 주입
- **표 독립화 (Table Isolation)**: 표 데이터를 별도 청크로 격리하여 Markdown 형식 유지
- **한국어 특화 청킹**: `kss`를 활용한 한국어 문장 분리 및 의미 단위 분할
- **Visual RAG**: 문서 내 이미지를 **GPT-5.4 Vision**으로 분석하여 텍스트로 통합
- **엔티티 인식 메타데이터** (v4.0): 한글 비율, Graph RAG 적격 여부 태깅

### 🖼️ 향상된 이미지 분석

- **이미지 유형별 분석 가이드**: UI 스크린샷, 표 이미지, 다이어그램/순서도, 차트/그래프
- **문맥 기반 이미지 설명**: 이미지 주변 텍스트를 분석하여 문서 맥락에 맞는 설명 생성
- **한국어 최적화**: 모든 이미지 설명을 자연스러운 한국어로 생성

## ⚡ 주요 특징

- **Graph-Enhanced RAG** (v4.0): 벡터 검색 + Knowledge Graph 결합으로 정확도 향상
- **Contextual Retrieval** (v4.1): Anthropic 방식 맥락 추가로 검색 실패율 49% 감소
- **Hybrid Search** (v4.1): BM25 키워드 + Vector 유사성 + Semantic Ranking 3단계 결합
- **Guardrails Orchestration** (v4.2): 검색 게이트, 숫자 검증, PII 마스킹, injection/faithfulness/hallucination 검사
- **Evidence-Based Answers** (v4.2): 규정형 질문에서 근거 우선 추출 후 답변 + 출처 복원
- **Batch Evaluation** (v4.2): `run_quality_evaluation.py`로 질문별 점수와 평균 점수 산출
- **Search Field Mapping** (v4.3): 기존 Azure AI Search 인덱스의 `id`, `content`, `title`, `parent_id`, `content_vector` 구조 재사용 가능
- **고성능 병렬 처리**: 다중 코어를 활용하여 대량의 문서를 빠르게 인덱싱 (`--workers` 옵션)
- **변경 감지 인덱싱**: 파일 해시 비교를 통한 선택적 처리
- **라이브 운영 검증 완료**: `idx-hr-handson` 인덱스에 실제 재색인 후 질의응답 검증
- **유형별 청크 통계**: 처리 완료 시 텍스트/표/이미지 청크 수 분류 표시

## 🏗️ 아키텍처 (v4.2+)

```
┌─────────────────────────────────────────────────────────────────┐
│                        doc_chunk_main.py                        │
│   (CLI 통합 실행 + Contextual Retrieval + Graph RAG + Guardrails) │
└─────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼              ▼
┌───────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
│  Document Process │ │  Graph RAG        │ │  Entity Extract       │
├───────────────────┤ ├──────────────────┤ ├──────────────────────┤
│ parser.py         │ │ graph_rag.py     │ │ entity_extractor.py  │
│ (Azure DI +       │ │ (LightRAG 기반   │ │ (LangExtract 기반    │
│  GPT-5.4 Vision)  │ │  Knowledge Graph)│ │  구조화 추출)         │
│                   │ │                  │ │                      │
│ chunker.py        │ │ - Entity 추출    │ │ - Few-Shot 추출      │
│ (Context-Rich     │ │ - Relationship   │ │ - Multi-Pass         │
│  Rolling Window   │ │ - Dual-Level 검색│ │ - Source Grounding   │
│  + Contextual     │ └──────────────────┘ └──────────────────────┘
│  Retrieval) [v4.1]│
└───────────────────┘
                    │             │              │
                    ▼             ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│       Q&A / Hybrid Search + Graph RAG + Guardrails              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌───────────────────────────┐            │
│  │    agent.py     │    │ Azure AI Search           │            │
│  │  (Query Rewrite │◀───│ BM25 + Vector + Semantic  │            │
│  │  + Graph RAG    │    │ (Contextual Retrieval)    │            │
│  │  + Hybrid Search│    │ + KG Context              │            │
│  │  + Guardrails)  │    │ + Existing Index Mapping  │            │
│  └─────────────────┘    └───────────────────────────┘            │
│             │                                                    │
│             └── retrieval_gate / evidence / verification         │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Azure AI Foundry / Azure OpenAI               │
├─────────────────────────────────────────────────────────────────┤
│  gpt-5.4 배포명 또는 model-router + text-embedding-3-small      │
└─────────────────────────────────────────────────────────────────┘
```

### 🔄 Graph RAG + Contextual Retrieval 데이터 흐름 (v4.1)

```
문서 파싱 → 청킹 → Contextual Retrieval (맥락 추가)
                   │
                   ├── Azure AI Search 업로드
                   │   ├── SEARCH_CONTENT_FIELD: 맥락 포함 텍스트 → BM25 Index + Embedding
                   │   ├── SEARCH_ORIGINAL_CONTENT_FIELD: 원본 텍스트 → 답변 생성용
                   │   └── SEARCH_VECTOR_FIELD: Contextual Embeddings
                   │
                   └── Knowledge Graph 구축 (선택적)
                        ├── 엔티티 추출 (GPT-5.4 + JSON)
                        ├── 관계 추출 (GPT-5.4 + JSON)
                        └── NetworkX 그래프 저장

질문 검색 → Query Rewrite → Hybrid Search
            ├── BM25 키워드 검색 (Contextual BM25)
            ├── Vector 유사성 검색 (Contextual Embeddings)
            ├── RRF 결합 → Semantic Ranker 재순위
            ├── Knowledge Graph 검색 (선택적)
            └── SEARCH_ORIGINAL_CONTENT_FIELD로 답변 생성 (깨끗한 컨텍스트)
```

## 🚀 시작하기

### 설치

```bash
pip install -r requirements.txt
```

### 환경 변수 설정 (.env)

프로젝트 루트에 `.env` 파일을 생성하고 아래 내용을 복사하여 값을 채워주세요:

> **참고**: `.env` 파일은 `.gitignore`에 포함되어 있어 Git에 커밋되지 않습니다.

```env
# ============================================================
# .env.example — azure_korean_doc_framework v4.1
# 이 내용을 복사하여 .env 파일로 저장하고 값을 채워주세요.
# ============================================================

# -----------------------------------------------------------
# Azure OpenAI (필수)
# -----------------------------------------------------------
AZURE_OPENAI_API_KEY=your_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# 고성능 모델 전용 엔드포인트 (기본값과 동일한 key/endpoint 사용 가능)
OPEN_AI_KEY_5=your_openai_api_key
OPEN_AI_ENDPOINT_5=https://your-resource.openai.azure.com/

# API 버전 (아래 3개 이름 모두 지원, 기본값: 2024-12-01-preview)
AZURE_OPENAI_API_VERSION=2024-12-01-preview
# AZURE_OPENAI_API_VER=2024-12-01-preview
# OPENAI_API_VER=2024-12-01-preview

# -----------------------------------------------------------
# 모델 배포명 (선택 — 미설정 시 기본값 사용)
# -----------------------------------------------------------
# GPT-5.x 는 model-router 를 통해 자동 라우팅됩니다.
AZURE_OPENAI_DEPLOYMENT_NAME=model-router
MODEL_DEPLOYMENT_GPT5_4=model-router
MODEL_DEPLOYMENT_GPT5_2=model-router
MODEL_DEPLOYMENT_GPT5_1=model-router
MODEL_DEPLOYMENT_GPT5=model-router
MODEL_DEPLOYMENT_GPT5_MINI=model-router
MODEL_DEPLOYMENT_GPT4_1=gpt-4.1
MODEL_DEPLOYMENT_O3=model-router
MODEL_DEPLOYMENT_O4_MINI=model-router
MODEL_DEPLOYMENT_CLAUDE_OPUS=model-router
MODEL_DEPLOYMENT_CLAUDE_SONNET=model-router

# 기본 모델 설정
DEFAULT_MODEL=gpt-5.4
VISION_MODEL=gpt-5.4
PARSING_MODEL=gpt-5.4

# -----------------------------------------------------------
# 임베딩 설정 (필수)
# -----------------------------------------------------------
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

# -----------------------------------------------------------
# Azure Document Intelligence (필수 — 문서 파싱용)
# -----------------------------------------------------------
AZURE_DI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_DI_KEY=your_doc_intel_key

# -----------------------------------------------------------
# Azure AI Search (필수 — 벡터/하이브리드 검색용)
# -----------------------------------------------------------
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_INDEX_NAME=your-index-name
AZURE_SEARCH_KEY=your_search_api_key
AZURE_SEARCH_ID_FIELD=chunk_id
AZURE_SEARCH_CONTENT_FIELD=chunk
AZURE_SEARCH_ORIGINAL_CONTENT_FIELD=original_chunk
AZURE_SEARCH_VECTOR_FIELD=text_vector
AZURE_SEARCH_TITLE_FIELD=title
AZURE_SEARCH_PARENT_FIELD=parent_id
AZURE_SEARCH_SOURCE_FIELD=parent_id
AZURE_SEARCH_SEMANTIC_CONFIG=my-semantic-config

# 권장: 사용자에게 보여줄 출처가 파일명/문서 제목이라면 SOURCE_FIELD=title 사용

# 호환 변수명도 지원합니다. 이미 다른 스크립트에서 아래 이름을 쓰는 경우
# config.py가 자동으로 fallback 합니다.
# AZURE_SEARCH_SERVICE_ENDPOINT=https://your-search.search.windows.net
# AZURE_SEARCH_API_KEY=your_search_api_key

# 현재 로컬 검증 환경 예시
# AZURE_SEARCH_ENDPOINT=https://ys-search.search.windows.net
# AZURE_SEARCH_INDEX_NAME=idx-hr-handson
# AZURE_SEARCH_ID_FIELD=id
# AZURE_SEARCH_CONTENT_FIELD=content
# AZURE_SEARCH_ORIGINAL_CONTENT_FIELD=original_chunk
# AZURE_SEARCH_VECTOR_FIELD=content_vector
# AZURE_SEARCH_TITLE_FIELD=title
# AZURE_SEARCH_PARENT_FIELD=parent_id
# AZURE_SEARCH_SOURCE_FIELD=title
# AZURE_SEARCH_SEMANTIC_CONFIG=rag-1766039944266-semantic-configuration1

# 주의: config.py는 load_dotenv(override=True)를 사용하므로,
# 워크스페이스 루트 .env 값이 프로젝트 하위 .env 값을 덮어쓸 수 있습니다.
# 현재 config.py는 AZURE_OPENAI_* 와 OPEN_AI_* 별칭을 함께 읽습니다.

# -----------------------------------------------------------
# [v4.0] Graph RAG 설정 (선택 — 기본값 제공)
# -----------------------------------------------------------
GRAPH_RAG_ENABLED=true
GRAPH_STORAGE_PATH=output/knowledge_graph.json
GRAPH_ENTITY_BATCH_SIZE=5
GRAPH_QUERY_MODE=hybrid
GRAPH_TOP_K=10

# -----------------------------------------------------------
# [v4.1] Contextual Retrieval 설정 (선택 — 기본값 제공)
# -----------------------------------------------------------
CONTEXTUAL_RETRIEVAL_ENABLED=true
CONTEXTUAL_RETRIEVAL_MODEL=gpt-5.4
CONTEXTUAL_RETRIEVAL_MAX_TOKENS=150
CONTEXTUAL_RETRIEVAL_BATCH_SIZE=5
QUERY_REWRITE_ENABLED=true
ANSWER_DIAGNOSTICS_ENABLED=true

# -----------------------------------------------------------
# [v4.2] Retrieval / Guardrails / Evaluation 설정
# -----------------------------------------------------------
RETRIEVAL_GATE_ENABLED=true
RETRIEVAL_GATE_MIN_TOP_SCORE=0.15
RETRIEVAL_GATE_MIN_DOC_COUNT=1
RETRIEVAL_GATE_MIN_DOC_SCORE=0.05
RETRIEVAL_GATE_SOFT_MODE=true
RETRIEVAL_GATE_NOT_FOUND_MESSAGE=관련 문서를 충분히 찾지 못했습니다. 다른 키워드로 다시 질문해 주세요.

EXACT_CITATION_ENABLED=true
NUMERIC_VERIFICATION_ENABLED=true
PII_DETECTION_ENABLED=true
INJECTION_DETECTION_ENABLED=true
FAITHFULNESS_ENABLED=true
FAITHFULNESS_THRESHOLD=0.85
HALLUCINATION_DETECTION_ENABLED=true
HALLUCINATION_THRESHOLD=0.8
EVALUATION_JUDGE_MODEL=gpt-5.4

# -----------------------------------------------------------
# [v4.0] 구조화 엔티티 추출 설정 (선택 — 기본값 제공)
# -----------------------------------------------------------
EXTRACTION_PASSES=1
EXTRACTION_MAX_CHUNK_CHARS=3000
EXTRACTION_MAX_WORKERS=4
```

## 🛠 사용 방법

### 1. 문서 인덱싱 (Ingestion)

```bash
# 디렉토리 내 모든 PDF 처리 (기본 3개 병렬 작업)
python doc_chunk_main.py --path "RAG_TEST_DATA"

# 특정 파일만 처리 및 병렬 작업 수 지정
python doc_chunk_main.py --path "data/sample.pdf" --workers 5

# Q&A 없이 인덱싱만 수행
python doc_chunk_main.py --path "data/sample.pdf" --skip-qa
```

동일한 파일명이 여러 폴더에 있어도 이제는 워크스페이스 기준 상대 경로를 문서 키로 사용하므로 인덱스 충돌 없이 함께 적재됩니다.

### 2. Graph RAG 활용 (v4.0 신규)

```bash
# 문서 인덱싱 + Knowledge Graph 구축 + Graph-Enhanced Q&A
python doc_chunk_main.py --path "RAG_TEST_DATA" --graph-rag

# Graph 검색 모드 지정
python doc_chunk_main.py --path "RAG_TEST_DATA" --graph-rag --graph-mode local
python doc_chunk_main.py --path "RAG_TEST_DATA" --graph-rag --graph-mode global
python doc_chunk_main.py --path "RAG_TEST_DATA" --graph-rag --graph-mode hybrid

# Knowledge Graph만 구축 (Q&A 생략)
python doc_chunk_main.py --path "RAG_TEST_DATA" --graph-rag --skip-qa

# 저장된 Graph로 Q&A만 수행
python doc_chunk_main.py --skip-ingest --graph-rag --question "질문"
```

### 3. 구조화 엔티티 추출 (v4.0 신규)

```bash
# 문서 인덱싱 + 엔티티 추출
python doc_chunk_main.py --path "RAG_TEST_DATA" --extract-entities

# Graph RAG + 엔티티 추출 동시 사용
python doc_chunk_main.py --path "RAG_TEST_DATA" --graph-rag --extract-entities
```

### 4. Q&A 테스트 및 모델 비교

```bash
# 인덱싱 후 바로 테스트 (기본 질문 사용)
python doc_chunk_main.py --path "RAG_TEST_DATA"

# 인덱싱 없이 특정 질문으로 테스트
python doc_chunk_main.py --skip-ingest --question "올해의 경제 전망은?"

# 특정 모델로 테스트
python doc_chunk_main.py --skip-ingest --question "늘봄학교란?" --model "gpt-5.4"
```

### 4-1. 운영 진단 정보 확인 (v4.4)

```python
from azure_korean_doc_framework.core.agent import KoreanDocAgent

agent = KoreanDocAgent()
artifacts = agent.answer_question(
  "늘봄학교란?",
  return_artifacts=True,
)

print(artifacts.answer)
print(artifacts.diagnostics)
```

`artifacts.diagnostics`에는 query variant 수, 상위 검색 점수, 상위 출처, graph context 사용 여부 등이 포함됩니다.

주의: `return_artifacts=True` 와 `return_context=True` 는 동시에 사용할 수 없습니다. 두 옵션을 함께 넘기면 `ValueError`가 발생합니다.

### 5. 출력 예시

```
✅ 처리 완료
   └── 총 세그먼트: 66개
   └── 생성된 청크: 30개
       ├── 텍스트: 23개
       ├── 표: 5개
       └── 이미지: 2개

📊 Knowledge Graph 구축 완료: 노드 45개, 엣지 67개
📋 엔티티 추출 완료: 38개 엔티티 (인물 12, 조직 8, 정책 6, ...)
```

## 📂 프로젝트 구조 (v4.2+)

```
azure_korean_doc_framework1/
├── azure_korean_doc_framework/
│   ├── generation/
│   │   └── evidence_extractor.py # [v4.2] 근거 추출 + 정확 인용 답변
│   ├── guardrails/
│   │   ├── retrieval_gate.py    # [v4.2] 검색 품질 게이트
│   │   ├── numeric_verifier.py  # [v4.2] 숫자/기간/횟수 검증
│   │   ├── pii.py               # [v4.2] 한국어 PII 탐지 및 마스킹
│   │   ├── injection.py         # [v4.2] 프롬프트 인젝션 탐지
│   │   ├── faithfulness.py      # [v4.2] 충실도 검증
│   │   ├── hallucination.py     # [v4.2] 할루시네이션 검증
│   │   └── question_classifier.py # [v4.2] 질문 유형 분류
│   ├── parsing/
│   │   ├── parser.py            # Azure DI + GPT-5.4 Vision 하이브리드 파싱
│   │   ├── chunker.py           # Context-Rich 청킹 + Contextual Retrieval (v4.1)
│   │   └── entity_extractor.py  # [v4.0] LangExtract 기반 구조화 추출
│   ├── core/
│   │   ├── vector_store.py      # Azure AI Search 인덱싱/증분 업데이트/필드 매핑
│   │   ├── agent.py             # RAG 에이전트 (v4.1: Hybrid Search + Contextual)
│   │   ├── graph_rag.py         # [v4.0] LightRAG 기반 Knowledge Graph
│   │   ├── multi_model_manager.py # 멀티 모델 관리
│   │   └── schema.py            # 문서 스키마
│   ├── utils/
│   │   ├── logger.py            # 청크 로그 저장
│   │   └── azure_clients.py     # Azure 클라이언트 초기화
│   └── config.py                # 환경 변수 검증 및 설정 (v4.1: Contextual Retrieval)
├── doc_chunk_main.py            # 통합 CLI 실행 (v4.1: Contextual Retrieval + Graph RAG)
├── run_quality_evaluation.py    # [v4.2] 배치형 품질 평가 스크립트
├── run_guardrail_scenarios.py   # [v4.2] 오프라인 가드레일 시나리오 데모
├── test_framework_v4.py         # 종합 테스트 스위트 (현재 193개 항목 검증)
├── requirements.txt             # Python 의존성 패키지
├── .gitignore                   # Git 무시 파일 설정
├── output/                      # 청크 로그 + Knowledge Graph 출력
│   ├── *_chunks.json            # 청크 로그
│   ├── *_entities.json          # [v4.0] 추출된 엔티티
│   └── knowledge_graph.json     # [v4.0] Knowledge Graph
├── .env                         # 환경 변수 설정
└── README.md
```

## 🔧 핵심 모듈 설명

### graph_rag.py (v4.0 신규 - LightRAG 기반)

| 클래스/함수 | 설명 |
|------------|------|
| `KnowledgeGraphManager` | Knowledge Graph 구축/검색 관리자 |
| `extract_from_chunks()` | 문서 청크에서 엔티티/관계 추출 (GPT-5.4) |
| `query()` | Dual-Level 검색 (Local/Global/Hybrid) |
| `_local_search()` | 엔티티 중심 검색 (Low-Level Keywords) |
| `_global_search()` | 관계/주제 중심 검색 (High-Level Keywords) |
| `get_subgraph()` | 특정 엔티티 중심 서브그래프 추출 |
| `save_graph() / load_graph()` | Knowledge Graph JSON 저장/로드 |

### entity_extractor.py (v4.0 신규 - LangExtract 기반)

| 클래스/함수 | 설명 |
|------------|------|
| `StructuredEntityExtractor` | Few-Shot 기반 구조화 추출기 |
| `extract()` | 텍스트에서 구조화된 엔티티 추출 |
| `extract_from_document_chunks()` | 청크 리스트에서 추출 |
| `KoreanUnicodeTokenizer` | 한국어/CJK Unicode 토크나이저 |
| `ExampleData` / `Extraction` | Few-Shot 예시 및 추출 결과 데이터 모델 |

### agent.py (RAG 에이전트) - v4.2 업데이트

| 함수/상수 | 설명 |
|----------|------|
| `_RAG_SYSTEM_PROMPT` | 모듈 상수 — 기본 RAG 시스템 프롬프트 (DRY 원칙) |
| `_GRAPH_RAG_SYSTEM_PROMPT` | 모듈 상수 — Graph RAG용 확장 시스템 프롬프트 |
| `_vector_search()` | **[v4.1]** Hybrid Search: BM25 + Vector + Semantic Ranking (Contextual Retrieval) |
| `answer_question()` | 기본 RAG 질의응답. `return_artifacts=True` 시 diagnostics 포함 산출물 반환 |
| `graph_enhanced_answer()` | **[v4.0]** Graph-Enhanced RAG (벡터 + KG 결합). `return_artifacts=True` 지원 |
| `_rewrite_query()` | Query Rewrite - 의미적 쿼리 확장 |

### v4.2 운영 안전장치

- **Retrieval Quality Gate**: 검색 점수가 임계값 미달이면 답변 생성을 차단하거나 soft-fail 처리
- **Question Classification**: extraction / regulatory / explanatory 분기
- **Exact Citation / Evidence Extraction**: 규정형 질문에 대해 근거 문장을 먼저 추출하고, 최종 답변에 출처를 복원
- **Numeric Verification**: 답변의 숫자/기간/횟수가 검색 컨텍스트에 실제 존재하는지 검증
- **PII Masking**: 이메일, 휴대전화, 주민등록번호 등 민감정보 자동 마스킹
- **Prompt Injection Detection**: 명시적 패턴 + LLM 판정 기반 질의 차단
- **Faithfulness / Hallucination Check**: 생성 답변의 충실도와 비근거 주장 비율 후검증

### 품질 평가 실행 (v4.2 신규)

```bash
# JSON dataset 예시: [{"question": "...", "ground_truth": "..."}]
python run_quality_evaluation.py --dataset "test_dataset.json"

# TSV dataset 예시: 질문<TAB>정답
python run_quality_evaluation.py --dataset "test_dataset.tsv" --model "gpt-5.4"
```

결과는 `output/evaluation_results.json`에 저장되며 평균 점수와 질문별 평가 사유를 포함합니다.

### 가드레일 시나리오 데모 (v4.2 신규)

```bash
python run_guardrail_scenarios.py
```

이 스크립트는 Azure 연결 없이 오프라인으로 다음 시나리오를 재현합니다.

- Retrieval Gate hard block
- 규정형 질문의 근거 추출
- 숫자 검증
- PII 마스킹
- Prompt Injection 차단

### parser.py (문서 분석)

| 함수 | 설명 |
|------|------|
| `parse()` | PDF를 Azure DI v4.0으로 분석하여 구조화된 세그먼트 추출 |
| `_describe_image()` | **GPT-5.4 Vision** (model-router)으로 이미지 분석 |
| `_enhance_numbered_content()` | 번호 목록 형식 정규화 |

### chunker.py (청킹) - v4.1 업데이트

| 함수 | 설명 |
|------|------|
| `chunk()` | 문서 세그먼트를 적응적 전략으로 청킹 + Contextual Retrieval 적용 |
| `_apply_contextual_retrieval()` | **[v4.1]** 모든 청크에 LLM 맥락 자동 추가 (Anthropic 방식) |
| `_generate_context()` | **[v4.1]** 전체 문서 참조 청크별 맥락 생성 |
| `_enrich_metadata()` | **[v4.0]** 한글 비율, Graph RAG 적격 여부 태깅 |
| `_split_korean_sentences()` | kss + 정규식 한국어 문장 분리 |
| `_merge_sentences_to_chunks()` | 오버랩 적용 청크 병합 |

### config.py (설정) - v4.2 업데이트

| 설정 | 설명 |
|------|------|
| `CONTEXTUAL_RETRIEVAL_ENABLED` | **[v4.1]** Contextual Retrieval 활성화 여부 (기본: true) |
| `CONTEXTUAL_RETRIEVAL_MODEL` | **[v4.1]** 맥락 생성 모델 (기본: gpt-5.4) |
| `CONTEXTUAL_RETRIEVAL_MAX_TOKENS` | **[v4.1]** 맥락 텍스트 최대 토큰 수 (기본: 150) |
| `CONTEXTUAL_RETRIEVAL_BATCH_SIZE` | **[v4.1]** 맥락 생성 배치 크기 (기본: 5) |
| `GRAPH_RAG_ENABLED` | **[v4.0]** Graph RAG 활성화 여부 |
| `GRAPH_STORAGE_PATH` | **[v4.0]** Knowledge Graph 저장 경로 |
| `GRAPH_QUERY_MODE` | **[v4.0]** 그래프 검색 모드 (local/global/hybrid) |
| `EXTRACTION_PASSES` | **[v4.0]** Multi-Pass 추출 횟수 |
| `SEARCH_*_FIELD` | 기존 Azure AI Search 인덱스 재사용을 위한 필드 매핑 |
| `SEARCH_SEMANTIC_CONFIG` | 사용할 Azure AI Search semantic configuration 이름 |
| `MODELS` | 모델명 → 배포명 매핑 (`gpt-5.4` → 환경 변수에 지정한 배포명) |

### vector_store.py (벡터 저장소)

| 함수 | 설명 |
|------|------|
| `upload_documents()` | 청크를 Azure AI Search에 업로드 (맥락 포함 텍스트 + 원본 분리 저장) |
| `create_index_if_not_exists()` | 기본 인덱스 스키마 자동 생성 |
| `_ensure_incremental_fields()` | 기존 인덱스에 parent/original/semantic 설정 보정 |
| `delete_documents_by_parent_id()` | 부모 문서 기준 증분 삭제 |

## 📈 청크 메타데이터 (v4.1+)

각 청크에는 다음 메타데이터가 포함됩니다:

| 필드 | 설명 |
|------|------|
| `chunk_type` | 청크 유형 (text, table, image) |
| `breadcrumb` | 상위 섹션 경로 (예: "1장 > 1.1절") |
| `source_file` | 원본 파일명 |
| `page_number` | 페이지 번호 |
| `is_table_data` | 표 데이터 여부 |
| `is_image_data` | 이미지 설명 여부 |
| `contains_numbered_section` | 번호 섹션 포함 여부 |
| `hangul_ratio` | **[v4.0]** 한글 문자 비율 (0.0~1.0) |
| `graph_rag_eligible` | **[v4.0]** Graph RAG 대상 여부 |
| `has_contextual_retrieval` | **[v4.1]** Contextual Retrieval 적용 여부 |
| `context` | **[v4.1]** LLM이 생성한 맥락 설명 텍스트 |
| `original_chunk` | **[v4.1]** 맥락 추가 전 원본 청크 텍스트 |

## 📋 요구사항

- Python 3.9+
- Azure 구독 및 다음 서비스:
  - **Azure AI Foundry / Azure OpenAI** (GPT-5.4 배포명 또는 model-router, text-embedding-3-small)
  - **Azure Document Intelligence** v4.0 GA (2024-11-30)
  - **Azure AI Search** (Semantic Ranking 활성화 권장)

### Python 패키지

```bash
pip install -r requirements.txt
```

`requirements.txt`에 모든 의존성이 정의되어 있습니다.

### 🧪 테스트 실행

```bash
# 전체 모듈 테스트 (12개 모듈)
python test_framework_v4.py

# Q&A 전용 테스트 (Ingest 없이)
python doc_chunk_main.py --skip-ingest -q "테스트 질문" --model gpt-5.4

# 현재 Search 인덱스에 대한 라이브 점검 예시
python doc_chunk_main.py --skip-ingest -q "근속기간 3년이면 연차휴가 일수가 어떻게 되지?" --model gpt-5.4

# 외부 인덱스 매핑 라이브 점검 예시
python doc_chunk_main.py --skip-ingest -q "성균관대 PDF에서 글로벌경영학 계열 입학정원은 몇 명인가요?" --model gpt-5.4

# Graph RAG 테스트
python doc_chunk_main.py --path "RAG_TEST_DATA" --graph-rag --graph-mode hybrid
```

## 🔄 변경 이력

### v4.1 (2026-02)
- ✅ **Contextual Retrieval** (Anthropic 방식)
  - 모든 청크에 LLM 기반 문서 맥락 자동 추가 (`_apply_contextual_retrieval()`)
  - 전체 문서 참조 Anthropic 스타일 프롬프트 (`<document>`, `<chunk>` 태그)
  - Contextual BM25: 맥락 포함 텍스트로 BM25 키워드 검색 정확도 향상
  - Contextual Embeddings: 맥락 포함 텍스트로 벡터 임베딩 정확도 향상
  - 검색 실패율 49% 감소 (BM25 + Vector + Semantic 결합 시)
- ✅ **Hybrid Search** (BM25 + Vector + Semantic Ranking)
  - Azure AI Search RRF(Reciprocal Rank Fusion) 활용
  - BM25 키워드 + Vector 유사성 + Semantic Ranker 3단계 결합
  - 검색은 chunk (맥락 포함), 답변은 original_chunk (원본) 분리
- ✅ **인덱스 스키마** `original_chunk` 필드 추가 (원본 텍스트 저장)
- ✅ **config.py** Contextual Retrieval 설정 추가 (CONTEXTUAL_RETRIEVAL_*)
- ✅ **chunker.py** 모든 청킹 전략에 맥락 추가 적용 (Legal 전략 단독 맥락 제거, 통합 방식으로 대체)
- ✅ **agent.py** `_vector_search()` 하이브리드 검색 개선 (Contextual BM25 + Embeddings)
- ✅ **doc_chunk_main.py** 모듈 주석 업데이트

### v4.2 (2026-03)
- ✅ **Retrieval Quality Gate** 추가 (`retrieval_gate.py`)
- ✅ **Question Classification** 추가 (`question_classifier.py`)
- ✅ **Exact Citation / Evidence Extraction** 추가 (`evidence_extractor.py`)
- ✅ **Numeric Verification** 추가 (`numeric_verifier.py`)
- ✅ **PII / Injection / Faithfulness / Hallucination** 가드레일 추가
- ✅ **agent.py**를 검색-검증-생성 오케스트레이터로 확장
- ✅ **run_quality_evaluation.py** 배치 평가 스크립트 추가
- ✅ **Azure AI Search 필드 매핑** 추가 (`SEARCH_ID_FIELD`, `SEARCH_PARENT_FIELD`, `SEARCH_TITLE_FIELD`, `SEARCH_SEMANTIC_CONFIG`)
- ✅ **출처 복원**: 근거 추출 기반 답변에도 `[출처: ...]` 자동 부착
- ✅ **라이브 검증**: `idx-hr-handson` + `성균관대.pdf` 재색인 및 질의응답 확인

### v4.0 (2026-02)
- ✅ **Graph RAG** (LightRAG 기반 Knowledge Graph)
  - GPT-5.4 기반 한국어 엔티티/관계 자동 추출
  - NetworkX 인메모리 Knowledge Graph 구축
  - Dual-Level Retrieval (Local + Global 검색)
  - Azure AI Search + KG 하이브리드 컨텍스트
  - 14개 한국어 특화 엔티티 타입
  - JSON 기반 그래프 영구 저장/로드
- ✅ **구조화 엔티티 추출** (LangExtract 기반)
  - Few-Shot 기반 LLM 추출 (사용자 정의 예시)
  - Multi-Pass Extraction (다중 패스 Recall 향상)
  - Source Grounding (원문 위치 char_interval 추적)
  - 한국어 Unicode 토크나이저 (CJK/한글 정확한 위치 매핑)
  - ThreadPoolExecutor 기반 병렬 추출
- ✅ **agent.py** Graph-Enhanced RAG 모드 추가
- ✅ **chunker.py** 엔티티 인식 메타데이터 (`hangul_ratio`, `graph_rag_eligible`)
- ✅ **config.py** Graph RAG / 추출 설정 추가
- ✅ **doc_chunk_main.py** CLI 옵션 확장 (`--graph-rag`, `--graph-mode`, `--extract-entities`)

### v4.0.1 (2026-02) - 코드 최적화
- ✅ 시스템 프롬프트 DRY 리팩토링 (`_RAG_SYSTEM_PROMPT` / `_GRAPH_RAG_SYSTEM_PROMPT` 모듈 상수)
- ✅ Dead code 제거 (`GPT_4O_DEPLOYMENT_NAME`, 주석 처리된 import, 빈 pass 블록)
- ✅ `import re` 모듈 최상단 이동 (parser.py)
- ✅ list comprehension 적용 (logger.py)
- ✅ 불필요한 `getattr()` 제거 (doc_chunk_main.py)
- ✅ 중복 스크립트 정리 (`doc_chunk_main_only_q.py`, `verify_index.py` 삭제)
- ✅ `requirements.txt` 추가, `test_framework_v4.py` 통합 테스트 (153 통과 / 9 스킵 / 0 실패)

### v3.0.1 (2026-01)
- ✅ model-router 배포 방식 지원 (Azure AI Foundry)
- ✅ 통합 클라이언트 아키텍처
- ✅ frozenset 최적화 (ADVANCED_MODELS, REASONING_MODELS)
- ✅ test_framework.py 추가

### v3.0 (2026-01)
- ✅ GPT-5.4 기본 모델로 전환
- ✅ max_completion_tokens 파라미터 지원
- ✅ Structured Outputs 지원
- ✅ Query Rewrite 기능 추가
- ✅ reasoning_effort 파라미터 지원

### v2.0
- 이미지 유형별 분석 가이드
- 표 전용 청크 분리
- 번호 목록 처리 개선

## 🔗 참조 프로젝트

- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) - Contextual Retrieval 아키텍처 참조
- [LightRAG](https://github.com/HKUDS/LightRAG) - Graph RAG 아키텍처 참조
- [LangExtract](https://github.com/google/langextract) - 구조화 추출 아키텍처 참조
- [Korean_Doc_Chunking_Azure](../Korean_Doc_Chunking_Azure) - 이전 버전
- [azure_search_foundry_agent.py](../azure_search_foundry_agent.py) - Foundry Agent 통합 버전

## 📜 라이선스

MIT License
