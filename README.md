# Azure Korean Document Understanding & Retrieval Framework

이 프로젝트는 Azure AI Services (Document Intelligence, OpenAI)를 활용하여 한국어 문서를 정밀하게 분석하고, RAG (Retrieval-Augmented Generation) 시스템을 위한 최적의 청킹(Chunking) 데이터를 생성 및 관리하는 프레임워크입니다.

## 🌟 핵심 기능: Context-Rich Rolling Window

단순한 텍스트 분할이 아닌, 문서의 **구조(Hierarchy)**와 **문맥(Context)**을 보존하는 **"Context-Rich Rolling Window"** 전략을 사용합니다.

- **구조적 하이브리드 파싱**: Azure Document Intelligence Layout 모델을 사용하여 헤더, 본문, 표, 이미지를 구분합니다.
- **계층적 문맥 주입**: 모든 청크에 Breadcrumb(상위 목차 경로)를 자동으로 주입하여 검색 품질을 향상시킵니다.
- **표 독립화 (Table Isolation)**: 표 데이터를 별도의 청크로 격리하여 Markdown 형식으로 구조를 유지합니다.
- **한국어 특화 청킹**: `kss`를 활용한 한국어 문장 분리 및 의미 단위 분할을 지원합니다.
- **Visual RAG**: 문서 내 이미지를 GPT-4o Vision으로 분석하여 텍스트로 통합합니다.

## ⚡ 주요 특징

- **고성능 병렬 처리**: 다중 코어를 활용하여 대량의 문서를 빠르게 인덱싱합니다. (`--workers` 옵션)
- **변경 감지 인덱싱**: 파일의 해시(Hash)를 비교하여 변경된 파일만 선택적으로 처리합니다.
- **멀티 모델 평가**: 동일한 질문에 대해 다양한 모델(GPT-4o, o1 등)의 답변을 비교 테스트할 수 있습니다.

## 🚀 시작하기

### 설치
```bash
pip install openai azure-ai-documentintelligence azure-search-documents \
    pymupdf pillow python-dotenv tiktoken kss
```

### 환경 변수 설정 (.env)
```env
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_VERSION=...
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=...
AZURE_DOCUMENT_INTELLIGENCE_KEY=...
AZURE_SEARCH_SERVICE_ENDPOINT=...
AZURE_SEARCH_INDEX_NAME=...
AZURE_SEARCH_API_KEY=...
```

## 🛠 사용 방법

### 1. 문서 인덱싱 (Ingestion)
```bash
# 디렉토리 내 모든 PDF 처리 (기본 3개 병렬 작업)
python doc_chunk_main.py --path "RAG_TEST_DATA"

# 특정 파일만 처리 및 병렬 작업 수 지정
python doc_chunk_main.py --path "data/sample.pdf" --workers 5
```

### 2. Q&A 테스트 및 모델 비교
```bash
# 인덱싱 후 바로 테스트 (기본 질문 사용)
python doc_chunk_main.py --path "RAG_TEST_DATA"

# 인덱싱 없이 특정 질문으로 테스트
python doc_chunk_main.py --skip-ingest --question "올해의 경제 전망은?" --model "gpt-5.2"
```

## 📂 프로젝트 구조
- `azure_korean_doc_framework/`
  - `parsing/`: `parser.py`(문서 분석), `chunker.py`(한국어 의미 단위 분할)
  - `core/`: `vector_store.py`(Azure AI Search), `agent.py`(RAG 로직)
  - `utils/`: `logger.py`(청크 로그 저장), `azure_clients.py`
- `doc_chunk_main.py`: 통합 실행 스크립트 (CLI 지원)
- `config.py`: 환경 변수 검증 및 모델 구성 관리
