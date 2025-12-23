# 🇰🇷 Azure Korean Document Framework

> 한국어 문서 이해 및 검색을 위한 RAG(Retrieval-Augmented Generation) 프레임워크

## ✨ 한눈에 보기

```
📄 PDF 문서 → 🔍 Azure DI 파싱 → ✂️ 스마트 청킹 → 🗄️ 벡터 검색 → 🤖 AI 답변
```

| 기능 | 설명 |
|------|------|
| **문서 파싱** | Azure Document Intelligence + GPT Vision으로 텍스트, 표, 이미지 추출 |
| **한국어 청킹** | `kss` 기반 문장 분리 + 토큰 기반 오버랩 청킹 |
| **벡터 검색** | Azure AI Search의 하이브리드 검색 (키워드 + 벡터) |
| **멀티 모델** | GPT-4.1, GPT-5.2, Claude 등 다양한 LLM 지원 |

---

## 💎 왜 이 프레임워크인가요? (Advantages)

1.  **한국어 최적화 (Korean-Centric)**: 일반적인 공백 기반 분할이 아닌, `kss` 라이브러리를 통해 한국어 문장 경계를 정확히 인식합니다.
2.  **구조 분석 기반 (Structure-Aware)**: 단순 길이 기반 분할 대신 문서의 제목(`header`), 표(`table`), 이미지(`image`) 구조를 이해하고 문맥을 보존합니다.
3.  **동적 전략 선택 (Adaptive Strategy)**: 문서의 성격(법률, 통계, 일반 보고서 등)을 파악하여 자동으로 최적의 청킹 전략(`LEGAL`, `TABULAR` 등)을 적용합니다.
4.  **문맥 보존 (Context-Rich)**: 상위 제목(Breadcrumb)정보를 하위 청크에 주입하여, 각 청크가 독립적으로도 충분한 의미를 가질 수 있게 합니다.

---

## 🔄 RAG 청킹 프로세스 (Step-by-Step)

이 프레임워크는 단순히 글자를 자르는 것이 아니라, 다음의 정교한 단계를 거칩니다:

1.  **데이터 파싱 (Parsing)**: Azure Document Intelligence를 사용하여 문서의 계층 구조(H1, H2, H3...)와 표, 이미지 설명을 추출합니다.
2.  **의미 단위 세그먼트화 (Segmentation)**: 추출된 데이터를 의미가 연결되는 블록 단위로 그룹화합니다.
3.  **한국어 문장 분리 (Sentence Splitting)**: `kss`를 사용하여 각 블록 내의 한국어 문장을 정확하게 하나하나 분리합니다.
4.  **토큰 정밀 카운팅 (Token Counting)**: `tiktoken`을 사용하여 LLM(GPT-4/5)이 이해하는 실제 토큰 단위로 길이를 계산합니다.
5.  **슬라이딩 윈도우 오버랩 (Overlap Grouping)**: 설정된 토큰 제한(Max Tokens)에 맞춰 문장들을 묶되, 청크 간에 일정한 오버랩(Overlap)을 두어 정보 단절을 방지합니다.
6.  **메타데이터 강화 (Enrichment)**: 각 청크에 파일명, 페이지 번호, 상위 제목 경로, 토큰 수 등의 정보를 주입합니다.
7.  **벡터 인덱싱 (Indexing)**: 최종 가공된 청크를 벡터로 변환하여 Azure AI Search에 안전하게 저장합니다.

---

## 📦 설치

```bash
pip install openai azure-ai-documentintelligence azure-search-documents \
    langchain langchain-openai langchain-experimental \
    pymupdf pillow python-dotenv tiktoken kss
```

---

## ⚡ 빠른 시작

### 1단계: 환경 변수 설정

`.env` 파일을 생성하고 Azure 정보를 입력하세요:

```env
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_API_KEY=your-search-key
MODEL_DEPLOYMENT_GPT4_1=gpt-4.1
```

### 2단계: 문서 처리 실행

```bash
# 단일 파일 처리
python doc_chunk_main.py --path "문서.pdf"

# 디렉토리 전체 처리
python doc_chunk_main.py --path "문서폴더/"

# Q&A 테스트 포함
python doc_chunk_main.py --path "문서.pdf" --question "문서 내용을 요약해줘"
```

---

## 🧩 핵심 컴포넌트

### 📁 디렉토리 구조

```
azure_korean_doc_framework/
├── core/
│   ├── agent.py              # RAG 에이전트 (질의 응답)
│   ├── multi_model_manager.py # 멀티 LLM 관리
│   └── vector_store.py       # Azure AI Search 연동
├── parsing/
│   ├── parser.py             # 문서 파싱 (Azure DI + GPT Vision)
│   └── chunker.py            # 스마트 청킹
├── utils/
│   └── azure_clients.py      # Azure 클라이언트 팩토리
└── config.py                 # 설정 관리
```

---

## ⚙️ 청킹 설정

### 기본 사용법

```python
from azure_korean_doc_framework.parsing.chunker import AdaptiveChunker

chunker = AdaptiveChunker()  # 기본 설정으로 사용
```

### 커스텀 설정

```python
from azure_korean_doc_framework.parsing.chunker import ChunkingConfig, AdaptiveChunker

config = ChunkingConfig(
    min_tokens=100,      # 최소 청크 크기
    max_tokens=500,      # 최대 청크 크기
    overlap_tokens=50,   # 청크 간 오버랩 (문맥 연속성)
)
chunker = AdaptiveChunker(config=config)
```

### 청킹 전략 (자동 선택)

| 전략 | 적용 조건 | 특징 |
|------|----------|------|
| **LEGAL** | 판례, 법률 문서 | 【주문】, 【이유】 등 구조 인식 |
| **TABULAR** | 표 중심 문서 | 표를 자연어 문장으로 변환 |
| **HIERARCHICAL** | 제목 구조 문서 | Breadcrumb 기반 문맥 보존 |
| **FALLBACK** | 기타 문서 | 오버랩 적용 단순 분할 |

---

## 💡 코드 예시

### Q&A 에이전트 사용

```python
from azure_korean_doc_framework.core.agent import KoreanDocAgent

agent = KoreanDocAgent()
answer = agent.answer_question(
    "회사의 복지 정책은?",
    model_key="gpt-5.2"  # 또는 "gpt-4.1", "claude-sonnet-4-5"
)
print(answer)
```

### 문서 파싱만 사용

```python
from azure_korean_doc_framework.parsing.parser import HybridDocumentParser

parser = HybridDocumentParser()
segments = parser.parse("문서.pdf")
# segments = [{"type": "text", "content": "..."}, {"type": "table", "content": "..."}]
```

### 전체 파이프라인

```python
from azure_korean_doc_framework.parsing.parser import HybridDocumentParser
from azure_korean_doc_framework.parsing.chunker import AdaptiveChunker
from azure_korean_doc_framework.core.vector_store import VectorStore
from azure_korean_doc_framework.core.agent import KoreanDocAgent

# 1. 컴포넌트 초기화
parser = HybridDocumentParser()
chunker = AdaptiveChunker()
vector_store = VectorStore()

# 2. 문서 처리
segments = parser.parse("문서.pdf")
chunks = chunker.chunk(segments, filename="문서.pdf")
vector_store.upload_documents(chunks)

# 3. 질의 응답
agent = KoreanDocAgent()
answer = agent.answer_question("문서 내용을 요약해줘")
print(answer)
```

---

## 📊 청크 메타데이터

각 청크에는 다음 정보가 포함됩니다:

| 필드 | 설명 | 예시 |
|------|------|------|
| `chunk_index` | 청크 순번 | `0`, `1`, `2` |
| `total_chunks` | 전체 청크 수 | `39` |
| `token_count` | 토큰 수 | `485` |
| `char_count` | 문자 수 | `2235` |
| `breadcrumb` | 섹션 경로 | `"1장 > 개요 > 배경"` |
| `source` | 원본 파일명 | `"문서.pdf"` |

---

## 🔗 참고 자료

- [Azure Document Intelligence](https://learn.microsoft.com/azure/ai-services/document-intelligence/)
- [Azure AI Search](https://learn.microsoft.com/azure/search/)
- [LangChain](https://python.langchain.com/)
