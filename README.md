# Azure Korean Document Understanding & Retrieval Framework

이 프레임워크는 한국어 문서의 깊은 이해(Deep Document Understanding)와 효율적인 검색(Retrieval)을 위해 설계되었습니다. Tencent의 WeKnora 및 Microsoft Agent Framework의 주요 패턴을 차용하여 Azure AI 기술과 한국어 최적화 로직을 결합했습니다.

## 🚀 주요 특징

- **Multi-Model Support**: GPT-4.1, GPT-5.2, claude-sonnet-4-5, claude-opus-4-5등 다양한 LLM을 동적으로 교체하며 사용 가능합니다.
- **Hybrid Document Parsing**: Azure Document Intelligence와 GPT-4.1을 연동하여 텍스트뿐만 아니라 차트, 표, 이미지의 의미를 텍스트로 추출하여 검색 성능을 극대화합니다.
- **Korean Semantic Chunking**: 단순 길이 기반 분할이 아닌, 마크다운 구조와 문맥의 의미를 파악하는 시맨틱 청킹을 지원합니다.
- **Azure AI Search Integration**: 한국어 최적화 분석기(`ko.microsoft`)와 벡터 검색을 활용한 하이브리드 검색 환경을 제공합니다.

## 📂 디렉토리 구조

```text
azure_korean_doc_framework/
├── core/
│   ├── agent.py               # RAG 오케스트레이션 및 답변 생성 기반
│   ├── multi_model_manager.py # GPT/Claude 멀티 모델 관리 및 호출
│   └── vector_store.py        # Azure AI Search 인덱스 및 업로드 관리
├── parsing/
│   ├── parser.py              # Azure DI + GPT Vision 기반 하이브리드 파서
│   └── chunker.py             # 마크다운 헤더 및 시맨틱 청커
├── utils/
│   └── azure_clients.py       # Azure 서비스 클라이언트 팩토리
├── config.py                  # 환경 변수 및 설정 관리
└── README.md                  # 본 문서
```

## 🛠️ 설치 및 설정

### 1. 필수 라이브러리 설치
```bash
pip install openai azure-ai-documentintelligence azure-search-documents langchain langchain-openai langchain-experimental pdf2image pillow python-dotenv
```
> [!NOTE]
> PDF 시각적 분석 기능을 위해서는 시스템에 `poppler`가 설치되어 있어야 합니다.

### 2. 환경 변수 설정
프로젝트 루트 폴더에 `.env` 파일을 생성하고 `.env.template`의 내용을 복사하여 Azure 정보를 입력하세요.

```env
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
MODEL_DEPLOYMENT_GPT4_1=gpt-4.1
AZURE_SEARCH_ENDPOINT=...
# ... (기타 설정)
```

## 📖 사용 방법

### 기본 실행 (`main.py`)
프레임워크의 전체 흐름(문서 파싱 -> 청킹 -> 검색 인덱싱 -> 멀티 모델 실습)을 한 번에 확인할 수 있습니다.

```bash
python main.py
```

### 코드 예시

#### 1. 인코딩 및 멀티 모델 답변
```python
from azure_korean_doc_framework.core.agent import KoreanDocAgent

agent = KoreanDocAgent()
# GPT-5.1 또는 Claude 등 원하는 모델 명시 가능
answer = agent.answer_question("회사의 복지 정책에 대해 알려줘", model_key="gpt-5.2")
print(answer)
```

#### 2. 하이브리드 문서 파싱
```python
from azure_korean_doc_framework.parsing.parser import HybridDocumentParser

parser = HybridDocumentParser()
markdown_text = parser.parse("document.pdf")
print(markdown_text) # 텍스트 + 표 + 이미지 설명이 포함된 마크다운
```

#### 3. 전체 사용 방법 : 파싱 > 청킹 > 인덱싱 > Q&A
```python
import os
from azure_korean_doc_framework.parsing.parser import HybridDocumentParser
from azure_korean_doc_framework.parsing.chunker import KoreanSemanticChunker
from azure_korean_doc_framework.core.vector_store import VectorStore
from azure_korean_doc_framework.core.agent import KoreanDocAgent

def main():
    print("🌟 Welcome to Azure Korean Document Understanding & Retrieval Framework 🌟")

    # 1. Initialize Components
    parser = HybridDocumentParser()
    chunker = KoreanSemanticChunker()
    vector_store = VectorStore()

    # 2. Document Ingestion (Example)
    # 실제 파일 경로로 수정 필요
    sample_file = "RAG_TEST_DATA/(1) 2024 달라지는 세금제도.pdf"

    if os.path.exists(sample_file):
        print(f"\n--- [Phase 1: Ingestion - {sample_file}] ---")
        # [수정] 파일 수정 시간 확인 및 업데이트 로직 적용
        file_mod_time = os.path.getmtime(sample_file)

        vector_store.create_index_if_not_exists()

        # 최신 상태인지 확인
        if vector_store.is_file_up_to_date(os.path.basename(sample_file), file_mod_time):
             print(f"⏩ File is up-to-date. Skipping parsing/upload.")
        else:
            print(f"🔄 File updated or new. Processing...")
            # 기존 데이터 삭제 (업데이트 시)
            vector_store.delete_documents_by_parent_id(os.path.basename(sample_file))

            # 파싱 및 청킹
            markdown_content = parser.parse(sample_file)

            # 메타데이터에 파일명과 수정 시간 추가
            extra_meta = {
                "source": os.path.basename(sample_file),
                "last_modified": file_mod_time
            }
            chunks = chunker.chunk(markdown_content, extra_metadata=extra_meta)

            vector_store.upload_documents(chunks)
    else:
        print(f"\nℹ️ Skip ingestion: {sample_file} not found. Running Q&A with existing search index.")

    # 3. Multi-Model Q&A Demonstration
    agent = KoreanDocAgent()
    question = "이 문서에서 가장 중요한 핵심 요약 세 가지만 말해줘."

    models_to_test = ["gpt-4.1", "gpt-5.2", "claude-sonnet-4-5"]

    print("\n--- [Phase 2: Multi-Model Q&A] ---")
    print(f"User Question: {question}")

    for model in models_to_test:
        print(f"\n--- Model: {model} ---")
        answer = agent.answer_question(question, model_key=model)
        print(f"Response:\n{answer}")

if __name__ == "__main__":
    main()
```

## 🤝 참조 프로젝트
- [Tencent WeKnora](https://github.com/Tencent/WeKnora)
- [Microsoft Agent Framework Samples](https://github.com/microsoft/agent-framework)

