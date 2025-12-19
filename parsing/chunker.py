from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import AzureOpenAIEmbeddings
from ..config import Config

class KoreanSemanticChunker:
    def __init__(self):
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=Config.EMBEDDING_DEPLOYMENT,
            openai_api_version=Config.OPENAI_API_VERSION,
            azure_endpoint=Config.OPENAI_ENDPOINT,
            api_key=Config.OPENAI_API_KEY
        )

    def chunk(self, markdown_text, extra_metadata=None):
        """
        1단계: Markdown Header 기준으로 의미 단위 분리
        2단계: Azure OpenAI Embedding을 이용한 Semantic Chunking
        """
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        header_splits = markdown_splitter.split_text(markdown_text)

        # Semantic Chunker 초기화
        text_splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile"
        )

        final_chunks = text_splitter.split_documents(header_splits)

        # 추가 메타데이터 병합
        if extra_metadata:
            for chunk in final_chunks:
                chunk.metadata.update(extra_metadata)

        print(f"✂️ Chunking completed: Created {len(final_chunks)} chunks using SemanticChunker.")
        return final_chunks
