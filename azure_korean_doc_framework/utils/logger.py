import os
import json
from typing import List
from ..core.schema import Document

class ChunkLogger:
    """
    청크(Document 객체 리스트)를 JSON 파일로 저장하는 로깅 유틸리티입니다.
    디버깅 및 데이터 검증 목적으로 사용됩니다.
    """

    @staticmethod
    def save_chunks_to_json(chunks: List[Document], filename: str, output_dir: str = "output"):
        """
        주어진 청크 리스트를 JSON 파일로 저장합니다.

        Args:
            chunks (List[Document]): 저장할 LangChain Document 객체 리스트
            filename (str): 원본 파일명 (이 이름을 기반으로 로그 파일명이 생성됨)
            output_dir (str): 저장할 디렉토리 경로 (기본값: "output")
        """
        try:
            os.makedirs(output_dir, exist_ok=True)

            # 안전한 파일명 생성
            safe_filename = os.path.basename(filename)
            json_filename = f"{safe_filename}_chunks.json"
            json_path = os.path.join(output_dir, json_filename)

            chunks_data = []
            for c in chunks:
                chunks_data.append({
                    "page_content": c.page_content,
                    "metadata": c.metadata
                })

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(chunks_data, f, ensure_ascii=False, indent=2)

            print(f"📄 청크 로그 저장 완료: {json_path}")
            return json_path

        except Exception as e:
            print(f"⚠️ 청크 로그 저장 실패 ({filename}): {e}")
            return None
