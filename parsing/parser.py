import os
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import fitz  # pymupdf
from ..utils.azure_clients import AzureClientFactory
from ..config import Config

class HybridDocumentParser:
    """
    다양한 문서 형식(PDF, PPTX, DOCX)을 처리하여 구조화된 세그먼트를 추출하는 파서.
    Azure Document Intelligence(Layout Model)와 GPT-4o(Vision)를 결합하여 사용.
    """

    def __init__(self):
        self.di_client = AzureClientFactory.get_di_client()
        self.aoai_client = AzureClientFactory.get_openai_client()
        self.gpt_model = Config.MODELS.get("gpt-4.1") # Default for vision/parsing

    def _encode_image_base64(self, pil_image: Image.Image) -> str:
        """PIL 이미지를 Base64 문자열로 인코딩합니다."""
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _describe_image(self, pil_image: Image.Image, image_idx: int) -> str:
        """GPT-4o를 사용하여 이미지의 핵심 인사이트를 텍스트로 추출"""
        print(f"   🤖 Model Analyzing Figure #{image_idx}...")
        base64_img = self._encode_image_base64(pil_image)

        system_prompt = (
            "당신은 데이터 분석 전문가입니다. 주어진 이미지(차트, 표, 다이어그램 등)를 보고 "
            "RAG(검색 증강 생성) 시스템이 이해할 수 있도록 상세하게 설명하세요. "
            "단순한 시각적 묘사보다는, '데이터의 수치', '추세', '핵심 메시지'를 한국어로 명확히 서술하세요."
        )

        try:
            response = self.aoai_client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": "이 이미지의 내용을 상세히 설명해줘."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"   ❌ Error analyzing image: {e}")
            return "[이미지 분석 실패]"

    def _pdf_to_images(self, file_path: str, dpi: int = 200) -> Optional[List[Image.Image]]:
        """PyMuPDF를 사용하여 PDF 페이지를 이미지로 변환합니다."""
        try:
            doc = fitz.open(file_path)
            images = []
            zoom = dpi / 72  # PDF는 기본 72 DPI
            matrix = fitz.Matrix(zoom, zoom)

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=matrix)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)

            doc.close()
            return images
        except Exception as e:
            print(f"   ⚠️ PDF Image conversion failed (Visual RAG will be skipped): {e}")
            return None

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        파일 형식(PDF, PPTX, DOCX)에 따라 하이브리드 파싱 수행
        반환값: List[Dict] (구조화된 문서 세그먼트 리스트)
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        print(f"🚀 Parsing started: {file_path} ({file_ext})")

        # 1. Image 변환 (PDF인 경우만 Visual RAG용)
        page_images = None
        if file_ext == '.pdf':
            page_images = self._pdf_to_images(file_path, dpi=200)

        # 2. Azure Document Intelligence 실행
        with open(file_path, "rb") as f:
            poller = self.di_client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=f,
                content_type="application/octet-stream",
                output_content_format="markdown" # Markdown 포맷 사용
            )
        result = poller.result()

        segments = []

        # 3. Tables 추출
        # 테이블의 오프셋 범위를 저장하여 나중에 텍스트 중복 제거에 사용
        table_spans = []
        for table in (result.tables or []):
            # 테이블 전체 스팬 계산 (최소 시작점 ~ 최대 끝점)
            if not table.spans: continue

            start_offset = min(span.offset for span in table.spans)

            # 테이블 컨텐츠 재구성 (Azure가 제공하는 cells 활용)
            table_content = self._table_to_markdown(table)

            segments.append({
                "type": "table",
                "content": table_content,
                "offset": start_offset,
                "page": table.bounding_regions[0].page_number if table.bounding_regions else 1
            })

            # 스팬 범위 저장
            for span in table.spans:
                table_spans.append((span.offset, span.offset + span.length))

        # 4. Paragraphs (Text & Headers) 추출
        for para in (result.paragraphs or []):
            # 테이블 내부에 있는 텍스트는 제외 (중복 방지)
            if self._is_offset_in_ranges(para.spans[0].offset, table_spans):
                continue

            content = para.content
            role = para.role if hasattr(para, 'role') else None

            seg_type = "text"
            if role in ["sectionHeading", "title", "pageHeader"]:
                seg_type = "header"

            segments.append({
                "type": seg_type,
                "content": content,
                "role": role, # 헤더 레벨 추론을 위해 저장
                "offset": para.spans[0].offset,
                "page": para.bounding_regions[0].page_number if para.bounding_regions else 1
            })

        # 5. Figure(이미지/차트) 감지 및 GPT-4o 처리
        if (result.figures or []) and page_images:
            print(f"📊 Found {len(result.figures)} figures. Starting vision analysis...")

            for idx, figure in enumerate(result.figures):
                if not figure.bounding_regions: continue

                region = figure.bounding_regions[0]
                page_num = region.page_number - 1

                if page_num >= len(page_images):
                    continue

                page_img = page_images[page_num]
                di_page = result.pages[page_num]

                # 좌표 변환 및 이미징 크롭
                polygon = region.polygon
                if not polygon: continue

                x_coords = []
                y_coords = []

                # polygon 포맷 대응 (객체 리스트 vs 단순 float 리스트)
                if all(hasattr(p, 'x') and hasattr(p, 'y') for p in polygon):
                    x_coords = [p.x for p in polygon]
                    y_coords = [p.y for p in polygon]
                elif len(polygon) >= 2 and isinstance(polygon[0], (int, float)):
                    # [x1, y1, x2, y2, ...] 형태인 경우
                    x_coords = polygon[0::2]
                    y_coords = polygon[1::2]

                if not x_coords or not y_coords:
                    continue

                if di_page.width == 0 or di_page.height == 0:
                    continue

                scale_x = page_img.width / di_page.width
                scale_y = page_img.height / di_page.height

                left = min(x_coords) * scale_x
                top = min(y_coords) * scale_y
                right = max(x_coords) * scale_x
                bottom = max(y_coords) * scale_y

                try:
                    cropped_img = page_img.crop((left, top, right, bottom))
                    if cropped_img.width < 50 or cropped_img.height < 50:
                        continue

                    desc_text = self._describe_image(cropped_img, idx + 1)

                    # Figure 세그먼트 추가
                    start_offset = figure.spans[0].offset if figure.spans else 0
                    segments.append({
                        "type": "image",
                        "content": f"> **[이미지/차트 설명 {idx+1}]**\n> {desc_text}",
                        "offset": start_offset,
                        "page": region.page_number
                    })

                except Exception as e:
                    print(f"   ⚠️ Error cropping/analyzing figure {idx+1}: {e}")

        # 6. 오프셋 기준 정렬 (문서 순서 복원)
        segments.sort(key=lambda x: x["offset"])

        print(f"✅ Parsing completed. Extracted {len(segments)} segments.")
        return segments

    def _is_offset_in_ranges(self, offset: int, ranges: List[Tuple[int, int]]) -> bool:
        for start, end in ranges:
            if start <= offset < end:
                return True
        return False

    def _table_to_markdown(self, table: Any) -> str:
        """Azure DI Table 객체를 Markdown 형식 문자열로 변환"""
        if not table.cells:
            return ""

        # 행/열 개수 파악
        rows = table.row_count
        cols = table.column_count

        # 2차원 배열 생성
        grid = [["" for _ in range(cols)] for _ in range(rows)]

        for cell in table.cells:
            # 셀 내용 (줄바꿈은 공백으로 대체하여 표 깨짐 방지)
            content = cell.content.replace('\n', ' ')

            # 병합된 셀 처리 (단순히 첫 번째 칸에만 내용 삽입하거나, 모두 채울 수 있음. 여기선 첫 칸만)
            r_idx = cell.row_index
            c_idx = cell.column_index

            if r_idx < rows and c_idx < cols:
                grid[r_idx][c_idx] = content

        # Markdown 문자열 조립
        md_lines = []

        # 헤더 (첫 번째 행)
        header_row = "|" + "|".join(grid[0]) + "|"
        md_lines.append(header_row)

        # 구분선
        separator = "|" + "|".join(["---"] * cols) + "|"
        md_lines.append(separator)

        # 데이터 행
        for i in range(1, rows):
            row_str = "|" + "|".join(grid[i]) + "|"
            md_lines.append(row_str)

        return "\n".join(md_lines)
