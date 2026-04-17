import os
import re
import base64
import requests
from bisect import bisect_right
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import fitz  # pymupdf
from ..utils.azure_clients import AzureClientFactory
from ..config import Config

class HybridDocumentParser:
    """
    다양한 문서 형식(PDF, PPTX, DOCX)을 처리하여 구조화된 세그먼트를 추출하는 파서.
    Azure Document Intelligence(Layout Model) + GPT-5.4(Vision + Reasoning)를 결합하여 사용.

    [2026-01 v3.0 업데이트]
    - GPT-5.4 Vision 사용 (향상된 이미지 분석)
    - Structured Outputs 지원
    - max_completion_tokens 파라미터 사용
    - 향상된 한국어 OCR 지원

    [2026-02 v4.0.1 최적화]
    - import re 모듈 최상단 이동

    [2026-04 v6.0 — Content Understanding]
    - Azure Content Understanding 통합 (Document Intelligence 진화형)
    - 텍스트 + 이미지 + 오디오 + 비디오 통합 분석
    - CONTENT_UNDERSTANDING_ENABLED=true 시 우선 사용
    """

    def __init__(self, vision_model: str = None):
        self.di_client = AzureClientFactory.get_di_client()
        # GPT-5.4 지원을 위해 is_advanced=True로 클라이언트 초기화
        self.aoai_client = AzureClientFactory.get_openai_client(is_advanced=True)
        # Vision 모델: GPT-5.4 기본 사용 (Config.VISION_MODEL)
        self.gpt_model = vision_model or Config.MODELS.get(Config.VISION_MODEL, Config.VISION_MODEL)

        # [v6.0] Content Understanding 클라이언트 초기화
        self._cu_enabled = (
            Config.CONTENT_UNDERSTANDING_ENABLED
            and Config.CONTENT_UNDERSTANDING_ENDPOINT
            and Config.CONTENT_UNDERSTANDING_KEY
        )
        if self._cu_enabled:
            self._cu_endpoint = Config.CONTENT_UNDERSTANDING_ENDPOINT.rstrip("/")
            self._cu_key = Config.CONTENT_UNDERSTANDING_KEY
            print("🧠 Content Understanding 활성화됨")

    def _analyze_with_content_understanding(self, file_path: str) -> Optional[List[Dict[str, Any]]]:
        """
        [v6.0] Azure Content Understanding API로 문서를 분석합니다.
        Document Intelligence의 진화형으로, 텍스트·이미지·표·차트를 통합 분석합니다.

        Args:
            file_path: 분석할 파일 경로

        Returns:
            세그먼트 리스트 또는 None (실패 시 기존 DI 파이프라인으로 폴백)
        """
        if not self._cu_enabled:
            return None

        try:
            print(f"🧠 Content Understanding 분석 시작: {os.path.basename(file_path)}")

            # 파일 업로드 및 분석 요청
            url = f"{self._cu_endpoint}/contentunderstanding/analyzers/document:analyze"
            params = {"api-version": "2025-05-01-preview"}
            headers = {
                "Ocp-Apim-Subscription-Key": self._cu_key,
                "Content-Type": "application/json",
            }

            # 파일을 base64로 인코딩하여 전송
            with open(file_path, "rb") as f:
                file_data = base64.b64encode(f.read()).decode("utf-8")

            ext = os.path.splitext(file_path)[1].lower()
            mime_map = {
                ".pdf": "application/pdf",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
            }
            mime_type = mime_map.get(ext, "application/octet-stream")

            body = {
                "url": f"data:{mime_type};base64,{file_data}",
            }

            resp = requests.post(url, headers=headers, params=params, json=body, timeout=120)

            if resp.status_code == 202:
                # 비동기 작업 — Operation-Location 헤더에서 폴링
                operation_url = resp.headers.get("Operation-Location", "")
                if operation_url:
                    return self._poll_content_understanding(operation_url)

            elif resp.status_code == 200:
                result = resp.json()
                return self._parse_cu_result(result)

            print(f"   ⚠️ Content Understanding 응답 코드: {resp.status_code}")
            return None

        except Exception as e:
            print(f"   ⚠️ Content Understanding 실패, DI 폴백: {e}")
            return None

    def _poll_content_understanding(self, operation_url: str, max_wait: int = 120) -> Optional[List[Dict[str, Any]]]:
        """Content Understanding 비동기 작업 폴링"""
        import time
        headers = {"Ocp-Apim-Subscription-Key": self._cu_key}
        elapsed = 0
        interval = 2

        while elapsed < max_wait:
            time.sleep(interval)
            elapsed += interval
            resp = requests.get(operation_url, headers=headers, timeout=30)
            if resp.status_code != 200:
                continue
            data = resp.json()
            status = data.get("status", "")
            if status == "succeeded":
                return self._parse_cu_result(data.get("result", data))
            elif status in ("failed", "canceled"):
                print(f"   ⚠️ Content Understanding 작업 실패: {status}")
                return None
            # running → 계속 대기

        print(f"   ⚠️ Content Understanding 타임아웃 ({max_wait}초)")
        return None

    def _parse_cu_result(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Content Understanding 분석 결과를 세그먼트 리스트로 변환"""
        segments = []
        contents = result.get("contents", result.get("analyzeResult", {}).get("contents", []))

        for idx, content in enumerate(contents):
            content_type = content.get("type", "text")
            text = content.get("content", content.get("text", ""))
            page = content.get("pageNumber", content.get("page", 1))

            if content_type in ("table",):
                seg_type = "table"
            elif content_type in ("figure", "image", "chart"):
                seg_type = "image"
                desc = content.get("description", "")
                if desc:
                    text = f"> **[이미지/차트 설명 {idx+1}]**\n> {desc}"
            elif content_type in ("title", "sectionHeading"):
                seg_type = "header"
            else:
                seg_type = "text"

            if text:
                segments.append({
                    "type": seg_type,
                    "content": text,
                    "offset": idx * 100,
                    "page": page,
                })

        print(f"   🧠 Content Understanding 완료: {len(segments)}개 세그먼트 추출")
        return segments

    def _encode_image_base64(self, pil_image: Image.Image) -> str:
        """PIL 이미지를 Base64 문자열로 인코딩합니다."""
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _describe_image(self, pil_image: Image.Image, image_idx: int, context_hint: str = "") -> str:
        """GPT-4o를 사용하여 이미지의 핵심 인사이트를 텍스트로 추출"""
        print(f"   🤖 Model Analyzing Figure #{image_idx}...")
        base64_img = self._encode_image_base64(pil_image)

        system_prompt = """당신은 한글 문서 분석 전문가입니다. 주어진 이미지를 보고 RAG(검색 증강 생성) 시스템이 이해할 수 있도록 상세하게 설명하세요.

[이미지 유형별 분석 가이드]
1. **소프트웨어 스크린샷/UI 화면**:
   - 어떤 프로그램/메뉴인지 명시
   - 클릭해야 할 버튼, 메뉴 경로, 설정값을 정확히 기술
   - 단계별 조작 방법이 보이면 순서대로 설명

2. **표(Table)**:
   - 행/열 구조를 파악하고 데이터를 텍스트로 변환
   - 번호(No.), 항목명, 설명 등 컬럼 정보 유지
   - 셀 병합이 있으면 해당 관계 설명

3. **다이어그램/순서도**:
   - 화살표 방향, 흐름 순서 설명
   - 각 단계/노드의 내용 기술

4. **차트/그래프**:
   - 데이터 수치, 추세, 핵심 메시지 서술
   - 범례, 축 레이블 정보 포함

[출력 규칙]
- 한국어로 명확하게 서술
- 검색에 유용한 키워드를 포함
- 단순 시각 묘사보다 '무엇을 할 수 있는지', '어떤 정보인지' 중심으로 설명"""

        user_prompt = "이 이미지의 내용을 상세히 설명해줘."
        if context_hint:
            user_prompt += f"\n\n참고 문맥: {context_hint}"

        try:
            # GPT-5.x 시리즈는 max_completion_tokens 사용
            is_gpt5 = "gpt-5" in self.gpt_model.lower()

            completion_params = {
                "model": self.gpt_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]}
                ],
                "temperature": 0.0,
            }

            # GPT-5.x: max_completion_tokens, 기타: max_tokens
            if is_gpt5:
                completion_params["max_completion_tokens"] = 2000
            else:
                completion_params["max_tokens"] = 2000

            response = self.aoai_client.chat.completions.create(**completion_params)
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if "ResponsibleAIPolicyViolation" in error_msg or "content_filter" in error_msg:
                print(f"   ⚠️ Image analysis skipped: Content filter violation (Responsible AI Policy).")
                return "[이미지 분석 생략: Azure OpenAI 콘텐츠 필터링 정책에 의해 차단되었습니다.]"

            print(f"   ❌ Error analyzing image: {e}")
            return "[이미지 분석 실패]"

    def _extract_context_around_offset(self, segments: List[Dict[str, Any]], target_offset: int, window: int = 300) -> str:
        """특정 오프셋 주변의 텍스트 문맥을 추출합니다."""
        context_parts = []
        for seg in segments:
            seg_offset = seg.get("offset", 0)
            if abs(seg_offset - target_offset) < window and seg.get("type") in ["text", "header"]:
                context_parts.append(seg.get("content", "")[:200])
        return " ".join(context_parts)[:400]

    def _enhance_numbered_content(self, content: str, role: str = None) -> Tuple[str, str]:
        """번호 목록 형식을 개선하고 유형을 반환합니다."""

        # "06 제목" 또는 "06. 제목" 형식 감지
        numbered_pattern = r'^(\d{2})\.?\s+(.+)$'
        match = re.match(numbered_pattern, content.strip())

        if match:
            num, title = match.groups()
            enhanced_content = f"### {num}. {title}"
            return enhanced_content, "numbered_section"

        return content, "text" if role is None else role

    def _normalize_polygon(self, polygon: Any) -> List[Dict[str, float]]:
        """Azure DI polygon을 직렬화 가능한 x/y 포인트 목록으로 정규화합니다."""
        if not polygon:
            return []

        points: List[Dict[str, float]] = []
        if all(hasattr(point, 'x') and hasattr(point, 'y') for point in polygon):
            for point in polygon:
                points.append({"x": float(point.x), "y": float(point.y)})
            return points

        if len(polygon) >= 2 and isinstance(polygon[0], (int, float)):
            for x_coord, y_coord in zip(polygon[0::2], polygon[1::2]):
                points.append({"x": float(x_coord), "y": float(y_coord)})

        return points

    def _polygon_to_bounding_box(self, polygon_points: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
        """정규화된 polygon 포인트를 사각 bounding box로 변환합니다."""
        if not polygon_points:
            return None

        x_coords = [point["x"] for point in polygon_points]
        y_coords = [point["y"] for point in polygon_points]
        return {
            "left": min(x_coords),
            "top": min(y_coords),
            "right": max(x_coords),
            "bottom": max(y_coords),
        }

    def _extract_layout_metadata(
        self,
        bounding_regions: Any,
        page_map: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """bounding_regions를 청크/세그먼트에서 재사용할 수 있는 메타데이터로 정규화합니다."""
        source_regions = []

        for region in (bounding_regions or []):
            polygon = self._normalize_polygon(getattr(region, "polygon", None))
            page_info = page_map.get(region.page_number, {})
            source_regions.append({
                "page_number": region.page_number,
                "polygon": polygon,
                "bounding_box": self._polygon_to_bounding_box(polygon),
                "unit": page_info.get("unit"),
                "page_width": page_info.get("width"),
                "page_height": page_info.get("height"),
            })

        if not source_regions:
            return {}

        primary_region = source_regions[0]
        metadata = {
            "source_regions": source_regions,
            "bounding_box": primary_region.get("bounding_box"),
            "polygon": primary_region.get("polygon"),
        }
        if primary_region.get("unit") is not None:
            metadata["page_unit"] = primary_region["unit"]
        return metadata

    def _pdf_to_images(self, file_path: str, dpi: int = 200) -> Optional[List[Image.Image]]:
        """는 PDF 페이지를 이미지로 변환합니다."""
        doc = None
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

            return images
        except Exception as e:
            print(f"   ⚠️ PDF Image conversion failed (Visual RAG will be skipped): {e}")
            return None
        finally:
            if doc:
                doc.close()

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        파일 형식(PDF, PPTX, DOCX)에 따라 하이브리드 파싱 수행
        반환값: List[Dict] (구조화된 문서 세그먼트 리스트)

        [v6.0] Content Understanding가 활성화된 경우 우선 사용,
               실패 시 기존 DI + GPT Vision 파이프라인으로 폴백
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        print(f"🚀 Parsing started: {file_path} ({file_ext})")

        # [v6.0] Content Understanding 우선 시도
        if self._cu_enabled:
            cu_result = self._analyze_with_content_understanding(file_path)
            if cu_result:
                return cu_result
            print("   ↩️ Content Understanding 폴백 → DI + GPT Vision 파이프라인")

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
        page_map = {
            page.page_number: {
                "width": getattr(page, "width", None),
                "height": getattr(page, "height", None),
                "unit": getattr(page, "unit", None),
            }
            for page in (result.pages or [])
        }

        segments = []

        # 3. Tables 추출
        # 테이블의 오프셋 범위를 저장하여 나중에 텍스트 중복 제거에 사용
        table_spans = []
        for table in (result.tables or []):
            # 테이블 전체 스팬 계산 (최소 시작점 ~ 최대 끝점)
            if not table.spans: continue

            start_offset = min(span.offset for span in table.spans)
            layout_metadata = self._extract_layout_metadata(table.bounding_regions, page_map)

            # 테이블 컨텐츠 재구성 (Azure가 제공하는 cells 활용)
            table_content = self._table_to_markdown(table)

            segments.append({
                "type": "table",
                "content": table_content,
                "offset": start_offset,
                "page": table.bounding_regions[0].page_number if table.bounding_regions else 1,
                **layout_metadata,
            })

            # 스팬 범위 저장 (이진 탐색 최적화를 위해 start 기준 정렬 후 사용)
            for span in table.spans:
                table_spans.append((span.offset, span.offset + span.length))

        # 이진 탐색을 위해 start 기준 정렬
        table_spans.sort()

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

            # 번호 목록 형식 개선 (06, 07 등)
            enhanced_content, detected_type = self._enhance_numbered_content(content, role)
            if detected_type == "numbered_section":
                seg_type = "header"
                content = enhanced_content

            layout_metadata = self._extract_layout_metadata(para.bounding_regions, page_map)

            segments.append({
                "type": seg_type,
                "content": content,
                "role": role, # 헤더 레벨 추론을 위해 저장
                "offset": para.spans[0].offset,
                "page": para.bounding_regions[0].page_number if para.bounding_regions else 1,
                **layout_metadata,
            })

        # 5. Figure(이미지/차트) 감지 및 GPT-4o 처리
        if (result.figures or []) and page_images:
            print(f"📊 Found {len(result.figures)} figures. Starting vision analysis...")

            for idx, figure in enumerate(result.figures):
                if not figure.bounding_regions: continue

                region = figure.bounding_regions[0]
                page_num = region.page_number - 1
                layout_metadata = self._extract_layout_metadata(figure.bounding_regions, page_map)

                if page_num >= len(page_images):
                    continue

                page_img = page_images[page_num]
                di_page = result.pages[page_num]

                # 좌표 변환 및 이미징 크롭
                normalized_polygon = self._normalize_polygon(region.polygon)
                if not normalized_polygon: continue

                x_coords = [point["x"] for point in normalized_polygon]
                y_coords = [point["y"] for point in normalized_polygon]

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

                    # 이미지 주변 문맥 추출
                    start_offset = figure.spans[0].offset if figure.spans else 0
                    context_hint = self._extract_context_around_offset(segments, start_offset)

                    # 문맥 힌트와 함께 이미지 분석
                    desc_text = self._describe_image(cropped_img, idx + 1, context_hint)

                    # Figure 세그먼트 추가
                    segments.append({
                        "type": "image",
                        "content": f"> **[이미지/차트 설명 {idx+1}]**\n> {desc_text}",
                        "offset": start_offset,
                        "page": region.page_number,
                        **layout_metadata,
                    })

                except Exception as e:
                    print(f"   ⚠️ Error cropping/analyzing figure {idx+1}: {e}")

        # 6. 오프셋 기준 정렬 (문서 순서 복원)
        segments.sort(key=lambda x: x["offset"])

        print(f"✅ Parsing completed. Extracted {len(segments)} segments.")
        return segments

    def _is_offset_in_ranges(self, offset: int, ranges: List[Tuple[int, int]]) -> bool:
        """이진 탐색으로 offset이 정렬된 범위 리스트 안에 있는지 확인 (O(log N))"""
        if not ranges:
            return False
        # 시작값 리스트를 기준으로 이진 탐색
        starts = [r[0] for r in ranges]
        idx = bisect_right(starts, offset) - 1
        if idx >= 0:
            start, end = ranges[idx]
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
