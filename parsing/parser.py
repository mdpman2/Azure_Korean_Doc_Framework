import os
import base64
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_path
from ..utils.azure_clients import AzureClientFactory
from ..config import Config

class HybridDocumentParser:
    def __init__(self):
        self.di_client = AzureClientFactory.get_di_client()
        self.aoai_client = AzureClientFactory.get_openai_client()
        self.gpt_model = Config.MODELS.get("gpt-4.1") # Default for vision/parsing

    def _encode_image_base64(self, pil_image):
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _describe_image(self, pil_image, image_idx):
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

    def parse(self, file_path):
        """
        파일 형식(PDF, PPTX, DOCX)에 따라 하이브리드 파싱 수행
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        print(f"🚀 Parsing started: {file_path} ({file_ext})")

        # 1. Image 변환 (PDF인 경우만 Visual RAG용)
        page_images = None
        if file_ext == '.pdf':
            try:
                page_images = convert_from_path(file_path, dpi=200)
            except Exception as e:
                print(f"   ⚠️ PDF Image conversion failed (Visual RAG will be skipped): {e}")

        # 2. Azure Document Intelligence 실행
        with open(file_path, "rb") as f:
            poller = self.di_client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=f,
                content_type="application/octet-stream",
                output_content_format="markdown"
            )
        result = poller.result()

        full_markdown = result.content
        descriptions = []

        # 3. Figure(이미지/차트) 감지 및 GPT-4o 처리
        if result.figures and page_images:
            print(f"📊 Found {len(result.figures)} figures. Starting vision analysis...")

            for idx, figure in enumerate(result.figures):
                if not figure.bounding_regions: continue

                region = figure.bounding_regions[0]
                page_num = region.page_number - 1

                if page_num >= len(page_images):
                    continue

                page_img = page_images[page_num]
                di_page = result.pages[page_num]

                polygon = region.polygon
                x_coords = [p.x for p in polygon]
                y_coords = [p.y for p in polygon]

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
                    insertion_block = f"\n\n> **[이미지/차트 설명 {idx+1}]**\n> {desc_text}\n\n"
                    offset = figure.spans[0].offset if figure.spans else len(full_markdown)
                    descriptions.append((offset, insertion_block))

                except Exception as e:
                    print(f"   ⚠️ Error cropping/analyzing figure {idx+1}: {e}")

        elif result.figures and not page_images:
             print(f"ℹ️ Figures detected but Visual RAG skipped for non-PDF format or missing poppler: {file_ext}")

        # 4. 설명 텍스트 병합
        descriptions.sort(key=lambda x: x[0], reverse=True)

        for offset, text in descriptions:
            if offset <= len(full_markdown):
                full_markdown = full_markdown[:offset] + text + full_markdown[offset:]
            else:
                full_markdown += text

        print("✅ Parsing completed.")
        return full_markdown
