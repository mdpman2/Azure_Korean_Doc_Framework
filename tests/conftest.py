"""pytest 공통 설정.

tests/ 디렉토리 하위에서 `pytest` 또는 `uv run pytest`를 실행할 때
azure_korean_doc_framework 패키지를 import할 수 있도록 상위(프로젝트 루트)
디렉토리를 sys.path 앞쪽에 추가합니다. conftest.py는 테스트 수집 전에
자동으로 로드되므로 각 테스트 파일의 경로 부트스트랩이 없어도 동작합니다.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
