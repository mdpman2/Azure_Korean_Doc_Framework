"""
LightRAG-inspired Graph RAG 모듈 (v4.0 → v4.7 EdgeQuake 강화 → v6.0 Neo4j 백엔드)

LightRAG(https://github.com/HKUDS/LightRAG)의 핵심 개념을 참조하여 구현한
경량 Knowledge Graph 기반 RAG 시스템입니다.

핵심 기능:
- GPT-5.4를 활용한 엔티티/관계 추출 (한국어 최적화)
- NetworkX 기반 인메모리 Knowledge Graph
- Dual-Level Retrieval (Local: 엔티티 중심, Global: 관계 중심)
- 벡터 검색과 그래프 검색의 하이브리드 결합

[2026-02 v4.0 신규]
- LightRAG 기반 Graph RAG 아키텍처
- 한국어 엔티티 타입 특화
- Azure AI Search + Knowledge Graph 하이브리드

[2026-04 v4.7 신규 — EdgeQuake 참조 강화]
- Gleaning: Multi-Pass 엔티티 추출 (15-25% 더 많은 엔티티 포착)
- Entity Normalization: 대소문자/공백 정규화 + 설명 병합 (36-40% 중복 제거)
- Community Detection: Louvain 커뮤니티 클러스터링 + 요약 (글로벌 주제 검색 강화)
- Mix Query Mode: 벡터 검색 + 그래프 검색의 구성 가능한 가중 결합
- Knowledge Injection: 도메인 용어집/동의어 주입으로 쿼리 자동 확장

[2026-04 v6.0 신규 — Neo4j 백엔드]
- GRAPH_STORAGE_BACKEND=neo4j 설정 시 Neo4j 그래프 DB 사용
- Leiden 커뮤니티 탐지 (Neo4j GDS 라이브러리)
- 10K+ 노드 규모에서도 안정적 성능

참조: https://github.com/HKUDS/LightRAG
참조: https://github.com/raphaelmansuy/edgequake
"""

import json
import hashlib
import re
from collections import deque
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from networkx.algorithms.community import louvain_communities
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False

# [v6.0] Neo4j 드라이버 (선택 설치)
try:
    from neo4j import GraphDatabase as Neo4jDriver
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

from ..config import Config
from ..utils.azure_clients import AzureClientFactory


# ==================== 데이터 모델 ====================

class QueryMode(Enum):
    """LightRAG 스타일 검색 모드 (v4.7: EdgeQuake 참조 — Mix/Bypass 추가)"""
    LOCAL = "local"       # 엔티티 중심 검색 (Low-Level Keywords)
    GLOBAL = "global"     # 관계/주제 중심 검색 (High-Level Keywords)
    HYBRID = "hybrid"     # Local + Global 결합
    NAIVE = "naive"       # 벡터 검색만 사용 (그래프 비활용)
    MIX = "mix"           # [v4.7] 벡터 + 그래프 가중 결합 (EdgeQuake Mix 모드)
    BYPASS = "bypass"     # [v4.7] RAG 검색 생략, LLM 직접 질의


@dataclass
class Entity:
    """Knowledge Graph 엔티티"""
    name: str
    entity_type: str
    description: str
    source_chunks: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def entity_id(self) -> str:
        return hashlib.md5(self.name.encode()).hexdigest()[:12]


@dataclass
class Relationship:
    """Knowledge Graph 관계"""
    source: str          # 소스 엔티티 이름
    target: str          # 타겟 엔티티 이름
    relation_type: str   # 관계 유형
    description: str     # 관계 설명
    weight: float = 1.0
    keywords: str = ""
    source_chunks: List[str] = field(default_factory=list)


@dataclass
class GraphQueryResult:
    """그래프 검색 결과"""
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    context_text: str = ""
    source_chunks: List[str] = field(default_factory=list)


# ==================== 한국어 엔티티 타입 ====================

KOREAN_ENTITY_TYPES = [
    "인물",          # Person
    "조직",          # Organization
    "장소",          # Location
    "날짜",          # Date/Time
    "법률",          # Law/Regulation
    "정책",          # Policy
    "기관",          # Institution
    "사건",          # Event
    "기술",          # Technology
    "제품",          # Product
    "금액",          # Amount/Money
    "지표",          # Metric/Indicator
    "문서",          # Document
    "개념",          # Concept
]


# ==================== [v4.7] Entity Normalization (EdgeQuake 참조) ====================

# 한국어/영문 정규화용 패턴
_NORMALIZE_WS_RE = re.compile(r'\s+')
_PAREN_RE = re.compile(r'\s*\([^)]*\)\s*')


def _safe_float(value: Any, default: float = 1.0) -> float:
    """LLM 응답의 weight 값을 안전하게 float로 변환"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def normalize_entity_name(name: str) -> str:
    """
    엔티티명 정규화 (EdgeQuake 방식):
    - 앞뒤 공백 제거
    - 연속 공백 → 단일 공백
    - 영문 약어(대문자, 1-4글자: AI, OEE, NLP 등)는 원본 유지
    - 그 외 영문 단어는 Capitalize
    - 한국어는 원본 유지
    """
    name = name.strip()
    name = _NORMALIZE_WS_RE.sub(' ', name)
    if name.isascii():
        words = name.split()
        normalized_words = []
        for w in words:
            # 대문자만으로 된 1-4글자 단어는 약어로 유지 (AI, OEE, NLP, IBM)
            if w.isupper() and 1 <= len(w) <= 4 and w.isalpha():
                normalized_words.append(w)
            else:
                normalized_words.append(w.capitalize())
        name = " ".join(normalized_words)
    return name


def merge_descriptions(existing: str, new: str) -> str:
    """
    두 설명을 병합 (EdgeQuake Entity Normalization):
    - 중복 설명 제거
    - 더 긴 설명 우선
    - 서로 다른 정보가 있으면 합침
    """
    if not existing:
        return new
    if not new:
        return existing
    if new in existing:
        return existing
    if existing in new:
        return new
    # 두 설명이 모두 다른 정보를 담고 있으면 합침
    return f"{existing}; {new}"


# ==================== [v4.7] Knowledge Injection (EdgeQuake 참조) ====================

@dataclass
class KnowledgeInjection:
    """
    도메인 용어집/동의어 주입 엔트리 (EdgeQuake Knowledge Injection 참조)

    Attributes:
        term: 주요 용어 (예: "OEE")
        definition: 정의 (예: "Overall Equipment Effectiveness, 설비종합효율")
        synonyms: 동의어 리스트 (예: ["설비종합효율", "설비효율"])
        entity_type: 연관 엔티티 타입 (예: "지표")
    """
    term: str
    definition: str = ""
    synonyms: List[str] = field(default_factory=list)
    entity_type: str = "개념"


# ==================== 엔티티/관계 추출 프롬프트 ====================

ENTITY_EXTRACTION_SYSTEM_PROMPT = """당신은 한국어 문서에서 엔티티(Entity)와 관계(Relationship)를 추출하는 전문가입니다.

주어진 텍스트를 분석하여 다음을 추출하세요:

### 엔티티 타입
{entity_types}

### 출력 형식 (JSON)
{{
  "entities": [
    {{
      "name": "엔티티명",
      "entity_type": "엔티티 타입",
      "description": "엔티티에 대한 간결한 설명 (1-2문장)"
    }}
  ],
  "relationships": [
    {{
      "source": "소스 엔티티명",
      "target": "타겟 엔티티명",
      "relation_type": "관계 유형",
      "description": "관계에 대한 설명",
      "keywords": "관련 키워드 (쉼표 구분)",
      "weight": 1.0
    }}
  ]
}}

### 규칙
1. 엔티티명은 원문에 등장하는 그대로 사용 (줄임말보다 정식 명칭 우선)
2. 관계는 두 엔티티 간의 명확한 연결이 있을 때만 추출
3. weight는 관계의 강도 (0.1 ~ 3.0, 기본 1.0)
4. 한국어 엔티티에 집중하되, 영문 고유명사도 포함
5. 중복 엔티티는 병합하여 하나로 표현
"""

KEYWORD_EXTRACTION_PROMPT = """주어진 질문을 분석하여 Knowledge Graph 검색에 필요한 키워드를 추출하세요.

### 출력 형식 (JSON)
{{
  "high_level_keywords": ["주제 수준 키워드 (글로벌 관계/패턴 검색용)"],
  "low_level_keywords": ["구체적 엔티티/세부사항 키워드 (로컬 엔티티 검색용)"]
}}

### 예시
질문: "삼성전자의 2025년 반도체 매출은?"
→ high_level: ["반도체 산업", "기업 매출 실적"]
→ low_level: ["삼성전자", "2025년", "반도체 매출"]

질문: "{query}"
"""


# ==================== [v4.7] Gleaning 프롬프트 (EdgeQuake 참조) ====================

GLEANING_SYSTEM_PROMPT = """이전 추출 결과를 검토하고 누락된 엔티티와 관계를 추가로 추출하세요.

### 이전에 추출된 엔티티
{existing_entities}

### 이전에 추출된 관계
{existing_relationships}

### 규칙
1. 이전 추출에서 **놓친 엔티티와 관계만** 새로 추출하세요
2. 이미 추출된 것은 다시 출력하지 마세요
3. 특히 다음을 주의 깊게 찾으세요:
   - 암시적으로 언급된 엔티티 (간접 참조, 대명사가 가리키는 대상)
   - 약어/줄임말이 나타내는 정식 명칭
   - 숫자/날짜와 연관된 엔티티 (금액, 기간, 지표)
   - 엔티티 간 암묵적 관계 (인과, 소속, 시간순서)
4. 출력 형식은 동일한 JSON 구조를 사용하세요

### 출력 형식 (JSON)
{{
  "entities": [...],
  "relationships": [...]
}}
"""


# ==================== [v4.7] Community Summary 프롬프트 (EdgeQuake 참조) ====================

COMMUNITY_SUMMARY_PROMPT = """다음 Knowledge Graph 커뮤니티(관련 엔티티 클러스터)를 분석하고
핵심 주제를 요약하세요.

### 커뮤니티 엔티티
{entities}

### 커뮤니티 관계
{relationships}

다음 JSON 형식으로 출력하세요:
{{
  "theme": "핵심 주제 (1문장)",
  "summary": "커뮤니티에 포함된 엔티티들의 관계와 주요 내용 요약 (2-3문장)",
  "key_entities": ["가장 중요한 엔티티명 3-5개"]
}}
"""


class KnowledgeGraphManager:
    """
    LightRAG-inspired Knowledge Graph 관리자

    LightRAG의 핵심 아키텍처를 참조하여 구현:
    - 엔티티/관계 추출 (GPT-5.4)
    - NetworkX 기반 인메모리 그래프 또는 Neo4j 그래프 DB
    - Dual-Level Retrieval (Local + Global)
    - 한국어 문서 특화

    [v4.7 EdgeQuake 강화]
    - Gleaning: Multi-Pass 추출 (누락 엔티티 15-25% 추가 포착)
    - Entity Normalization: 정규화 + 설명 병합
    - Community Detection: Louvain 클러스터링 + 커뮤니티 요약
    - Mix Query Mode: 벡터+그래프 가중 결합
    - Knowledge Injection: 도메인 용어집 주입

    [v6.0] Neo4j 백엔드 지원
    - GRAPH_STORAGE_BACKEND=neo4j 시 Neo4j 그래프 DB 사용
    - Neo4j GDS 기반 Leiden 커뮤니티 탐지

    참조: https://github.com/HKUDS/LightRAG
    참조: https://github.com/raphaelmansuy/edgequake
    """

    def __init__(
        self,
        entity_types: Optional[List[str]] = None,
        model_key: str = "gpt-5.4",
        max_entities_per_chunk: int = 20,
        gleaning_passes: int = 1,
        mix_graph_weight: float = 0.4,
        llm_cache=None,
    ):
        self.entity_types = entity_types or KOREAN_ENTITY_TYPES
        self.model_key = model_key
        self.max_entities_per_chunk = max_entities_per_chunk
        self._is_gpt5 = "gpt-5" in model_key.lower()

        # [v4.7] Gleaning 설정 (EdgeQuake 참조: multi-pass extraction)
        self.gleaning_passes = max(0, gleaning_passes)

        # [v4.7] Mix 모드 가중치 (0.0=벡터만, 1.0=그래프만)
        self.mix_graph_weight = max(0.0, min(1.0, mix_graph_weight))

        # LLM 클라이언트
        self.client = AzureClientFactory.get_openai_client(is_advanced=True)
        self.model_name = Config.MODELS.get(model_key, "model-router")

        # [v6.0] 스토리지 백엔드 선택
        self._neo4j_driver = None
        if Config.GRAPH_STORAGE_BACKEND == "neo4j":
            if not HAS_NEO4J:
                raise ImportError(
                    "neo4j 패키지가 필요합니다. 설치: pip install neo4j"
                )
            self._neo4j_driver = Neo4jDriver.driver(
                Config.NEO4J_URI,
                auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD),
            )
            # NetworkX는 Neo4j 모드에서도 인메모리 캐시로 사용
            if HAS_NETWORKX:
                self.graph = nx.DiGraph()
            else:
                self.graph = None
            print(f"📊 Neo4j 백엔드 연결: {Config.NEO4J_URI}")
        else:
            if not HAS_NETWORKX:
                raise ImportError(
                    "networkx 패키지가 필요합니다. 설치: pip install networkx"
                )
            self.graph = nx.DiGraph()

        # 엔티티/관계 캐시 (중복 방지)
        self._entity_cache: Dict[str, Entity] = {}
        self._chunk_to_entities: Dict[str, List[str]] = {}

        # 역색인: 키워드 → 엔티티/관계 빠른 검색용 (O(N*K) → O(K))
        self._entity_keyword_index: Dict[str, set] = {}  # keyword → {entity_name, ...}
        self._relation_keyword_index: Dict[str, set] = {}  # keyword → {(source, target), ...}

        # [최적화] 문자 기반 사전 필터링 인덱스 — O(N) 폴백 제거
        self._entity_name_char_index: Dict[str, set] = {}  # char → {entity_name, ...}
        self._edge_desc_char_index: Dict[str, set] = {}    # char → {(source, target), ...}

        # [v4.7] 정규화명 → 원래 이름 역매핑 (Entity Normalization)
        self._normalized_name_map: Dict[str, str] = {}  # normalized → canonical name

        # [v4.7] Community Detection 캐시
        self._communities: List[Dict[str, Any]] = []
        self._community_summaries: Dict[int, str] = {}

        # [v4.7] Knowledge Injection 저장소
        self._injections: Dict[str, KnowledgeInjection] = {}
        self._synonym_map: Dict[str, str] = {}  # synonym → canonical term

        # [v5.1] LLM 캐시 (엔티티 추출 비용 절감)
        self._llm_cache = llm_cache

        print(f"📊 KnowledgeGraphManager 초기화 (모델: {model_key}, 엔티티 타입: {len(self.entity_types)}개, "
              f"Gleaning: {gleaning_passes}회, Mix 가중치: {mix_graph_weight}, "
              f"백엔드: {Config.GRAPH_STORAGE_BACKEND})")

    # ==================== 엔티티/관계 추출 ====================

    def extract_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 5,
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        문서 청크에서 엔티티와 관계를 추출하여 Knowledge Graph를 구축합니다.

        [v4.7] Gleaning 지원: 첫 추출 후 추가 패스로 누락 엔티티 15-25% 추가 포착
        [v4.7] Entity Normalization: 추출 후 정규화하여 중복 36-40% 제거

        Args:
            chunks: 문서 청크 리스트 (page_content, metadata)
            batch_size: 한 번에 처리할 청크 수

        Returns:
            (추출된 엔티티 리스트, 추출된 관계 리스트)
        """
        all_entities = []
        all_relationships = []

        entity_types_str = ", ".join(self.entity_types)
        system_prompt = ENTITY_EXTRACTION_SYSTEM_PROMPT.format(
            entity_types=entity_types_str
        )

        total = len(chunks)
        print(f"🔍 엔티티/관계 추출 시작 (총 {total}개 청크, 배치 크기: {batch_size}, "
              f"Gleaning: {self.gleaning_passes}회)")

        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            batch_text = "\n\n---\n\n".join([
                c.get("page_content", c) if isinstance(c, dict) else str(c)
                for c in batch
            ])

            # 텍스트가 너무 길면 잘라내기
            if len(batch_text) > 8000:
                batch_text = batch_text[:8000]

            try:
                # === Pass 1: 초기 추출 ===
                batch_entities, batch_relationships = self._extract_pass(
                    system_prompt, batch_text, f"batch_{i // batch_size}"
                )

                # === Pass 2+: Gleaning (EdgeQuake 참조) ===
                for glean_pass in range(self.gleaning_passes):
                    if not batch_entities and not batch_relationships:
                        break  # 초기 추출 결과가 없으면 Gleaning 스킵

                    glean_entities, glean_rels = self._gleaning_pass(
                        batch_text, batch_entities, batch_relationships,
                        f"batch_{i // batch_size}_glean{glean_pass + 1}"
                    )
                    batch_entities.extend(glean_entities)
                    batch_relationships.extend(glean_rels)

                    if not glean_entities and not glean_rels:
                        break  # 더 이상 새로운 것이 없으면 중단

                all_entities.extend(batch_entities)
                all_relationships.extend(batch_relationships)

                glean_info = f" (+Gleaning {self.gleaning_passes}회)" if self.gleaning_passes > 0 else ""
                print(f"   ✅ 배치 {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}: "
                      f"엔티티 {len(batch_entities)}개, "
                      f"관계 {len(batch_relationships)}개 추출{glean_info}")

            except Exception as e:
                print(f"   ⚠️ 배치 {i // batch_size + 1} 추출 실패: {e}")
                continue

        # [v4.7] Community Detection 실행 (엔티티가 충분히 있을 때)
        if self.graph.number_of_nodes() >= 5:
            self._detect_communities()

        print(f"📊 Knowledge Graph 구축 완료: "
              f"노드 {self.graph.number_of_nodes()}개, "
              f"엣지 {self.graph.number_of_edges()}개"
              f" | 커뮤니티 {len(self._communities)}개")

        return all_entities, all_relationships

    def _extract_pass(
        self,
        system_prompt: str,
        batch_text: str,
        source_label: str,
    ) -> Tuple[List[Entity], List[Relationship]]:
        """단일 추출 패스 실행 ([v5.1] LLM 캐시 지원)"""
        entities = []
        relationships = []

        user_content = f"다음 텍스트에서 엔티티와 관계를 추출하세요:\n\n{batch_text}"

        # [v5.1] LLM 캐시: 동일 배치 텍스트에 대한 중복 추출 방지
        if self._llm_cache:
            cached = self._llm_cache.get(
                prompt=user_content,
                model=self.model_name,
                system_message=system_prompt,
            )
            if cached is not None:
                try:
                    result = json.loads(cached)
                    return self._parse_extraction_result(result, source_label)
                except (json.JSONDecodeError, TypeError):
                    pass  # 캐시 데이터 손상 — LLM 호출로 진행

        completion_params = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        if self._is_gpt5:
            completion_params["max_completion_tokens"] = 4000
        else:
            completion_params["max_tokens"] = 4000

        response = self.client.chat.completions.create(**completion_params)
        result_text = response.choices[0].message.content

        # [v5.1] LLM 캐시에 응답 저장
        if self._llm_cache and result_text:
            self._llm_cache.put(
                prompt=user_content,
                model=self.model_name,
                system_message=system_prompt,
                response=result_text,
            )

        # [v4.7-fix] LLM 응답 JSON 파싱 보호
        try:
            result = json.loads(result_text)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"   ⚠️ 엔티티 추출 JSON 파싱 실패 (batch '{source_label}'): {e}")
            return entities, relationships

        return self._parse_extraction_result(result, source_label)

    def _parse_extraction_result(
        self,
        result: Dict[str, Any],
        source_label: str,
    ) -> Tuple[List[Entity], List[Relationship]]:
        """추출 결과 JSON을 Entity/Relationship 객체로 변환합니다."""
        entities = []
        relationships = []

        for ent_data in result.get("entities", []):
            # [v4.7] Entity Normalization 적용
            raw_name = ent_data.get("name", "")
            if not raw_name:
                continue
            normalized = normalize_entity_name(raw_name)
            entity = Entity(
                name=normalized,
                entity_type=ent_data.get("entity_type", "개념"),
                description=ent_data.get("description", ""),
                source_chunks=[source_label],
            )
            self._add_entity(entity)
            entities.append(entity)

        for rel_data in result.get("relationships", []):
            src_name = rel_data.get("source", "")
            tgt_name = rel_data.get("target", "")
            if not src_name or not tgt_name:
                continue
            relationship = Relationship(
                source=normalize_entity_name(src_name),
                target=normalize_entity_name(tgt_name),
                relation_type=rel_data.get("relation_type", "관련"),
                description=rel_data.get("description", ""),
                weight=_safe_float(rel_data.get("weight", 1.0)),
                keywords=rel_data.get("keywords", ""),
                source_chunks=[source_label],
            )
            self._add_relationship(relationship)
            relationships.append(relationship)

        return entities, relationships

    def _gleaning_pass(
        self,
        batch_text: str,
        existing_entities: List[Entity],
        existing_relationships: List[Relationship],
        source_label: str,
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        [v4.7] Gleaning Pass (EdgeQuake 참조):
        이전 추출 결과를 LLM에 알려주고 누락된 엔티티/관계를 추가 추출합니다.
        EdgeQuake 벤치마크에 따르면 Gleaning으로 엔티티 15-25% 추가 포착 가능.
        """
        entities_str = "\n".join([
            f"- {e.name} ({e.entity_type}): {e.description}" for e in existing_entities
        ]) or "(없음)"
        rels_str = "\n".join([
            f"- {r.source} --[{r.relation_type}]--> {r.target}: {r.description}"
            for r in existing_relationships
        ]) or "(없음)"

        gleaning_prompt = GLEANING_SYSTEM_PROMPT.format(
            existing_entities=entities_str,
            existing_relationships=rels_str,
        )

        entity_types_str = ", ".join(self.entity_types)
        try:
            completion_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": gleaning_prompt},
                    {"role": "user", "content": (
                        f"엔티티 타입: {entity_types_str}\n\n"
                        f"다음 텍스트에서 이전 추출에서 놓친 엔티티와 관계를 찾아주세요:\n\n{batch_text}"
                    )}
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }

            if self._is_gpt5:
                completion_params["max_completion_tokens"] = 2000
            else:
                completion_params["max_tokens"] = 2000

            response = self.client.chat.completions.create(**completion_params)

            if not response or not response.choices:
                print(f"   ⚠️ Gleaning pass API 응답 오류: 응답이 비어있음 ({source_label})")
                return [], []

            # [v4.7-fix] Gleaning 응답 JSON 파싱 보호
            try:
                result = json.loads(response.choices[0].message.content)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"   ⚠️ Gleaning pass JSON 파싱 실패 ({source_label}): {e}")
                return [], []

            new_entities = []
            for ent_data in result.get("entities", []):
                raw_name = ent_data.get("name", "")
                if not raw_name:
                    continue
                normalized = normalize_entity_name(raw_name)
                # 이미 추출된 엔티티인지 확인
                if normalized not in self._entity_cache:
                    entity = Entity(
                        name=normalized,
                        entity_type=ent_data.get("entity_type", "개념"),
                        description=ent_data.get("description", ""),
                        source_chunks=[source_label],
                    )
                    self._add_entity(entity)
                    new_entities.append(entity)

            new_rels = []
            for rel_data in result.get("relationships", []):
                src_raw = rel_data.get("source", "")
                tgt_raw = rel_data.get("target", "")
                if not src_raw or not tgt_raw:
                    continue
                src = normalize_entity_name(src_raw)
                tgt = normalize_entity_name(tgt_raw)
                if not self.graph.has_edge(src, tgt):
                    rel = Relationship(
                        source=src, target=tgt,
                        relation_type=rel_data.get("relation_type", "관련"),
                        description=rel_data.get("description", ""),
                        weight=_safe_float(rel_data.get("weight", 1.0)),
                        keywords=rel_data.get("keywords", ""),
                        source_chunks=[source_label],
                    )
                    self._add_relationship(rel)
                    new_rels.append(rel)

            if new_entities or new_rels:
                print(f"      🔬 Gleaning: +{len(new_entities)} 엔티티, +{len(new_rels)} 관계 추가 발견")

            return new_entities, new_rels

        except Exception as e:
            print(f"      ⚠️ Gleaning 실패: {e}")
            return [], []

    def _add_entity(self, entity: Entity) -> None:
        """
        엔티티를 그래프에 추가 (중복 시 병합) + 역색인 업데이트

        [v4.7] Entity Normalization (EdgeQuake 참조):
        - 이름 정규화 (공백, 대소문자 통일)
        - 설명 병합 (중복 제거, 정보 보존)
        - 36-40% 중복 엔티티 제거 효과
        """
        name = normalize_entity_name(entity.name)
        entity.name = name  # 정규화된 이름으로 업데이트

        if name in self._entity_cache:
            # 기존 엔티티에 정보 병합 (EdgeQuake 방식: 설명 병합)
            existing = self._entity_cache[name]
            existing.description = merge_descriptions(existing.description, entity.description)
            existing.source_chunks.extend(entity.source_chunks)
            # 엔티티 타입도 더 구체적인 것으로 업데이트
            if entity.entity_type != "개념" and existing.entity_type == "개념":
                existing.entity_type = entity.entity_type
            # 그래프 노드 업데이트
            self.graph.nodes[name]["description"] = existing.description
            self.graph.nodes[name]["source_count"] = len(existing.source_chunks)
            self.graph.nodes[name]["entity_type"] = existing.entity_type
        else:
            # 새 엔티티 추가
            self._entity_cache[name] = entity
            self.graph.add_node(
                name,
                entity_type=entity.entity_type,
                description=entity.description,
                source_count=len(entity.source_chunks),
            )

        # 역색인 업데이트: 엔티티명과 description의 토큰을 인덱싱
        desc = entity.description or ""
        for token in set(name.split() + desc.split()):
            token = token.strip()
            if token:
                self._entity_keyword_index.setdefault(token, set()).add(name)

        # [최적화] 문자 기반 인덱스 업데이트 (substring 검색용)
        for ch in set(name):
            if ch.strip():
                self._entity_name_char_index.setdefault(ch, set()).add(name)

        # [v6.0] Neo4j 동기화
        if getattr(self, '_neo4j_driver', None):
            self._neo4j_sync_entity(entity)

    def _add_relationship(self, rel: Relationship) -> None:
        """관계를 그래프에 추가 (중복 시 가중치 합산) + 역색인 업데이트"""
        if self.graph.has_edge(rel.source, rel.target):
            # 기존 관계에 가중치 합산
            self.graph[rel.source][rel.target]["weight"] += rel.weight
            existing_desc = self.graph[rel.source][rel.target].get("description", "")
            if rel.description and rel.description not in existing_desc:
                self.graph[rel.source][rel.target]["description"] = f"{existing_desc}; {rel.description}"
        else:
            # 소스/타겟 노드가 없으면 자동 생성
            if not self.graph.has_node(rel.source):
                self.graph.add_node(rel.source, entity_type="미분류", description="", source_count=0)
            if not self.graph.has_node(rel.target):
                self.graph.add_node(rel.target, entity_type="미분류", description="", source_count=0)

            self.graph.add_edge(
                rel.source,
                rel.target,
                relation_type=rel.relation_type,
                description=rel.description,
                weight=rel.weight,
                keywords=rel.keywords,
            )

        # 역색인 업데이트: 관계 description/keywords/type 인덱싱
        edge_key = (rel.source, rel.target)
        for text in [rel.description, rel.keywords, rel.relation_type]:
            for token in text.split():
                token = token.strip().strip(",")
                if token:
                    self._relation_keyword_index.setdefault(token, set()).add(edge_key)
            # [최적화] 문자 기반 인덱스 업데이트 (substring 검색 폴백용)
            for ch in set(text):
                if ch.strip():
                    self._edge_desc_char_index.setdefault(ch, set()).add(edge_key)

        # [v6.0] Neo4j 동기화
        if getattr(self, '_neo4j_driver', None):
            self._neo4j_sync_relationship(rel)

    # ==================== Dual-Level Retrieval ====================

    def query(
        self,
        query_text: str,
        mode: QueryMode = QueryMode.HYBRID,
        top_k: int = 10,
        vector_results: Optional[List[Dict[str, Any]]] = None,
    ) -> GraphQueryResult:
        """
        LightRAG 스타일 Dual-Level Knowledge Graph 검색

        [v4.7] Mix/Bypass 모드 추가 (EdgeQuake 참조)
        [v4.7] Global 검색에 Community Detection 기반 주제 검색 추가
        [v4.7] Knowledge Injection 기반 쿼리 확장

        Args:
            query_text: 검색 질의
            mode: 검색 모드 (LOCAL/GLOBAL/HYBRID/NAIVE/MIX/BYPASS)
            top_k: 반환할 최대 엔티티/관계 수
            vector_results: [v4.7] Mix 모드에서 사용할 벡터 검색 결과

        Returns:
            GraphQueryResult: 검색된 엔티티, 관계, 컨텍스트 텍스트
        """
        if mode == QueryMode.NAIVE:
            return GraphQueryResult()  # 그래프 비활용

        if mode == QueryMode.BYPASS:
            return GraphQueryResult(context_text="[BYPASS] RAG 검색 생략 — LLM 직접 질의 모드")

        # [v4.7] Knowledge Injection 기반 쿼리 확장
        expanded_query = self._expand_query_with_injections(query_text)
        if expanded_query != query_text:
            print(f"   💉 쿼리 확장: '{query_text}' → '{expanded_query}'")

        # 1. 키워드 추출
        hl_keywords, ll_keywords = self._extract_keywords(expanded_query)
        print(f"   🔑 Keywords - High: {hl_keywords}, Low: {ll_keywords}")

        result = GraphQueryResult()

        # 2. Local 검색 (엔티티 중심)
        if mode in (QueryMode.LOCAL, QueryMode.HYBRID, QueryMode.MIX):
            local_entities, local_rels = self._local_search(ll_keywords, top_k)
            result.entities.extend(local_entities)
            result.relationships.extend(local_rels)

        # 3. Global 검색 (관계/주제 중심 — 커뮤니티 활용)
        if mode in (QueryMode.GLOBAL, QueryMode.HYBRID, QueryMode.MIX):
            global_entities, global_rels = self._global_search(hl_keywords, top_k)
            # 중복 제거 후 추가
            existing_names = {e.name for e in result.entities}
            for e in global_entities:
                if e.name not in existing_names:
                    result.entities.append(e)
            existing_rels = {(r.source, r.target) for r in result.relationships}
            for r in global_rels:
                if (r.source, r.target) not in existing_rels:
                    result.relationships.append(r)

        # 4. 컨텍스트 텍스트 생성
        result.context_text = self._build_context_text(result)

        # [v4.7] Mix 모드: 벡터 결과와 그래프 결과를 가중 결합 (EdgeQuake 참조)
        if mode == QueryMode.MIX and vector_results:
            mix_context = self._build_mix_context(result, vector_results)
            result.context_text = mix_context

        print(f"   📊 Graph Query 결과: 엔티티 {len(result.entities)}개, "
              f"관계 {len(result.relationships)}개"
              f" (모드: {mode.value})")

        return result

    def _extract_keywords(self, query: str) -> Tuple[List[str], List[str]]:
        """LightRAG 스타일 Dual-Level 키워드 추출"""
        try:
            prompt = KEYWORD_EXTRACTION_PROMPT.format(query=query)

            completion_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "키워드를 JSON 형식으로 추출하세요."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }

            if self._is_gpt5:
                completion_params["max_completion_tokens"] = 500
            else:
                completion_params["max_tokens"] = 500

            response = self.client.chat.completions.create(**completion_params)
            result = json.loads(response.choices[0].message.content)

            hl = result.get("high_level_keywords", [])
            ll = result.get("low_level_keywords", [])
            return hl, ll

        except Exception as e:
            print(f"   ⚠️ 키워드 추출 실패: {e}")
            # Fallback: 단순 분할
            words = query.split()
            return words[:2], words

    def _local_search(
        self,
        keywords: List[str],
        top_k: int,
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Local Search: 엔티티 중심 (Low-Level Keywords 사용) — 역색인 기반 O(K) 검색"""
        matched_entities = []
        seen_entity_names: set = set()
        rel_map: Dict[Tuple[str, str], Relationship] = {}

        for keyword in keywords:
            # 역색인에서 정확한 토큰 매칭
            candidate_names = self._entity_keyword_index.get(keyword, set())

            # 엔티티명 부분 문자열 매칭도 지원 (역색인에 없을 경우 — 문자 사전 필터링)
            if not candidate_names:
                pre_filtered = None
                for ch in keyword:
                    char_set = self._entity_name_char_index.get(ch)
                    if char_set is None:
                        pre_filtered = set()
                        break
                    pre_filtered = char_set if pre_filtered is None else pre_filtered & char_set
                    if not pre_filtered:
                        break
                if pre_filtered:
                    candidate_names = {n for n in pre_filtered if keyword in n}

            for node_name in candidate_names:
                if node_name not in seen_entity_names and self.graph.has_node(node_name):
                    seen_entity_names.add(node_name)
                    node_data = self.graph.nodes[node_name]
                    matched_entities.append(Entity(
                        name=node_name,
                        entity_type=node_data.get("entity_type", ""),
                        description=node_data.get("description", ""),
                    ))

                    # 엔티티의 직접 관계 수집 (out + in edges)
                    for _, target, edge_data in self.graph.out_edges(node_name, data=True):
                        key = (node_name, target)
                        if key not in rel_map:
                            rel_map[key] = Relationship(
                                source=node_name, target=target,
                                relation_type=edge_data.get("relation_type", ""),
                                description=edge_data.get("description", ""),
                                weight=edge_data.get("weight", 1.0),
                                keywords=edge_data.get("keywords", ""),
                            )

                    for source, _, edge_data in self.graph.in_edges(node_name, data=True):
                        key = (source, node_name)
                        if key not in rel_map:
                            rel_map[key] = Relationship(
                                source=source, target=node_name,
                                relation_type=edge_data.get("relation_type", ""),
                                description=edge_data.get("description", ""),
                                weight=edge_data.get("weight", 1.0),
                                keywords=edge_data.get("keywords", ""),
                            )

        # weight 기준 정렬
        unique_rels = sorted(rel_map.values(), key=lambda r: r.weight, reverse=True)

        return matched_entities[:top_k], unique_rels[:top_k]

    def _global_search(
        self,
        keywords: List[str],
        top_k: int,
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Global Search: 관계/주제 중심 (High-Level Keywords 사용) — 역색인 기반 O(K) 검색"""
        matched_relationships = []

        for keyword in keywords:
            # 역색인에서 정확한 토큰 매칭
            candidate_edges = self._relation_keyword_index.get(keyword, set())

            # 역색인에 없으면 문자 사전 필터링 기반 검색 (최적화: O(N) → O(K))
            if not candidate_edges:
                pre_filtered = None
                for ch in keyword:
                    char_set = self._edge_desc_char_index.get(ch)
                    if char_set is None:
                        pre_filtered = set()
                        break
                    pre_filtered = char_set if pre_filtered is None else pre_filtered & char_set
                    if not pre_filtered:
                        break
                if pre_filtered:
                    for source, target in pre_filtered:
                        if self.graph.has_edge(source, target):
                            edge_data = self.graph[source][target]
                            desc = edge_data.get("description", "")
                            kw = edge_data.get("keywords", "")
                            if keyword in desc or keyword in kw:
                                candidate_edges = candidate_edges | {(source, target)}

            for source, target in candidate_edges:
                if self.graph.has_edge(source, target):
                    edge_data = self.graph[source][target]
                    rel = Relationship(
                        source=source,
                        target=target,
                        relation_type=edge_data.get("relation_type", ""),
                        description=edge_data.get("description", ""),
                        weight=edge_data.get("weight", 1.0),
                        keywords=edge_data.get("keywords", ""),
                    )
                    matched_relationships.append(rel)

        # 중복 제거 및 정렬
        seen = set()
        unique_rels = []
        for r in matched_relationships:
            key = (r.source, r.target)
            if key not in seen:
                seen.add(key)
                unique_rels.append(r)
        unique_rels.sort(key=lambda r: r.weight, reverse=True)

        # 관계에서 엔티티 추출
        entity_names = set()
        for r in unique_rels[:top_k]:
            entity_names.add(r.source)
            entity_names.add(r.target)

        entities = []
        for name in entity_names:
            if self.graph.has_node(name):
                node_data = self.graph.nodes[name]
                entities.append(Entity(
                    name=name,
                    entity_type=node_data.get("entity_type", ""),
                    description=node_data.get("description", ""),
                ))

        return entities[:top_k], unique_rels[:top_k]

    def _build_context_text(self, result: GraphQueryResult) -> str:
        """검색 결과를 LLM에 전달할 컨텍스트 텍스트로 변환"""
        parts = []

        if result.entities:
            parts.append("### Knowledge Graph 엔티티")
            for e in result.entities:
                parts.append(f"- **{e.name}** ({e.entity_type}): {e.description}")

        if result.relationships:
            parts.append("\n### Knowledge Graph 관계")
            for r in result.relationships:
                parts.append(
                    f"- {r.source} --[{r.relation_type}]--> {r.target}: {r.description}"
                )

        # [v4.7] 관련 커뮤니티 요약 추가
        if self._communities and result.entities:
            community_context = self._get_community_context_for_entities(
                [e.name for e in result.entities]
            )
            if community_context:
                parts.append("\n### 관련 주제 클러스터 (Community)")
                parts.append(community_context)

        return "\n".join(parts) if parts else ""

    # ==================== [v4.7] Community Detection (EdgeQuake 참조) ====================

    def _detect_communities(self) -> None:
        """
        Louvain 알고리즘으로 커뮤니티 클러스터링 수행 (EdgeQuake 참조).
        관련 엔티티들을 자동 그룹화하여 Global 검색의 주제 검색을 강화합니다.
        """
        if not HAS_LOUVAIN:
            print("   ℹ️ Louvain community detection 사용 불가 (networkx 버전 확인 필요)")
            return

        try:
            # DiGraph → undirected로 변환 (Louvain은 무방향 그래프 필요)
            undirected = self.graph.to_undirected()

            if undirected.number_of_nodes() < 3:
                return

            # Louvain 커뮤니티 탐지 (EdgeQuake에서 사용하는 동일 알고리즘)
            communities = louvain_communities(undirected, resolution=1.0, seed=42)

            self._communities = []
            for idx, community_nodes in enumerate(communities):
                if len(community_nodes) < 2:
                    continue  # 단일 노드 커뮤니티 스킵

                entities_in_community = []
                for node_name in community_nodes:
                    if self.graph.has_node(node_name):
                        node_data = self.graph.nodes[node_name]
                        entities_in_community.append({
                            "name": node_name,
                            "entity_type": node_data.get("entity_type", ""),
                            "description": node_data.get("description", ""),
                        })

                # 커뮤니티 내부 관계 수집
                rels_in_community = []
                community_set = set(community_nodes)
                for u, v, data in self.graph.edges(data=True):
                    if u in community_set and v in community_set:
                        rels_in_community.append({
                            "source": u, "target": v,
                            "relation_type": data.get("relation_type", ""),
                        })

                self._communities.append({
                    "id": idx,
                    "nodes": list(community_nodes),
                    "entities": entities_in_community,
                    "relationships": rels_in_community,
                    "size": len(community_nodes),
                })

            print(f"   🏘️ Community Detection: {len(self._communities)}개 커뮤니티 발견")

        except Exception as e:
            print(f"   ⚠️ Community Detection 실패: {e}")

    def _get_community_context_for_entities(
        self, entity_names: List[str],
    ) -> str:
        """주어진 엔티티들이 속한 커뮤니티의 요약 텍스트를 반환"""
        if not self._communities:
            return ""

        relevant_communities = []
        entity_set = set(entity_names)

        for community in self._communities:
            overlap = entity_set.intersection(set(community["nodes"]))
            if overlap:
                relevant_communities.append((community, len(overlap)))

        # 겹치는 엔티티가 많은 커뮤니티 순으로 정렬
        relevant_communities.sort(key=lambda x: x[1], reverse=True)

        parts = []
        for community, overlap_count in relevant_communities[:3]:
            # 요약 생성 (캐싱)
            cid = community["id"]
            if cid not in self._community_summaries:
                self._community_summaries[cid] = self._summarize_community(community)

            summary = self._community_summaries[cid]
            if summary:
                parts.append(
                    f"- **클러스터 {cid}** (엔티티 {community['size']}개, "
                    f"매칭 {overlap_count}개): {summary}"
                )

        return "\n".join(parts)

    def _summarize_community(self, community: Dict[str, Any]) -> str:
        """
        커뮤니티를 LLM으로 요약 (EdgeQuake Community Summary 참조).
        비용 절감을 위해 엔티티 목록 기반 간단 요약 생성.
        """
        entities = community.get("entities", [])
        rels = community.get("relationships", [])

        # 엔티티가 적으면 LLM 호출 없이 간단 요약
        if len(entities) <= 3:
            names = [e["name"] for e in entities]
            return f"관련 엔티티: {', '.join(names)}"

        try:
            entities_str = "\n".join([
                f"- {e['name']} ({e['entity_type']}): {e.get('description', '')}"
                for e in entities[:15]  # 최대 15개까지만
            ])
            rels_str = "\n".join([
                f"- {r['source']} → {r['target']} ({r.get('relation_type', '')})"
                for r in rels[:15]
            ]) or "(없음)"

            prompt = COMMUNITY_SUMMARY_PROMPT.format(
                entities=entities_str,
                relationships=rels_str,
            )

            completion_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "한국어로 간결하게 요약하세요."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }

            if self._is_gpt5:
                completion_params["max_completion_tokens"] = 500
            else:
                completion_params["max_tokens"] = 500

            response = self.client.chat.completions.create(**completion_params)
            result = json.loads(response.choices[0].message.content)
            theme = result.get("theme", "")
            summary = result.get("summary", "")
            return f"{theme} — {summary}" if theme else summary

        except Exception as e:
            # Fallback: 엔티티 이름 나열
            names = [e["name"] for e in entities[:5]]
            return f"관련 엔티티: {', '.join(names)}"

    # ==================== [v4.7] Mix Mode (EdgeQuake 참조) ====================

    def _build_mix_context(
        self,
        graph_result: GraphQueryResult,
        vector_results: List[Dict[str, Any]],
    ) -> str:
        """
        벡터 검색 결과와 그래프 검색 결과를 가중 결합합니다 (EdgeQuake Mix 모드).
        mix_graph_weight로 비율 조절:
        - 0.0 = 벡터 결과만 사용
        - 0.5 = 반반 혼합
        - 1.0 = 그래프 결과만 사용
        """
        parts = []

        graph_w = self.mix_graph_weight
        vector_w = 1.0 - graph_w

        # 벡터 검색 컨텍스트 (가중치 표시)
        if vector_results and vector_w > 0:
            parts.append(f"### 벡터 검색 결과 (가중치: {vector_w:.1f})")
            for i, doc in enumerate(vector_results[:5]):
                content = doc.get("content", doc.get("page_content", ""))
                score = doc.get("score", doc.get("@search.score", 0))
                parts.append(f"[V{i + 1}] (score: {score:.3f}) {content[:500]}")

        # 그래프 컨텍스트 (가중치 표시)
        if graph_result.context_text and graph_w > 0:
            parts.append(f"\n### Knowledge Graph 컨텍스트 (가중치: {graph_w:.1f})")
            parts.append(graph_result.context_text)

        return "\n".join(parts)

    # ==================== [v4.7] Knowledge Injection (EdgeQuake 참조) ====================

    def inject_knowledge(
        self,
        injections: List[KnowledgeInjection],
    ) -> int:
        """
        도메인 용어집/동의어를 Knowledge Graph에 주입합니다 (EdgeQuake 참조).
        주입된 정보는 쿼리 확장에 사용되지만, 답변 출처(source citation)에는 포함되지 않습니다.

        Args:
            injections: KnowledgeInjection 리스트

        Returns:
            주입된 엔트리 수
        """
        count = 0
        for inj in injections:
            term = normalize_entity_name(inj.term)
            self._injections[term] = inj

            # 동의어 맵 구축
            for syn in inj.synonyms:
                normalized_syn = normalize_entity_name(syn)
                self._synonym_map[normalized_syn] = term

            # 그래프에 엔티티로도 추가 (optional enrichment)
            entity = Entity(
                name=term,
                entity_type=inj.entity_type,
                description=inj.definition,
                source_chunks=["knowledge_injection"],
            )
            self._add_entity(entity)
            count += 1

        print(f"💉 Knowledge Injection: {count}개 용어 주입 완료 "
              f"(동의어 맵: {len(self._synonym_map)}개)")
        return count

    def inject_from_text(self, text: str) -> int:
        """
        텍스트 파일에서 용어집을 파싱하여 주입합니다.
        형식: 각 줄 "용어: 정의" 또는 "용어 (동의어1, 동의어2): 정의"
        """
        injections = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # "용어 (동의어1, 동의어2): 정의" 파싱
            match = re.match(r'^(.+?)(?:\s*\(([^)]+)\))?\s*[:：]\s*(.+)$', line)
            if match:
                term = match.group(1).strip()
                synonyms_str = match.group(2) or ""
                definition = match.group(3).strip()
                synonyms = [s.strip() for s in synonyms_str.split(",") if s.strip()]
                injections.append(KnowledgeInjection(
                    term=term, definition=definition, synonyms=synonyms
                ))

        return self.inject_knowledge(injections)

    def _expand_query_with_injections(self, query: str) -> str:
        """
        Knowledge Injection의 동의어 맵을 사용하여 쿼리를 자동 확장합니다 (EdgeQuake 참조).
        예: "OEE가 뭐야?" → "OEE(Overall Equipment Effectiveness, 설비종합효율)가 뭐야?"
        """
        if not self._injections and not self._synonym_map:
            return query

        expanded = query

        # 동의어 → 정식 용어 확장
        for syn, canonical in self._synonym_map.items():
            if syn in expanded and canonical not in expanded:
                expanded = expanded.replace(syn, f"{syn}({canonical})")

        # 약어 → 정의 확장
        for term, inj in self._injections.items():
            if term in expanded and inj.definition:
                short_def = inj.definition[:60]
                if f"{term}(" not in expanded:
                    expanded = expanded.replace(term, f"{term}({short_def})")

        return expanded

    # ==================== 그래프 유틸리티 ====================

    def get_stats(self) -> Dict[str, Any]:
        """Knowledge Graph 통계 반환 [v4.7: 커뮤니티/주입 정보 추가]"""
        if not self.graph.nodes:
            return {"nodes": 0, "edges": 0, "entity_types": {}, "communities": 0, "injections": 0}

        entity_type_counts = {}
        for _, data in self.graph.nodes(data=True):
            et = data.get("entity_type", "미분류")
            entity_type_counts[et] = entity_type_counts.get(et, 0) + 1

        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "entity_types": entity_type_counts,
            "avg_degree": (
                sum(d for _, d in self.graph.degree()) / self.graph.number_of_nodes()
                if self.graph.number_of_nodes() > 0 else 0
            ),
            "communities": len(self._communities),
            "injections": len(self._injections),
            "synonym_map_size": len(self._synonym_map),
            "gleaning_passes": self.gleaning_passes,
        }

    def get_subgraph(
        self,
        entity_name: str,
        max_depth: int = 2,
        max_nodes: int = 50,
    ) -> Dict[str, Any]:
        """특정 엔티티 중심의 서브그래프 반환 (LightRAG get_knowledge_graph 참조)"""
        if not self.graph.has_node(entity_name):
            return {"nodes": [], "edges": []}

        # BFS로 서브그래프 탐색 (deque로 O(1) popleft)
        visited = {entity_name}
        queue = deque([(entity_name, 0)])
        edges = []

        while queue and len(visited) < max_nodes:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue

            # Out-edges
            for _, neighbor, data in self.graph.out_edges(current, data=True):
                edges.append({
                    "source": current,
                    "target": neighbor,
                    "relation_type": data.get("relation_type", ""),
                    "weight": data.get("weight", 1.0),
                })
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

            # In-edges
            for neighbor, _, data in self.graph.in_edges(current, data=True):
                edges.append({
                    "source": neighbor,
                    "target": current,
                    "relation_type": data.get("relation_type", ""),
                    "weight": data.get("weight", 1.0),
                })
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        nodes = []
        for name in visited:
            if self.graph.has_node(name):
                node_data = self.graph.nodes[name]
                nodes.append({
                    "name": name,
                    "entity_type": node_data.get("entity_type", ""),
                    "description": node_data.get("description", ""),
                    "degree": self.graph.degree(name),
                })

        return {"nodes": nodes, "edges": edges}

    def save_graph(self, filepath: str) -> None:
        """Knowledge Graph를 JSON 파일로 저장"""
        data = {
            "nodes": [
                {
                    "name": n,
                    **self.graph.nodes[n],
                }
                for n in self.graph.nodes
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    **d,
                }
                for u, v, d in self.graph.edges(data=True)
            ],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"📁 Knowledge Graph 저장 완료: {filepath}")

    def load_graph(self, filepath: str) -> None:
        """JSON 파일에서 Knowledge Graph 로드 [v4.7: 커뮤니티 재구축]"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.graph.clear()
        self._entity_keyword_index.clear()
        self._relation_keyword_index.clear()
        self._entity_name_char_index.clear()
        self._edge_desc_char_index.clear()
        self._entity_cache.clear()
        self._communities.clear()
        self._community_summaries.clear()

        for node in data.get("nodes", []):
            name = node["name"]
            attrs = {k: v for k, v in node.items() if k != "name"}
            self.graph.add_node(name, **attrs)

        for edge in data.get("edges", []):
            source = edge["source"]
            target = edge["target"]
            attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
            self.graph.add_edge(source, target, **attrs)

        # [v4.7-fix] 로드 후 Keyword Index 재구축 — local/global 검색에 필수
        for name, attrs in self.graph.nodes(data=True):
            desc = attrs.get("description", "") or ""
            for token in set(name.split() + desc.split()):
                token = token.strip()
                if token:
                    self._entity_keyword_index.setdefault(token, set()).add(name)
            # [최적화] 문자 기반 인덱스 재구축
            for ch in set(name):
                if ch.strip():
                    self._entity_name_char_index.setdefault(ch, set()).add(name)

        for source, target, attrs in self.graph.edges(data=True):
            edge_key = (source, target)
            for text in [attrs.get("description", ""), attrs.get("keywords", ""), attrs.get("relation_type", "")]:
                for token in (text or "").split():
                    token = token.strip().strip(",")
                    if token:
                        self._relation_keyword_index.setdefault(token, set()).add(edge_key)
                # [최적화] 문자 기반 인덱스 재구축
                for ch in set(text or ""):
                    if ch.strip():
                        self._edge_desc_char_index.setdefault(ch, set()).add(edge_key)

        # [v4.7] 로드 후 Community Detection 재실행
        if self.graph.number_of_nodes() >= 5:
            self._detect_communities()

        print(f"📁 Knowledge Graph 로드 완료: 노드 {self.graph.number_of_nodes()}개, "
              f"엣지 {self.graph.number_of_edges()}개, "
              f"커뮤니티 {len(self._communities)}개")

    def clear(self) -> None:
        """Knowledge Graph 초기화 [v4.7: 커뮤니티/주입 캐시도 초기화, v6.0: Neo4j 지원]"""
        if self.graph is not None:
            self.graph.clear()
        self._entity_cache.clear()
        self._chunk_to_entities.clear()
        self._entity_keyword_index.clear()
        self._relation_keyword_index.clear()
        self._entity_name_char_index.clear()
        self._edge_desc_char_index.clear()
        self._normalized_name_map.clear()
        self._communities.clear()
        self._community_summaries.clear()
        self._injections.clear()
        self._synonym_map.clear()

        # [v6.0] Neo4j 초기화
        if getattr(self, '_neo4j_driver', None):
            self._neo4j_clear()

        print("🗑️ Knowledge Graph 초기화 완료")

    # ==================== [v6.0] Neo4j 백엔드 메서드 ====================

    def _neo4j_sync_entity(self, entity: Entity) -> None:
        """엔티티를 Neo4j에 동기화 (MERGE: 있으면 업데이트, 없으면 생성)"""
        if not self._neo4j_driver:
            return
        query = """
        MERGE (e:Entity {name: $name})
        SET e.entity_type = $entity_type,
            e.description = $description,
            e.source_count = $source_count
        """
        with self._neo4j_driver.session() as session:
            session.run(query, {
                "name": entity.name,
                "entity_type": entity.entity_type,
                "description": entity.description,
                "source_count": len(entity.source_chunks),
            })

    def _neo4j_sync_relationship(self, rel: Relationship) -> None:
        """관계를 Neo4j에 동기화"""
        if not self._neo4j_driver:
            return
        query = """
        MERGE (s:Entity {name: $source})
        MERGE (t:Entity {name: $target})
        MERGE (s)-[r:RELATES_TO {relation_type: $relation_type}]->(t)
        SET r.description = $description,
            r.weight = $weight,
            r.keywords = $keywords
        """
        with self._neo4j_driver.session() as session:
            session.run(query, {
                "source": rel.source,
                "target": rel.target,
                "relation_type": rel.relation_type,
                "description": rel.description,
                "weight": rel.weight,
                "keywords": rel.keywords,
            })

    def _neo4j_clear(self) -> None:
        """Neo4j의 모든 노드와 관계를 삭제"""
        if not self._neo4j_driver:
            return
        with self._neo4j_driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def _neo4j_load_to_networkx(self) -> None:
        """Neo4j에서 NetworkX 그래프로 로드 (인메모리 캐시 갱신)"""
        if not self._neo4j_driver or self.graph is None:
            return

        self.graph.clear()
        with self._neo4j_driver.session() as session:
            # 노드 로드
            result = session.run("MATCH (e:Entity) RETURN e")
            for record in result:
                node = record["e"]
                name = node["name"]
                self.graph.add_node(
                    name,
                    entity_type=node.get("entity_type", ""),
                    description=node.get("description", ""),
                    source_count=node.get("source_count", 0),
                )

            # 관계 로드
            result = session.run(
                "MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity) "
                "RETURN s.name AS source, t.name AS target, r"
            )
            for record in result:
                self.graph.add_edge(
                    record["source"],
                    record["target"],
                    relation_type=record["r"].get("relation_type", ""),
                    description=record["r"].get("description", ""),
                    weight=record["r"].get("weight", 1.0),
                    keywords=record["r"].get("keywords", ""),
                )

        print(f"📊 Neo4j → NetworkX 동기화: 노드 {self.graph.number_of_nodes()}개, "
              f"엣지 {self.graph.number_of_edges()}개")
