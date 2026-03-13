"""
LightRAG-inspired Graph RAG 모듈 (v4.0)

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
"""

import json
import hashlib
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from ..config import Config
from ..utils.azure_clients import AzureClientFactory


# ==================== 데이터 모델 ====================

class QueryMode(Enum):
    """LightRAG 스타일 검색 모드"""
    LOCAL = "local"       # 엔티티 중심 검색 (Low-Level Keywords)
    GLOBAL = "global"     # 관계/주제 중심 검색 (High-Level Keywords)
    HYBRID = "hybrid"     # Local + Global 결합
    NAIVE = "naive"       # 벡터 검색만 사용 (그래프 비활용)


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


class KnowledgeGraphManager:
    """
    LightRAG-inspired Knowledge Graph 관리자

    LightRAG의 핵심 아키텍처를 참조하여 구현:
    - 엔티티/관계 추출 (GPT-5.4)
    - NetworkX 기반 인메모리 그래프
    - Dual-Level Retrieval (Local + Global)
    - 한국어 문서 특화

    참조: https://github.com/HKUDS/LightRAG
    """

    def __init__(
        self,
        entity_types: Optional[List[str]] = None,
        model_key: str = "gpt-5.4",
        max_entities_per_chunk: int = 20,
    ):
        if not HAS_NETWORKX:
            raise ImportError(
                "networkx 패키지가 필요합니다. 설치: pip install networkx"
            )

        self.graph = nx.DiGraph()
        self.entity_types = entity_types or KOREAN_ENTITY_TYPES
        self.model_key = model_key
        self.max_entities_per_chunk = max_entities_per_chunk
        self._is_gpt5 = "gpt-5" in model_key.lower()

        # LLM 클라이언트
        self.client = AzureClientFactory.get_openai_client(is_advanced=True)
        self.model_name = Config.MODELS.get(model_key, "model-router")

        # 엔티티/관계 캐시 (중복 방지)
        self._entity_cache: Dict[str, Entity] = {}
        self._chunk_to_entities: Dict[str, List[str]] = {}

        # 역색인: 키워드 → 엔티티/관계 빠른 검색용 (O(N*K) → O(K))
        self._entity_keyword_index: Dict[str, set] = {}  # keyword → {entity_name, ...}
        self._relation_keyword_index: Dict[str, set] = {}  # keyword → {(source, target), ...}

        print(f"📊 KnowledgeGraphManager 초기화 (모델: {model_key}, 엔티티 타입: {len(self.entity_types)}개)")

    # ==================== 엔티티/관계 추출 ====================

    def extract_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 5,
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        문서 청크에서 엔티티와 관계를 추출하여 Knowledge Graph를 구축합니다.

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
        print(f"🔍 엔티티/관계 추출 시작 (총 {total}개 청크, 배치 크기: {batch_size})")

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
                completion_params = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"다음 텍스트에서 엔티티와 관계를 추출하세요:\n\n{batch_text}"}
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

                # JSON 파싱
                result = json.loads(result_text)

                # 엔티티 처리
                for ent_data in result.get("entities", []):
                    entity = Entity(
                        name=ent_data["name"],
                        entity_type=ent_data.get("entity_type", "개념"),
                        description=ent_data.get("description", ""),
                        source_chunks=[f"batch_{i // batch_size}"],
                    )
                    self._add_entity(entity)
                    all_entities.append(entity)

                # 관계 처리
                for rel_data in result.get("relationships", []):
                    relationship = Relationship(
                        source=rel_data["source"],
                        target=rel_data["target"],
                        relation_type=rel_data.get("relation_type", "관련"),
                        description=rel_data.get("description", ""),
                        weight=float(rel_data.get("weight", 1.0)),
                        keywords=rel_data.get("keywords", ""),
                        source_chunks=[f"batch_{i // batch_size}"],
                    )
                    self._add_relationship(relationship)
                    all_relationships.append(relationship)

                print(f"   ✅ 배치 {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}: "
                      f"엔티티 {len(result.get('entities', []))}개, "
                      f"관계 {len(result.get('relationships', []))}개 추출")

            except Exception as e:
                print(f"   ⚠️ 배치 {i // batch_size + 1} 추출 실패: {e}")
                continue

        print(f"📊 Knowledge Graph 구축 완료: "
              f"노드 {self.graph.number_of_nodes()}개, "
              f"엣지 {self.graph.number_of_edges()}개")

        return all_entities, all_relationships

    def _add_entity(self, entity: Entity) -> None:
        """엔티티를 그래프에 추가 (중복 시 병합) + 역색인 업데이트"""
        name = entity.name

        if name in self._entity_cache:
            # 기존 엔티티에 정보 병합
            existing = self._entity_cache[name]
            if entity.description and len(entity.description) > len(existing.description):
                existing.description = entity.description
            existing.source_chunks.extend(entity.source_chunks)
            # 그래프 노드 업데이트
            self.graph.nodes[name]["description"] = existing.description
            self.graph.nodes[name]["source_count"] = len(existing.source_chunks)
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

    # ==================== Dual-Level Retrieval ====================

    def query(
        self,
        query_text: str,
        mode: QueryMode = QueryMode.HYBRID,
        top_k: int = 10,
    ) -> GraphQueryResult:
        """
        LightRAG 스타일 Dual-Level Knowledge Graph 검색

        Args:
            query_text: 검색 질의
            mode: 검색 모드 (LOCAL/GLOBAL/HYBRID/NAIVE)
            top_k: 반환할 최대 엔티티/관계 수

        Returns:
            GraphQueryResult: 검색된 엔티티, 관계, 컨텍스트 텍스트
        """
        if mode == QueryMode.NAIVE:
            return GraphQueryResult()  # 그래프 비활용

        # 1. 키워드 추출
        hl_keywords, ll_keywords = self._extract_keywords(query_text)
        print(f"   🔑 Keywords - High: {hl_keywords}, Low: {ll_keywords}")

        result = GraphQueryResult()

        # 2. Local 검색 (엔티티 중심)
        if mode in (QueryMode.LOCAL, QueryMode.HYBRID):
            local_entities, local_rels = self._local_search(ll_keywords, top_k)
            result.entities.extend(local_entities)
            result.relationships.extend(local_rels)

        # 3. Global 검색 (관계/주제 중심)
        if mode in (QueryMode.GLOBAL, QueryMode.HYBRID):
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

        print(f"   📊 Graph Query 결과: 엔티티 {len(result.entities)}개, "
              f"관계 {len(result.relationships)}개")

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

            # 엔티티명 부분 문자열 매칭도 지원 (역색인에 없을 경우)
            if not candidate_names:
                for node_name in self.graph.nodes:
                    if keyword in node_name:
                        candidate_names = candidate_names | {node_name}

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

            # 역색인에 없으면 전체 검색 폴백
            if not candidate_edges:
                for source, target, edge_data in self.graph.edges(data=True):
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

        return "\n".join(parts) if parts else ""

    # ==================== 그래프 유틸리티 ====================

    def get_stats(self) -> Dict[str, Any]:
        """Knowledge Graph 통계 반환"""
        if not self.graph.nodes:
            return {"nodes": 0, "edges": 0, "entity_types": {}}

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
        """JSON 파일에서 Knowledge Graph 로드"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.graph.clear()
        self._entity_keyword_index.clear()
        self._relation_keyword_index.clear()
        self._entity_cache.clear()

        for node in data.get("nodes", []):
            name = node["name"]
            attrs = {k: v for k, v in node.items() if k != "name"}
            self.graph.add_node(name, **attrs)

        for edge in data.get("edges", []):
            source = edge["source"]
            target = edge["target"]
            attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
            self.graph.add_edge(source, target, **attrs)

        print(f"📁 Knowledge Graph 로드 완료: 노드 {self.graph.number_of_nodes()}개, "
              f"엣지 {self.graph.number_of_edges()}개")

    def clear(self) -> None:
        """Knowledge Graph 초기화"""
        self.graph.clear()
        self._entity_cache.clear()
        self._chunk_to_entities.clear()
        self._entity_keyword_index.clear()
        self._relation_keyword_index.clear()
        print("🗑️ Knowledge Graph 초기화 완료")
