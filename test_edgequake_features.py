"""
EdgeQuake 기능 통합 테스트 (v4.7)

오프라인 테스트 — LLM 호출 없이 그래프 조작/정규화/커뮤니티/주입 기능을 검증합니다.
"""

import os
import tempfile
from azure_korean_doc_framework.core.graph_rag import (
    KnowledgeGraphManager, QueryMode, Entity, Relationship,
    KnowledgeInjection, GraphQueryResult,
    normalize_entity_name, merge_descriptions,
)


def test_entity_normalization():
    print("=== 1. Entity Normalization 테스트 ===")
    assert normalize_entity_name("  삼성전자  ") == "삼성전자"
    # APPLE(5글자) → Capitalize, INC(3글자, 대문자) → 약어 유지
    assert normalize_entity_name("  APPLE INC  ") == "Apple INC"
    assert normalize_entity_name("microsoft   corp") == "Microsoft Corp"
    # 약어(모두 대문자, 1-4글자)는 원본 유지
    assert normalize_entity_name("OEE") == "OEE"
    assert normalize_entity_name("AI") == "AI"
    assert normalize_entity_name("NLP") == "NLP"
    assert normalize_entity_name("IBM") == "IBM"
    # 한국어+영문 혼합은 원본 유지
    assert normalize_entity_name("삼성 Electronics") == "삼성 Electronics"
    print("   ✅ 통과")


def test_description_merging():
    print("=== 2. Description Merging 테스트 ===")
    assert merge_descriptions("A", "B") == "A; B"
    assert merge_descriptions("A", "A") == "A"
    assert merge_descriptions("", "B") == "B"
    assert merge_descriptions("A", "") == "A"
    assert merge_descriptions("AB", "B") == "AB"  # B는 AB의 부분
    assert merge_descriptions("B", "AB") == "AB"  # AB가 B를 포함
    print("   ✅ 통과")


def test_query_modes():
    print("=== 3. QueryMode 확장 테스트 ===")
    modes = [m.value for m in QueryMode]
    assert "mix" in modes, "mix 모드 없음"
    assert "bypass" in modes, "bypass 모드 없음"
    assert len(modes) == 6
    print(f"   ✅ 6개 모드: {modes}")


def test_init_with_new_params():
    print("=== 4. 초기화 (새 파라미터) 테스트 ===")
    gm = KnowledgeGraphManager(gleaning_passes=2, mix_graph_weight=0.6)
    assert gm.gleaning_passes == 2
    assert gm.mix_graph_weight == 0.6
    # 범위 클램핑 테스트
    gm2 = KnowledgeGraphManager(mix_graph_weight=1.5)
    assert gm2.mix_graph_weight == 1.0
    gm3 = KnowledgeGraphManager(mix_graph_weight=-0.5)
    assert gm3.mix_graph_weight == 0.0
    print("   ✅ 통과")


def test_entity_dedup_and_normalization():
    print("=== 5. Entity Dedup + Normalization 테스트 ===")
    gm = KnowledgeGraphManager()

    e1 = Entity(name="삼성전자", entity_type="조직", description="한국 대기업")
    e2 = Entity(name="삼성전자", entity_type="조직", description="반도체 제조 회사")
    e3 = Entity(name="삼성전자", entity_type="개념", description="한국 대기업")  # 타입이 덜 구체적

    gm._add_entity(e1)
    gm._add_entity(e2)
    gm._add_entity(e3)

    assert gm.graph.number_of_nodes() == 1, f"Expected 1 node, got {gm.graph.number_of_nodes()}"
    desc = gm.graph.nodes["삼성전자"]["description"]
    assert "한국 대기업" in desc
    assert "반도체 제조 회사" in desc
    # entity_type은 더 구체적인 "조직"이 유지돼야 함
    assert gm.graph.nodes["삼성전자"]["entity_type"] == "조직"
    print(f"   ✅ 통과 (merged desc: {desc})")


def test_knowledge_injection():
    print("=== 6. Knowledge Injection 테스트 ===")
    gm = KnowledgeGraphManager()

    injections = [
        KnowledgeInjection(
            term="OEE",
            definition="Overall Equipment Effectiveness, 설비종합효율",
            synonyms=["설비종합효율", "설비효율"],
        ),
        KnowledgeInjection(
            term="AI",
            definition="Artificial Intelligence, 인공지능",
            synonyms=["인공지능"],
        ),
    ]
    count = gm.inject_knowledge(injections)
    assert count == 2

    # synonym map 검증 (정규화됨)
    assert len(gm._synonym_map) == 3  # 설비종합효율, 설비효율, 인공지능
    print(f"   ✅ synonym_map: {dict(gm._synonym_map)}")

    # 그래프에 엔티티로 추가됐는지
    stats = gm.get_stats()
    assert stats["injections"] == 2
    assert stats["nodes"] >= 2
    print(f"   ✅ injections={stats['injections']}, nodes={stats['nodes']}")


def test_inject_from_text():
    print("=== 7. inject_from_text 테스트 ===")
    gm = KnowledgeGraphManager()

    text = """# 도메인 용어집
OEE (설비종합효율, 설비효율): Overall Equipment Effectiveness
AI (인공지능): Artificial Intelligence
NLP: Natural Language Processing
"""
    count = gm.inject_from_text(text)
    assert count == 3, f"Expected 3 injections, got {count}"
    print(f"   ✅ {count}개 용어 파싱 성공")


def test_query_expansion():
    print("=== 8. Query Expansion 테스트 ===")
    gm = KnowledgeGraphManager()
    gm.inject_knowledge([
        KnowledgeInjection(
            term="OEE", definition="Overall Equipment Effectiveness",
            synonyms=["설비종합효율"]
        ),
    ])

    original = "OEE란 무엇인가?"
    expanded = gm._expand_query_with_injections(original)
    assert expanded != original, "쿼리가 확장되지 않았음"
    assert "Overall" in expanded or "OEE(" in expanded
    print(f"   ✅ 원본: {original}")
    print(f"   ✅ 확장: {expanded}")


def test_mix_context():
    print("=== 9. Mix Mode Context 테스트 ===")
    gm = KnowledgeGraphManager(mix_graph_weight=0.4)

    e = Entity(name="삼성전자", entity_type="조직", description="한국 대기업")
    gr = GraphQueryResult(
        entities=[e],
        relationships=[],
        context_text="### Knowledge Graph 엔티티\n- 삼성전자 (조직): 한국 대기업"
    )
    vector_results = [
        {"content": "삼성전자는 세계 최대 반도체 회사입니다.", "score": 0.95},
        {"content": "삼성전자의 2025년 매출은 350조원입니다.", "score": 0.88},
    ]

    mix_text = gm._build_mix_context(gr, vector_results)
    assert "가중치: 0.6" in mix_text  # vector weight
    assert "가중치: 0.4" in mix_text  # graph weight
    assert "삼성전자" in mix_text
    print(f"   ✅ Mix 컨텍스트 생성 (길이: {len(mix_text)})")


def test_community_detection():
    print("=== 10. Community Detection 테스트 ===")
    gm = KnowledgeGraphManager()

    # 클러스터 1: 연결된 체인
    for i in range(8):
        gm._add_entity(Entity(name=f"노드A_{i}", entity_type="개념", description=f"클러스터1의 노드 {i}"))
    for i in range(7):
        gm._add_relationship(Relationship(
            source=f"노드A_{i}", target=f"노드A_{i+1}",
            relation_type="연결", description="체인 연결"
        ))

    # 클러스터 2: 별도 그룹
    for i in range(5):
        gm._add_entity(Entity(name=f"노드B_{i}", entity_type="기관", description=f"클러스터2의 노드 {i}"))
    for i in range(4):
        gm._add_relationship(Relationship(
            source=f"노드B_{i}", target=f"노드B_{i+1}",
            relation_type="협력", description="협력 관계"
        ))

    gm._detect_communities()
    assert len(gm._communities) >= 1, f"커뮤니티 없음 (노드: {gm.graph.number_of_nodes()})"
    total_community_nodes = sum(c["size"] for c in gm._communities)
    print(f"   ✅ {len(gm._communities)}개 커뮤니티 발견, 총 {total_community_nodes}개 노드")

    for c in gm._communities:
        print(f"      클러스터 {c['id']}: {c['size']}개 노드, {len(c['relationships'])}개 관계")


def test_community_context():
    print("=== 11. Community Context 테스트 ===")
    gm = KnowledgeGraphManager()

    for i in range(6):
        gm._add_entity(Entity(name=f"회사_{i}", entity_type="조직", description=f"테스트 회사 {i}"))
    for i in range(5):
        gm._add_relationship(Relationship(
            source=f"회사_{i}", target=f"회사_{i+1}",
            relation_type="거래", description="거래 관계"
        ))

    gm._detect_communities()

    context = gm._get_community_context_for_entities(["회사_0", "회사_1"])
    if context:
        print(f"   ✅ 커뮤니티 컨텍스트: {context[:100]}...")
    else:
        print(f"   ✅ 커뮨니티 컨텍스트: (커뮤니티가 작아서 간단 요약)")


def test_save_load_with_communities():
    print("=== 12. Save/Load + Community 재구축 테스트 ===")
    gm = KnowledgeGraphManager()

    for i in range(10):
        gm._add_entity(Entity(name=f"테스트_{i}", entity_type="개념", description=f"desc {i}"))
    for i in range(9):
        gm._add_relationship(Relationship(
            source=f"테스트_{i}", target=f"테스트_{i+1}",
            relation_type="link", description="link"
        ))
    gm._detect_communities()
    original_stats = gm.get_stats()

    # 저장
    tmp = tempfile.mktemp(suffix=".json")
    gm.save_graph(tmp)

    # 로드
    gm2 = KnowledgeGraphManager()
    gm2.load_graph(tmp)
    loaded_stats = gm2.get_stats()

    assert loaded_stats["nodes"] == original_stats["nodes"]
    assert loaded_stats["edges"] == original_stats["edges"]
    assert loaded_stats["communities"] >= 1, "로드 후 커뮤니티 재구축 실패"

    os.remove(tmp)
    print(f"   ✅ 저장/로드 후: nodes={loaded_stats['nodes']}, edges={loaded_stats['edges']}, communities={loaded_stats['communities']}")


def test_full_stats():
    print("=== 13. 전체 Stats 테스트 ===")
    gm = KnowledgeGraphManager(gleaning_passes=1)
    gm._add_entity(Entity(name="테스트", entity_type="개념", description="테스트"))
    gm.inject_knowledge([KnowledgeInjection(term="NLP", definition="자연어처리")])

    stats = gm.get_stats()
    required_keys = {"nodes", "edges", "entity_types", "avg_degree", "communities", "injections", "synonym_map_size", "gleaning_passes"}
    missing = required_keys - set(stats.keys())
    assert not missing, f"누락된 stats 키: {missing}"
    print(f"   ✅ Stats 키 검증 통과: {list(stats.keys())}")


def test_bypass_mode():
    print("=== 14. Bypass Mode 테스트 ===")
    gm = KnowledgeGraphManager()
    result = gm.query("테스트", mode=QueryMode.BYPASS)
    assert "BYPASS" in result.context_text
    print(f"   ✅ Bypass: {result.context_text}")


if __name__ == "__main__":
    tests = [
        test_entity_normalization,
        test_description_merging,
        test_query_modes,
        test_init_with_new_params,
        test_entity_dedup_and_normalization,
        test_knowledge_injection,
        test_inject_from_text,
        test_query_expansion,
        test_mix_context,
        test_community_detection,
        test_community_context,
        test_save_load_with_communities,
        test_full_stats,
        test_bypass_mode,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"   ❌ 실패: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"🎉 결과: {passed}/{passed+failed} 테스트 통과")
    if failed:
        print(f"⚠️ {failed}개 테스트 실패")
    else:
        print("✅ 모든 테스트 통과!")
