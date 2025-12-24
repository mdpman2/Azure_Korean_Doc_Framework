from azure_korean_doc_framework.core.vector_store import VectorStore
from azure_korean_doc_framework.config import Config

def main():
    try:
        Config.validate()
        vector_store = VectorStore()

        print(f"🔍 Index: {vector_store.index_name}")

        # 1. Total Count
        count_results = vector_store.search_client.get_document_count()
        print(f"📊 Total Documents in Index: {count_results}")

        # 2. Check for the specific filename using exact filter
        targets = ["눈건강+관리를+위한+9대+생활수칙.pdf", "WP22-05.pdf", "3. 향후 통화신용정책 방향.pdf", "한-호주 퇴직연금 포럼_책자(최종).pdf", "★2019 제1회 증시콘서트 자료집_최종★.pdf"]

        print("\n--- Document Presence Check ---")
        for target_filename in targets:
            try:
                results_exact = vector_store.search_client.search(
                    search_text="*",
                    filter=f"parent_id eq '{target_filename}'",
                    select=["parent_id", "chunk_id"],
                    top=1
                )

                found = False
                for r in results_exact:
                     found = True
                     print(f"✅ [FOUND] '{target_filename}' (Chunk ID: {r['chunk_id']})")

                if not found:
                    print(f"❌ [MISSING] '{target_filename}'")

            except Exception as e:
                print(f"⚠️ Error checking '{target_filename}': {e}")

        # 3. Inspect a sample of documents to see IDs
        print(f"\n🔍 Inspecting ID format (Top 5 docs):")
        results_sample = vector_store.search_client.search(
            search_text="*",
            select=["chunk_id", "parent_id"],
            top=5
        )

        for r in results_sample:
            print(f"   🆔 ID: {r['chunk_id']} | Parent: {r['parent_id']}")

        print("\n" + "="*50)
        print("💡 TIP: 만약 하나의 파일만 보이고 나머지가 'MISSING'이라면,")
        print("   청크 ID 충돌(chunk_0, chunk_1...)로 인해 파일이 서로 덮어씌워졌을 가능성이 높습니다.")
        print("   이미 소스 코드를 수정하였으니, 'doc_chunk_main.py'를 다시 실행하여")
        print("   문서들을 다시 인덱싱(Ingestion) 해주세요.")
        print("="*50)

    except Exception as e:
        print(f"⚠️ Error verifying index: {e}")

if __name__ == "__main__":
    main()
