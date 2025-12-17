import os
import chromadb
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

# 1. GÜVENLİK: .env dosyasını oku
load_dotenv()

api_key = os.getenv("CHROMA_API_KEY")
tenant = os.getenv("CHROMA_TENANT")
database = os.getenv("CHROMA_DATABASE")

# 2. BAĞLANTIYI KUR
print("Chroma Cloud'a .env anahtarlarıyla bağlanılıyor...")

try:
    # Embedding fonksiyonunu tanımla (Sorgu için gerekli)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.CloudClient(
        api_key=api_key,
        tenant=tenant,
        database=database
    )

    # DİKKAT: Burada 'create' değil 'get_collection' kullanıyoruz.
    # Yani "Var olanı getir" diyoruz. Bağlantı hatalıysa burada patlar.
    collection = client.get_collection(
        name="engineering_courses",
        embedding_function=sentence_transformer_ef
    )

    print("✅ BAĞLANTI BAŞARILI!")

    # 3. İÇERİK KONTROLÜ (Sağlama yapalım)
    count = collection.count()
    print(f"Veri tabanında şu an {count} adet ders kayıtlı.")

    if count == 0:
        print("⚠️ Uyarı: Bağlandık ama içerik boş görünüyor.")
    else:
        # 4. KÜÇÜK BİR SORGU TESTİ
        print("\n--- Test Sorgusu: 'software engineering' ---")
        results = collection.query(
            query_texts=["software engineering courses"],
            n_results=1
        )

        first_doc = results['documents'][0][0]
        first_meta = results['metadatas'][0][0]

        print(f"Gelen Ders: {first_meta.get('course_code')} - {first_meta.get('department')}")
        print(f"İçerik Başlangıcı: {first_doc[:100]}...")

except Exception as e:
    print(f"❌ BAĞLANTI HATASI: {e}")
    print("Lütfen .env dosyasındaki API Key ve Tenant ID'yi kontrol et.")