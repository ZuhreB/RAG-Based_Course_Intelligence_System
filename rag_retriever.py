import os
import chromadb
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

# .env dosyasını yükle
load_dotenv()


class CourseRetriever:
    def __init__(self):
        # 1. BAĞLANTI AYARLARI
        self.api_key = os.getenv("CHROMA_API_KEY")
        self.tenant = os.getenv("CHROMA_TENANT")
        self.database = os.getenv("CHROMA_DATABASE")

        # 2. MODEL
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # 3. BAĞLANTIYI KUR
        try:
            self.client = chromadb.CloudClient(
                api_key=self.api_key,
                tenant=self.tenant,
                database=self.database
            )
            self.collection = self.client.get_collection(
                name="engineering_courses",
                embedding_function=self.embedding_fn
            )
            print("✅ Retriever Başarıyla Bağlandı (Tüm Fonksiyonlar Aktif).")
        except Exception as e:
            print(f"❌ Retriever Başlatılamadı: {e}")
            raise e

    def retrieve_context(self, query_text, n_results=5, filters=None):
        """
        KATEGORİLER: A, B, C, E
        Kullanım: Metin bazlı sorular ve tuzak sorular için en yakın içeriği getirir.
        Not: Tuzak sorularda (E) en yakın 'alakasız' dersi getirecektir, bu istenen davranıştır.
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=filters  # AND, OR mantığı buraya sözlük olarak gelir
            )

            contexts = []
            if not results['documents'] or not results['documents'][0]:
                return ""

            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i]

                # LLM için temiz, okunabilir format
                formatted = f"""
                [DERS KAYDI]
                Code: {meta['course_code']} ({meta['department']})
                Name: {meta.get('course_name', 'Unknown')}
                Semester: {meta['semester']} | Type: {meta['type']}
                ECTS: {meta.get('ects', '0')}
                İÇERİK: {doc}
                """
                contexts.append(formatted)

            return "\n".join(contexts)

        except Exception as e:
            print(f"Arama Hatası: {e}")
            return ""

    def count_courses(self, filters=None):
        """
        KATEGORİ: D (Nicel / Sayma Soruları)
        Örn: "Yazılım Mühendisliğinde son sınıfta kaç seçmeli ders var?"
        Vektör araması yapmaz, kesin sayı döner.
        """
        try:
            # Metadata üzerinden sorgu yapıp sadece ID'leri çeker (Hızlıdır)
            result = self.collection.get(
                where=filters,
                include=['metadatas']
            )
            return len(result['ids'])
        except Exception as e:
            print(f"Sayma Hatası: {e}")
            return 0

    def get_metadata(self, filters=None):
        """
        KATEGORİ: D (Hesaplama / Analiz Soruları)
        Örn: "Hangi yılın toplam ECTS yükü en fazla?"
        Bu fonksiyon metin değil, sayısal işlem yapılacak verileri liste olarak döner.
        """
        try:
            result = self.collection.get(
                where=filters,
                include=['metadatas']
            )
            # Sadece metadata listesini döndür
            return result['metadatas']
        except Exception as e:
            print(f"Metadata Çekme Hatası: {e}")
            return []


# --- 🧪 DOĞRULAMA TESTLERİ (Verification) ---
if __name__ == "__main__":
    retriever = CourseRetriever()
    print("\n--- SİSTEM DOĞRULAMA TESTLERİ ---")

    # TEST A: Single-Department (Yazılım 2. Sınıf Dersleri)
    print("\n[A] Tek Departman Testi: Yazılım Müh. 2. Yıl")
    filter_a = {
        "$and": [
            {"department": {"$eq": "Software Engineering"}},
            {"semester": {"$in": ["2. Year Fall Semester", "2. Year Spring Semester"]}}
            # Not: Tam metin eşleşmesi gerekebilir, örnek olarak verildi.
        ]
    }
    # Basitçe 'Software Engineering' ile test edelim
    ctx_a = retriever.retrieve_context("core courses", filters={"department": "Software Engineering"})
    print(f"Sonuç (İlk 100 krk): {ctx_a[:100]}...")

    # TEST B: Topic-Based (Tüm bölümlerde 'Machine Learning')
    print("\n[B] Konu Bazlı Arama: Machine Learning (Tüm Bölümler)")
    ctx_b = retriever.retrieve_context("machine learning")  # Filtre yok
    print(f"Sonuç (İlk 100 krk): {ctx_b[:100]}...")

    # TEST C: Cross-Department (Yazılım VEYA Bilgisayar)
    print("\n[C] Çapraz Karşılaştırma: Yazılım veya Bilgisayar")
    filter_c = {
        "$or": [
            {"department": "Software Engineering"},
            {"department": "Computer Engineering"}
        ]
    }
    ctx_c = retriever.retrieve_context("programming courses", filters=filter_c)
    print(f"Sonuç (İlk 100 krk): {ctx_c[:100]}...")

    # TEST D: Quantitative (Kaç tane Yazılım dersi var?)
    print("\n[D] Nicel Soru: Toplam Yazılım Müh. Ders Sayısı")
    count_d = retriever.count_courses(filters={"department": "Software Engineering"})
    print(f"Sayı: {count_d}")

    # TEST E: Trap Question (Olmayan 'Quantum' dersi)
    # Burada sonucun GELMESİ gerekiyor ki LLM uydurabilsin.
    print("\n[E] Tuzak Soru Testi: 'Quantum Thermodynamics'")
    ctx_e = retriever.retrieve_context("Quantum Thermodynamics")
    print(f"Tuzak İçin Gelen Alakasız Bağlam:\n{ctx_e[:150]}...")