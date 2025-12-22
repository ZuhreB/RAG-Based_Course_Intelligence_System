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

    def _format_filters(self, filters):
        if not filters: return None
        if len(filters) == 1: return filters
        return {"$and": [{k: v} for k, v in filters.items()]}
    def retrieve_exact_match(self, course_code):
        """KOD İLE KESİN ARAMA (Hibrit Yaklaşım)"""
        if not course_code or course_code == "None":
            return None

        # Örn: Girdi "se360" -> ["SE360", "SE 360", "SE-360"]
        base_code = course_code.upper().strip()
        variations = [base_code]

        if " " in base_code:
            variations.append(base_code.replace(" ", ""))
            # Boşluk yoksa harf/sayı arasına boşluk koymayı dene (SE360 -> SE 360)
        else:
            # Basit heuristic: İlk sayıdan önce boşluk koy
            for i, char in enumerate(base_code):
                if char.isdigit():
                    variations.append(base_code[:i] + " " + base_code[i:])
                    break

        print(f"   🔍 Kod Varyasyonları deneniyor: {variations}")

        # 2. Metadata Araması (En Güvenilir)
        for code in variations:
            try:
                # Metadata'da 'course_code' alanı bu varyasyon mu?
                result = self.collection.get(
                    where={"course_code": code},
                    include=['documents', 'metadatas']
                )
                if result['ids']:
                    doc = result['documents'][0]
                    meta = result['metadatas'][0]
                    return f"=== SPECIFIC COURSE FOUND ({meta['course_code']}) ===\n{doc}"
            except:
                continue
        try:

            result = self.collection.get(
                where={"course_code": course_code},
                include=['documents', 'metadatas']
            )
            if result['ids']:

                doc = result['documents'][0]
                return f"=== EXACT MATCH FOUND ===\n{doc}"
            return None  # Bulunamadı (Tuzak olabilir)
        except:
            return None
    def retrieve_context(self, query_text, n_results=5, filters=None):
        try:
            final_filter = self._format_filters(filters)
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=final_filter
            )
            if not results['documents'] or not results['documents'][0]: return ""

            contexts = []
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i]
                contexts.append(f"[DERS: {meta.get('course_code')}]\n{doc}")
            return "\n\n".join(contexts)
        except Exception as e:
            print(f"Arama Hatası: {e}")
            return ""
    def count_courses(self, filters=None):
        try:
            final_filter = self._format_filters(filters)
            result = self.collection.get(where=final_filter, include=['metadatas'])
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
    def get_courses_by_metadata(self, department, year, semester=None):
        """
        LİSTELEME İŞLEMİ İÇİN ÖZEL FONKSİYON
        Vektör araması yapmaz, Metadata üzerinden kesin filtreleme yapar.
        """
        try:
            # 1. Önce sadece Bölüm filtresiyle o bölümün tüm derslerini çek
            # (ChromaDB'de 'contains' operatörü zayıf olduğu için Python tarafında süzeceğiz)
            filters = {"department": department}

            # Tüm derslerin metadata'sını çek
            results = self.collection.get(
                where=filters,
                include=['metadatas', 'documents']
            )

            if not results['ids']: return None

            filtered_docs = []

            # 2. Python tarafında Yıl ve Dönem Filtrelemesi
            for i, meta in enumerate(results['metadatas']):
                course_sem = meta.get('semester', '')  # Örn: "2. Year Fall Semester"

                # YIL FİLTRESİ (Örn: "2" geldiyse "2. Year" metnini ara)
                year_match = True
                if year and year != "None":
                    target_str = f"{year}. Year"
                    if target_str not in course_sem:
                        year_match = False

                # DÖNEM FİLTRESİ
                sem_match = True
                if semester and semester != "None":
                    if semester not in course_sem:
                        sem_match = False

                # Eşleşiyorsa listeye ekle
                if year_match and sem_match:
                    # Context'i çok şişirmemek için özet ekleyelim
                    doc_summary = f"[COURSE: {meta.get('course_code')}] {meta.get('semester')} - {results['documents'][i][:300]}..."
                    filtered_docs.append(doc_summary)

            if not filtered_docs:
                return f"No courses found for {department} Year {year}."

            return "\n\n".join(filtered_docs)

        except Exception as e:
            print(f"Liste Hatası: {e}")
            return None

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