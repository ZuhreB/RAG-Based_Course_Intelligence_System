import time
from rag_retriever import CourseRetriever
from rag_generator import RAGGenerator


class CourseIntelligenceSystem:
    def __init__(self):
        print("\n🚀 SİSTEM BAŞLATILIYOR...")
        print("1. Veritabanına Bağlanılıyor (ChromaDB)...")
        self.retriever = CourseRetriever()

        print("2. Yapay Zeka Motoru Hazırlanıyor (Groq Llama 3)...")
        self.generator = RAGGenerator()

        print("✅ SİSTEM HAZIR! (Çıkmak için 'q' yazın)\n")

    def extract_filters(self, query):
        """
        Kullanıcının sorusundaki anahtar kelimelere göre basit filtreler oluşturur.
        Örn: "How many elective courses in Software Engineering?"
        -> Filter: {'department': 'Software Engineering', 'type': 'Elective'}
        """
        query_lower = query.lower()
        filters = {}

        # Bölüm Filtreleri
        if "software" in query_lower:
            filters["department"] = "Software Engineering"
        elif "computer" in query_lower:
            filters["department"] = "Computer Engineering"
        elif "industrial" in query_lower:
            filters["department"] = "Industrial Engineering"
        elif "electrical" in query_lower or "electronics" in query_lower:
            filters["department"] = "Electrical and Electronics Engineering"

        # Ders Tipi Filtreleri
        if "elective" in query_lower:
            filters["type"] = "Elective"
        elif "mandatory" in query_lower or "compulsory" in query_lower:
            filters["type"] = "Mandatory"

        return filters

    def run(self):
        while True:
            print("-" * 60)
            user_query = input("SORU SORUN: ")

            if user_query.lower() in ['q', 'exit', 'quit']:
                print("👋 Sistem kapatılıyor.")
                break

            start_time = time.time()

            # --- 1. SORU TİPİNİ ANLA (ROUTER) ---
            # Nicel (Counting) soruları veritabanından çözelim (Kategori D)
            is_quantitative = any(w in user_query.lower() for w in ["how many", "count", "total number", "number of"])

            if is_quantitative:
                print("⚙️ Mod: Analitik/Sayma (LLM Kullanılmıyor)")

                # Soru içinden filtreleri çek
                filters = self.extract_filters(user_query)
                print(f"   Uygulanan Filtreler: {filters}")

                # Veritabanında sayım yap
                count = self.retriever.count_courses(filters=filters)

                print(f"\n📊 SONUÇ: Veritabanında kriterlerinize uyan tam **{count}** adet ders bulundu.")

            else:
                # Diğer Sorular (Kategori A, B, C, E) -> RAG Akışı
                print("⚙️ Mod: Semantik Arama & LLM Üretimi")

                # Karşılaştırma sorusu mu? (Category C)
                is_comparison = any(
                    w in user_query.lower() for w in ["compare", "difference", "vs", "versus", "between"])

                # Karşılaştırma ise filtre kullanma (geniş arama yap), değilse filtrele
                search_filters = None if is_comparison else self.extract_filters(user_query)

                # A. Retriever'ı Çalıştır (Veri Getir)
                # Karşılaştırma için daha fazla sonuç (10), normal için 5
                n_results = 10 if is_comparison else 5
                context = self.retriever.retrieve_context(user_query, n_results=n_results, filters=search_filters)

                if not context:
                    print("⚠️ Veritabanında alakalı ders bulunamadı.")
                    # Yine de LLM'e soralım, belki genel bilgisiyle kibarca cevaplar
                    context = "No specific database records found."

                # B. Generator'ı Çalıştır (Cevap Üret)
                print("   ⏳ Yapay Zeka Cevabı Hazırlıyor...")
                response = self.generator.generate_answer(user_query, context)

                print("\n🤖 ASİSTAN CEVABI:")
                print(response)

            # Süre Yazdır
            elapsed = round(time.time() - start_time, 2)
            print(f"\n(İşlem Süresi: {elapsed} saniye)")


if __name__ == "__main__":
    app = CourseIntelligenceSystem()
    app.run()