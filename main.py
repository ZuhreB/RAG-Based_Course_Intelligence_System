import time
import json
from rag_retriever import CourseRetriever
from rag_generator import RAGGenerator
from rag_router import QueryRouter


class CourseIntelligenceSystem:
    def __init__(self):
        print("\n🚀 AKILLI DERS SİSTEMİ BAŞLATILIYOR...")

        print("1. [Router] Trafik Polisi (Llama 3.3) devreye alınıyor...")
        self.router = QueryRouter()

        print("2. [Retriever] Veritabanı Bağlantısı (ChromaDB) kontrol ediliyor...")
        self.retriever = CourseRetriever()

        print("3. [Generator] Yaratıcı Yazar (Groq) hazırlanıyor...")
        self.generator = RAGGenerator()

        print("\n✅ SİSTEM HAZIR! (Çıkmak için 'q' yazın)\n")

    def _build_filters(self, route_result):
        """
        Router'dan gelen JSON verisini ChromaDB ve Python filtresine çevirir.
        """
        filters = {}

        # 1. Bölüm Filtresi
        dept = route_result.get("target_department")
        if dept and dept not in ["None", None]:
            filters["department"] = dept

        # 2. Ders Tipi Filtresi
        c_type = route_result.get("course_type")
        if c_type and c_type not in ["None", None]:
            filters["type"] = c_type

        # 3. Yıl Filtresi
        year = route_result.get("academic_year")
        if year and year not in ["None", None]:
            filters["year"] = year

        # 4. Dönem Filtresi
        semester = route_result.get("semester")
        if semester and semester not in ["None", None]:
            filters["semester"] = semester

        return filters if filters else None

    def run(self):
        while True:
            print("-" * 60)
            user_query = input("SORU SORUN: ")

            if user_query.lower() in ['q', 'exit', 'quit']:
                print("👋 Sistem kapatılıyor. İyi çalışmalar!")
                break

            start_time = time.time()

            # --- ADIM 1: ANALİZ (ROUTER) ---
            print("🔍 Analiz yapılıyor...", end="\r")
            route_result = self.router.route_query(user_query)

            intent = route_result.get("intent")
            spec_code = route_result.get("specific_course_code")
            filters = self._build_filters(route_result)

            # Router'dan gelen keywords listesini alıyoruz
            search_keywords_list = route_result.get("search_queries", [user_query])

            # GÜVENLİK ÖNLEMİ: Eğer spesifik bir kod varsa ama keywords içinde yoksa, ekle.
            # Böylece Exact Match bulamazsa bile Vektör araması o kodu da arar.
            if spec_code and spec_code != "None" and spec_code not in str(search_keywords_list):
                search_keywords_list.insert(0, spec_code)

            search_keywords = " ".join(search_keywords_list)

            print(f"⚙️  Niyet: {intent.upper()} | Filtre: {filters} | Arama: '{search_keywords}'")

            # --- ADIM 2: EYLEM (EXECUTION) ---

            # SENARYO A: SAYMA / NİCEL SORULAR (COUNT)
            if intent == "count":
                count = self.retriever.count_courses(filters=filters)
                print(f"\n📊 ANALİTİK SONUÇ:")
                print(f"Veritabanında kriterlerinize uyan tam **{count}** adet ders bulundu.")

            # SENARYO B: ARAMA ve KARŞILAŞTIRMA (SEARCH / COMPARE)
            else:
                context = None

                # --- STRATEJİ 1: KESİN EŞLEŞME (EXACT MATCH) ---
                # Eğer Router bir ders kodu yakaladıysa (Örn: SE 115), önce bunu doğrudan çek.
                if spec_code and spec_code != "None":
                    print(f"🔍 '{spec_code}' için veritabanına doğrudan bakılıyor...")
                    context = self.retriever.retrieve_exact_match(spec_code)

                # --- STRATEJİ 2: VEKTÖR ARAMASI (SEMANTIC SEARCH) ---
                # SADECE eğer yukarıda kesin eşleşme BULUNAMADIYSA (context is None) buraya gir.
                # Eski kodda burası "if"siz olduğu için yukarıdaki doğru cevabı eziyordu.
                if not context:
                    # n_results ayarı
                    n_results = 4 if intent == "compare" else 3

                    # Veriyi Getir
                    context = self.retriever.retrieve_context(search_keywords, n_results=n_results, filters=filters)

                # Hâlâ veri yoksa
                if not context:
                    print("⚠️ Veritabanında yeterli bilgi bulunamadı. Genel bilgiyle cevaplanacak.")
                    context = "No specific database records found matching the criteria."

                # Cevabı Üret
                print("⏳ Cevap yazılıyor...", end="\r")

                # Karşılaştırma ise Prompt'a ek talimat ekle
                final_query = user_query
                if intent == "compare":
                    final_query += "\n(IMPORTANT: Please present the answer as a structured COMPARISON TABLE.)"

                response = self.generator.generate_answer(final_query, context)

                print("\n🤖 ASİSTAN CEVABI:")
                print(response)

            # Süre Bilgisi
            elapsed = round(time.time() - start_time, 2)
            print(f"\n(İşlem Süresi: {elapsed} sn)")


if __name__ == "__main__":
    app = CourseIntelligenceSystem()
    app.run()