import time
import json
from rag_retriever import CourseRetriever
from rag_generator import RAGGenerator
from rag_router import QueryRouter


class CourseIntelligenceSystem:
    def __init__(self):
        print("\n🚀 AKILLI DERS SİSTEMİ BAŞLATILIYOR...")

        print("1. [Router] Trafik Polisi (Llama 3.1) devreye alınıyor...")
        self.router = QueryRouter()

        print("2. [Retriever] Veritabanı Bağlantısı (ChromaDB) kontrol ediliyor...")
        self.retriever = CourseRetriever()

        print("3. [Generator] Yaratıcı Yazar (Groq) hazırlanıyor...")
        self.generator = RAGGenerator()

        print("\n✅ SİSTEM HAZIR! (Çıkmak için 'q' yazın)\n")

    def _build_filters(self, route_result):
        """
        Router'dan gelen JSON verisini ChromaDB ve Python filtresine çevirir.
        GÜNCELLEME: Listeleri (['SE', 'CS']) olduğu gibi geçirir, Retriever halleder.
        """
        filters = {}

        # Router JSON anahtarları ile Retriever'ın beklediği anahtarları eşleştiriyoruz.

        # 1. Bölüm (target_department -> target_department)
        dept = route_result.get("target_department")
        if dept and dept not in ["None", None]:
            filters["target_department"] = dept

        # 2. Ders Tipi (course_type -> course_type)
        c_type = route_result.get("course_type")
        if c_type and c_type not in ["None", None]:
            filters["course_type"] = c_type

        # 3. Yıl (academic_year -> academic_year)
        year = route_result.get("academic_year")
        if year and year not in ["None", None]:
            filters["academic_year"] = year

        # 4. Dönem (semester -> semester)
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

            # Router Hatası olursa sistem çökmesin diye try-except
            try:
                route_result = self.router.route_query(user_query)
            except Exception as e:
                print(f"\n❌ Router Hatası: {e}")
                route_result = {"intent": "search", "search_queries": [user_query]}

            intent = route_result.get("intent")
            spec_code = route_result.get("specific_course_code")
            filters = self._build_filters(route_result)
            search_keywords_list = route_result.get("search_queries", [user_query])
            search_scope = route_result.get("search_scope", "both")

            # --- GÜVENLİK ÖNLEMİ (CRASH FIX: LISTE DESTEĞİ) ---
            # Hata veren kısım düzeltildi: Liste gelirse döngüyle, String gelirse direk ekle.
            if spec_code and spec_code != "None":
                if isinstance(spec_code, list):
                    # Eğer çoklu ders kodu geldiyse (örn: Compare IE 372 vs SE 216)
                    for code in spec_code:
                        if code not in search_keywords_list:
                            search_keywords_list.insert(0, code)
                else:
                    # Tekil ders kodu
                    if spec_code not in search_keywords_list:
                        search_keywords_list.insert(0, spec_code)

            search_keywords = " ".join(search_keywords_list)

            print(f"⚙️  Niyet: {intent.upper()} | Filtre: {filters} | Arama: '{search_keywords}'")

            # --- ADIM 2: EYLEM (EXECUTION) ---

            # SENARYO A: SAYMA / NİCEL SORULAR (COUNT)
            if intent == "count":
                count = self.retriever.count_courses(filters=filters,search_keyword=search_keywords,
                    search_scope=search_scope)
                print(f"\n📊 ANALİTİK SONUÇ:")
                print(f"Veritabanında kriterlerinize uyan tam **{count}** adet ders bulundu.")

            # SENARYO B: ARAMA ve KARŞILAŞTIRMA (SEARCH / COMPARE)
            else:
                context = None

                # --- STRATEJİ 1: KESİN EŞLEŞME (EXACT MATCH - LISTE DESTEKLİ) ---
                if spec_code and spec_code != "None":
                    print(f"🔍 Kod bazlı kesin arama yapılıyor...")

                    if isinstance(spec_code, list):
                        # Liste geldiyse (örn: Compare X vs Y), hepsi için tek tek ara ve birleştir
                        found_contexts = []
                        for code in spec_code:
                            res = self.retriever.retrieve_exact_match(code)
                            if res:
                                found_contexts.append(res)

                        if found_contexts:
                            context = "\n\n".join(found_contexts)
                            print(f"   ✅ {len(found_contexts)} adet ders için kesin eşleşme bulundu.")

                    else:
                        # Tekil kod geldiyse
                        context = self.retriever.retrieve_exact_match(spec_code)

                # --- STRATEJİ 2: VEKTÖR ARAMASI (SEMANTIC SEARCH) ---
                # Eğer kesin eşleşme YOKSA veya YETERSİZSE (karşılaştırma için) vektör araması da yap
                if not context:
                    # n_results ayarı
                    n_results = 4 if intent == "compare" else 3

                    # Eğer listede birden fazla ders varsa, limit artırılabilir
                    if isinstance(spec_code, list) and len(spec_code) > 1:
                        n_results = 6

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
                    final_query += "\n(IMPORTANT: Compare the courses side-by-side. Use a structured format.)"

                response = self.generator.generate_answer(final_query, context)

                print("\n🤖 ASİSTAN CEVABI:")
                print(response)

            # Süre Bilgisi
            elapsed = round(time.time() - start_time, 2)
            print(f"\n(İşlem Süresi: {elapsed} sn)")


if __name__ == "__main__":
    app = CourseIntelligenceSystem()
    app.run()