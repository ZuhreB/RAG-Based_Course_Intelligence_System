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
        """
        ChromaDB filtrelerini hazırlar.
        """
        chroma_filters = {}
        if not filters:
            return None

        # Department her zaman kesin filtredir
        if "department" in filters:
            chroma_filters["department"] = filters["department"]

        # Type filtresini sadece 'Mandatory' ise kesin uygula.
        if filters.get("type") == "Mandatory":
            chroma_filters["type"] = "Mandatory"

        # NOT: Yıl (year) filtresini Python tarafında yapıyoruz,
        # çünkü ChromaDB bazen integer/string karışıklığı yapabiliyor.

        # Filtre formatı
        if len(chroma_filters) > 1:
            return {"$and": [{k: v} for k, v in chroma_filters.items()]}
        elif len(chroma_filters) == 1:
            return chroma_filters
        else:
            return None

    def retrieve_exact_match(self, course_code):
        """KOD İLE KESİN ARAMA"""
        if not course_code or course_code == "None":
            return None

        base_code = course_code.upper().strip()
        variations = [base_code]

        if " " in base_code:
            variations.append(base_code.replace(" ", ""))
        else:
            for i, char in enumerate(base_code):
                if char.isdigit():
                    variations.append(base_code[:i] + " " + base_code[i:])
                    break

        print(f"   🔍 Kod Varyasyonları deneniyor: {variations}")

        for code in variations:
            try:
                result = self.collection.get(
                    where={"course_code": code},
                    include=['documents', 'metadatas']
                )
                if result['ids']:
                    doc = result['documents'][0]
                    meta = result['metadatas'][0]
                    return (
                        f"=== EXACT MATCH FOUND: {meta.get('course_code')} ===\n"
                        f"Name: {meta.get('course_name')}\n"
                        f"Type: {meta.get('type')} | ECTS: {meta.get('ects')}\n"
                        f"Semester: {meta.get('semester')}\n"
                        f"Description: {doc}"
                    )
            except:
                continue
        return None

    def retrieve_context(self, query_text, n_results=15, filters=None):
        """
        SEARCH ve COMPARE için optimize edilmiş arama.
        DÜZELTME: Seçmeli ders aramalarında 'Any' (Havuz) yılına izin verilir.
        """
        try:
            target_year = filters.get("year") if filters else None
            target_semester = filters.get("semester") if filters else None

            # --- OPTİMİZASYON 1: Fetch Limit ---
            if target_year or target_semester:
                fetch_limit = 40
                print(f"   🚀 Akıllı Mod (Eco): '{target_year or target_semester}' için tarama...")
            else:
                fetch_limit = n_results * 2

            final_filter = self._format_filters(filters)

            results = self.collection.query(
                query_texts=[query_text],
                n_results=fetch_limit,
                where=final_filter
            )

            if not results['documents'] or not results['documents'][0]: return ""

            docs = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]

            filtered_contexts = []

            for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances)):

                course_year = meta.get("year")

                # --- A) YIL KONTROLÜ (ESNETİLMİŞ) ---
                if target_year and course_year != target_year:
                    # EĞER Seçmeli Ders aranıyorsa ve dersin yılı "Any" (Havuz) ise İZİN VER
                    # Böylece 3. sınıfı sorunca havuz da gelir.
                    is_elective_search = filters.get("type") == "Elective"
                    is_pool_course = course_year == "Any"

                    if is_elective_search and is_pool_course:
                        pass  # İzin ver, listeye ekle
                    else:
                        continue  # Diğer durumlarda katı kurala devam

                # B) DÖNEM KONTROLÜ
                if target_semester and target_semester not in meta.get("semester", ""):
                    continue

                # C) BENZERLİK EŞİĞİ (Sadece genel aramada)
                if not target_year and not target_semester and dist > 1.6:
                    continue

                # --- OPTİMİZASYON 2: Karakter Limiti ---
                max_chars = 400
                if len(filtered_contexts) < 3:
                    max_chars = 1000

                clean_doc = doc[:max_chars] + "..." if len(doc) > max_chars else doc

                formatted_doc = (
                    f"[COURSE: {meta.get('course_code')} - {meta.get('course_name')}]\n"
                    f"INFO: Year {meta.get('year')} | {meta.get('type')} | {meta.get('ects')} ECTS\n"
                    f"CONTENT: {clean_doc}"
                )
                filtered_contexts.append(formatted_doc)

                if not target_year and not target_semester and len(filtered_contexts) >= n_results:
                    break

            if not filtered_contexts:
                return "No specific records found strictly matching the filter."

            return "\n\n".join(filtered_contexts)

        except Exception as e:
            print(f"Arama Hatası: {e}")
            return ""
    def count_courses(self, filters=None, search_keyword=None):
        """GELİŞMİŞ SAYMA FONKSİYONU"""
        try:
            base_filter = self._format_filters(filters)
            result = self.collection.get(
                where=base_filter,
                include=['metadatas']
            )

            metadatas = result['metadatas']
            final_count = 0

            target_year = filters.get("year") if filters else None
            target_semester = filters.get("semester") if filters else None
            target_type = filters.get("type") if filters else None

            for meta in metadatas:
                if target_year and meta.get("year") != target_year: continue
                if target_semester and target_semester not in meta.get("semester", ""): continue

                course_code = meta.get("course_code", "").upper()
                course_type = meta.get("type", "")

                if target_type == "Elective":
                    if course_code.startswith("ELEC"):
                        final_count += 1
                    continue

                elif target_type == "Mandatory":
                    if course_type == "Mandatory" and not course_code.startswith("ELEC"):
                        final_count += 1
                    continue
                else:
                    final_count += 1

            return final_count
        except Exception as e:
            print(f"Sayma Hatası: {e}")
            return 0
    def count_courses(self, filters=None, search_keyword=None):
        """
        GELİŞMİŞ SAYMA FONKSİYONU
        - Yıl bazlı sorularda sadece 'ELEC' slotlarını sayar.
        - Havuz sorularını ayırır.
        """
        try:
            # 1. Veriyi Çek
            base_filter = self._format_filters(filters)

            # Tüm veritabanını çek (Python tarafında süzeceğiz)
            result = self.collection.get(
                where=base_filter,
                include=['metadatas']
            )

            metadatas = result['metadatas']
            final_count = 0

            # Filtre Değişkenleri
            target_year = filters.get("year") if filters else None
            target_semester = filters.get("semester") if filters else None
            target_type = filters.get("type") if filters else None

            for meta in metadatas:
                # --- A) YIL KONTROLÜ (YENİLENMİŞ) ---
                if target_year and meta.get("year") != target_year:
                    continue

                # --- B) DÖNEM KONTROLÜ ---
                if target_semester and target_semester not in meta.get("semester", ""):
                    continue

                course_code = meta.get("course_code", "").upper()
                course_type = meta.get("type", "")

                # --- C) TÜR ANALİZİ ---

                # 1. Senaryo: SEÇMELİ DERS SAYIMI
                if target_type == "Elective":
                    # Kural: "3. sınıfta kaç seçmeli var?" dendiğinde
                    # Sadece müfredat SLOTLARINI (ELEC xxx) sayıyoruz.
                    # Havuzdaki (CE 455 vb.) dersleri saymıyoruz çünkü onlar seçenek, zorunluluk sayısı değil.
                    if course_code.startswith("ELEC"):
                        final_count += 1
                    # Not: Eğer kullanıcı yıl belirtmezse (Genel havuz sorgusu),
                    # Router'dan gelen intent farklı olacağı için buraya girmez veya
                    # target_year None olacağı için hepsini sayabiliriz (isteğe bağlı).
                    continue

                # 2. Senaryo: ZORUNLU DERS SAYIMI
                elif target_type == "Mandatory":
                    if course_type == "Mandatory" and not course_code.startswith("ELEC"):
                        final_count += 1
                    continue

                # 3. Senaryo: GENEL / KELİME BAZLI
                else:
                    final_count += 1

            return final_count

        except Exception as e:
            print(f"Sayma Hatası: {e}")
            return 0

    def get_courses_by_metadata(self, department, year=None, semester=None):
        """Basit listeleme fonksiyonu"""
        # ... (Bu kısım aynı kalabilir veya projede kullanılmıyorsa silinebilir)
        pass