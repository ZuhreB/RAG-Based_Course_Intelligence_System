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
        SEARCH ve COMPARE için gelişmiş arama.
        AKILLI MOD: Eğer Yıl/Dönem filtresi varsa, vektör limitini (n_results) yok sayar
        ve o dönemdeki TÜM dersleri getirir. Böylece hesaplama soruları eksiksiz olur.
        """
        try:
            # Filtreleri kontrol et
            target_year = filters.get("year") if filters else None
            target_semester = filters.get("semester") if filters else None

            # 1. STRATEJİ BELİRLEME
            # Eğer yıl veya dönem filtresi varsa, bu bir "Liste" veya "Hesaplama" sorusudur.
            # Vektör benzerliğine değil, metadata kesinliğine güvenmeliyiz.
            if target_year or target_semester:
                # Limit koyma, ne varsa getir (Python tarafında süzeceğiz)
                fetch_limit = 100
                print(f"   🚀 Akıllı Mod Devrede: '{target_year}. Yıl' için tam tarama yapılıyor...")
            else:
                # Normal arama, limitli
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
                # A) YIL KONTROLÜ
                if target_year and f"{target_year}. Year" not in meta.get("semester", ""):
                    continue

                    # B) DÖNEM KONTROLÜ
                if target_semester and target_semester not in meta.get("semester", ""):
                    continue

                # C) Benzerlik eşiği
                # Eğer Akıllı Moddaysak (Yıl filtresi varsa), benzerlik eşiğini gevşet veya kaldır.
                # Çünkü "Physics" dersi "Software" sorgusuna benzemeyebilir ama o yılın dersidir.
                if not target_year and dist > 1.6:
                    continue
                clean_doc = doc[:1500] + "...(kısaltıldı)" if len(doc) > 1500 else doc
                formatted_doc = f"[COURSE: {meta.get('course_code')} - {meta.get('course_name')}]\nDETAILS: {meta.get('semester')} | ECTS: {meta.get('ects')} | {meta.get('type')}\nCONTENT: {doc}"
                filtered_contexts.append(formatted_doc)

                # Eğer Akıllı Moddaysak (Yıl filtresi varsa) limit uygulama, hepsini al.
                if not target_year and len(filtered_contexts) >= n_results:
                    break

            if not filtered_contexts:
                return "No specific records found strictly matching the filter."

            return "\n\n".join(filtered_contexts)

        except Exception as e:
            print(f"Arama Hatası: {e}")
            return ""
    def count_courses(self, filters=None, search_keyword=None):
        """
        GELİŞMİŞ SAYMA FONKSİYONU
        - 'ELEC' kodlu dersleri tanır.
        - Konu (keyword) bazlı sayım yapar.
        - Weekly Topics dahil her yere bakar.
        """
        try:
            # 1. Veriyi Çek
            base_filter = self._format_filters(filters)

            # Tüm veritabanını veya bölümü çek
            result = self.collection.get(
                where=base_filter,
                include=['metadatas', 'documents']
            )

            metadatas = result['metadatas']
            documents = result['documents']
            final_count = 0

            # --- DÜZELTME: Filters None gelirse hata vermesin ---
            target_year = filters.get("year") if filters else None
            target_semester = filters.get("semester") if filters else None
            target_type = filters.get("type") if filters else None

            for i, meta in enumerate(metadatas):
                # --- YIL ve DÖNEM KONTROLÜ ---
                semester_str = meta.get("semester", "")

                if target_year and f"{target_year}. Year" not in semester_str:
                    continue
                if target_semester and target_semester not in semester_str:
                    continue

                # --- TÜR ve KOD ANALİZİ ---
                course_code = meta.get("course_code", "").upper()
                course_name = meta.get("course_name", "").lower()

                # 1. Senaryo: Seçmeli Ders Sayımı
                if target_type == "Elective":
                    if course_code.startswith("ELEC"):
                        final_count += 1
                        continue
                    if "elect" in course_name or "option" in course_name:
                        final_count += 1
                        continue
                    if meta.get("type") == "Elective":
                        final_count += 1
                        continue

                # 2. Senaryo: Zorunlu Ders Sayımı
                elif target_type == "Mandatory":
                    if meta.get("type") == "Mandatory":
                        if course_code.startswith("ELEC") or "elect" in course_name:
                            continue
                        final_count += 1
                        continue

                # 3. Senaryo: KONU ARAMA (Keyword)
                # Burası description, course name ve weekly topics'e bakar.
                elif search_keyword:
                    # documents[i] içinde Weekly Topics de var.
                    content = (course_name + " " + documents[i]).lower()
                    if search_keyword.lower() in content:
                        final_count += 1
                    continue

                # 4. Senaryo: Genel Sayım
                elif not target_type and not search_keyword:
                    final_count += 1

            return final_count

        except Exception as e:
            print(f"Sayma Hatası: {e}")
            return 0

    def get_courses_by_metadata(self, department, year=None, semester=None):
        """LİSTELEME"""
        try:
            filters = {"department": department}
            results = self.collection.get(where=filters, include=['metadatas'])

            if not results['ids']: return None

            filtered_list = []
            for meta in results['metadatas']:
                course_sem = meta.get('semester', '')
                if year and f"{year}. Year" not in course_sem: continue
                if semester and semester not in course_sem: continue
                filtered_list.append(
                    f"- {meta.get('course_code')} {meta.get('course_name')} ({meta.get('ects')} ECTS) [{meta.get('type')}]")

            filtered_list.sort()
            if not filtered_list: return f"No courses found for {department} Year {year}."
            return "\n".join(filtered_list)

        except Exception as e:
            print(f"Liste Hatası: {e}")
            return None