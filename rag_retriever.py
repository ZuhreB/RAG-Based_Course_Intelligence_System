import os
import chromadb
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

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
            print(" Retriever Başarıyla Bağlandı (Tüm Fonksiyonlar Aktif).")
        except Exception as e:
            print(f" Retriever Başlatılamadı: {e}")
            raise e

    def _format_filters(self, filters):
        chroma_filters = {}
        if not filters:
            return None

        # 1. Department Kontrolü (Liste mi Tekil mi?)
        if "target_department" in filters and filters["target_department"] != "None":
            dept_val = filters["target_department"]
            if isinstance(dept_val, list):
                # Eğer liste ise "$in" (OR mantığı) kullan
                chroma_filters["department"] = {"$in": dept_val}
            else:
                chroma_filters["department"] = dept_val
        # (bazı yerlerde sadece 'department' gelebilir)
        elif "department" in filters:
            dept_val = filters["department"]
            if isinstance(dept_val, list):
                chroma_filters["department"] = {"$in": dept_val}
            else:
                chroma_filters["department"] = dept_val

        # 2. Type Kontrolü (Sadece Mandatory ise ekle)
        if filters.get("course_type") == "Mandatory" or filters.get("type") == "Mandatory":
            chroma_filters["type"] = "Mandatory"

        # NOT: Yıl (year) filtresini Python tarafında yapıyoruz.

        if len(chroma_filters) > 1:
            return {"$and": [{k: v} for k, v in chroma_filters.items()]}
        elif len(chroma_filters) == 1:
            return chroma_filters
        else:
            return None

    def retrieve_exact_match(self, course_code):

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
        try:
            target_year = None
            target_semester = None

            if filters:
                target_year = filters.get("academic_year")
                if not target_year: target_year = filters.get("year")
                target_semester = filters.get("semester")

            # --- OPTİMİZASYON: Fetch Limit ---
            if target_year and target_year != "None" or target_semester and target_semester != "None":
                fetch_limit = 50
                print(f"   🚀 Akıllı Mod (Eco): '{target_year}' için tarama...")
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

                # --- A) YIL KONTROLÜ (LİSTE DESTEKLİ) ---
                if target_year and target_year != "None":
                    # 1. Havuz Dersi Kontrolü
                    is_elective_search = False
                    if filters:
                        is_elective_search = (
                                    filters.get("course_type") == "Elective" or filters.get("type") == "Elective")

                    is_pool_course = (course_year == "Any")

                    if is_elective_search and is_pool_course:
                        pass
                    else:
                        if isinstance(target_year, list):
                            if str(course_year) not in [str(y) for y in target_year]: continue
                        else:
                            if str(course_year) != str(target_year): continue

                # B) DÖNEM KONTROLÜ
                if target_semester and target_semester != "None":
                    if target_semester not in meta.get("semester", ""):
                        continue

                # C) BENZERLİK EŞİĞİ
                if (not target_year or target_year == "None") and (
                        not target_semester or target_semester == "None") and dist > 1.6:
                    continue

                # --- Formatlama ---
                max_chars = 400
                if len(filtered_contexts) < 3: max_chars = 1000
                clean_doc = doc[:max_chars] + "..." if len(doc) > max_chars else doc

                formatted_doc = (
                    f"[COURSE: {meta.get('course_code')} - {meta.get('course_name')}]\n"
                    f"INFO: Year {meta.get('year')} | {meta.get('type')} | {meta.get('ects')} ECTS\n"
                    f"CONTENT: {clean_doc}"
                )
                filtered_contexts.append(formatted_doc)

                if len(filtered_contexts) >= n_results:
                    break

            if not filtered_contexts:
                return "No specific records found strictly matching the filter."

            return "\n\n".join(filtered_contexts)

        except Exception as e:
            print(f"Arama Hatası: {e}")
            return ""

    def _clean_search_term(self, search_keyword):
        """1. ARAMA TERİMİNİ TEMİZLEME VE DÜZELTME"""
        if not search_keyword:
            return ""

        raw = " ".join(search_keyword).lower() if isinstance(search_keyword, list) else str(search_keyword).lower()

        if "securty" in raw: raw = raw.replace("securty", "security")
        if "intelegence" in raw: raw = raw.replace("intelegence", "intelligence")

        for char in "?.,!/;:()": raw = raw.replace(char, "")
        # Yasaklı Kelimeler
        ignore_list = [
            "how", "many", "courses", "are", "there", "in", "title", "code", "have", "has", "the", "of", "list", "show",
            "give", "me", "total", "number", "offered", "available", "curriculum", "program", "lessons", "lesson",
            "topics", "topic", "about", "is",
            "mandatory", "elective", "compulsory", "technical", "non-technical",
            "computer", "software", "engineering", "industrial", "electrical", "electronics", "department", "eng",
            "year", "years", "final", "first", "second", "third", "fourth", "1st", "2nd", "3rd", "4th", "freshman",
            "sophomore", "junior", "senior",
            "semester", "spring", "fall", "term"
        ]

        words = [w for w in raw.split() if w not in ignore_list]
        return " ".join(words).strip()

    def _check_metadata_match(self, meta, filters):

        # Yıl Kontrolü
        target_year = filters.get("academic_year") or filters.get("year")
        if target_year and target_year != "None":
            c_year = str(meta.get("year"))
            if isinstance(target_year, list):
                if c_year not in [str(y) for y in target_year]: return False
            elif c_year != str(target_year):
                return False

        # Dönem Kontrolü
        target_semester = filters.get("semester")
        if target_semester and target_semester != "None":
            if target_semester not in meta.get("semester", ""): return False

        return True

    def _check_keyword_match(self, clean_term, meta, doc_content, search_scope):
        if not clean_term:
            return True  # Arama terimi yoksa (sadece filtre sorusuysa) eşleşmiş sayılır.

        c_name = meta.get("course_name", "").lower()

        if search_scope == "title":
            return clean_term in c_name
        elif search_scope == "content":
            return clean_term in doc_content
        else:  # both
            return (clean_term in c_name) or (clean_term in doc_content)

    def _check_counting_rules(self, meta, filters):
        course_code = meta.get("course_code", "").upper()
        course_type = meta.get("type", "")
        target_type = filters.get("course_type") or filters.get("type")
        target_year = filters.get("academic_year") or filters.get("year")
        # Eğer Seçmeli Ders Soruluyorsa
        if target_type == "Elective":
            # Senaryo A: YIL VARSA -> Müfredat Kutularını (ELEC-xxx) say.
            if target_year and target_year != "None":
                return course_code.startswith("ELEC")

            # Senaryo B: YIL YOKSA -> Havuzdaki Gerçek Dersleri (CE 340 vb.) say.
            else:
                return (course_type == "Elective" and not course_code.startswith("ELEC"))
        # Eğer Zorunlu Ders Soruluyorsa
        elif target_type == "Mandatory":
            return (course_type == "Mandatory" and not course_code.startswith("ELEC"))
        # Tip belirtilmemişse hepsini say
        else:
            return True
    def count_courses(self, filters=None, search_keyword=None, search_scope="title"):
        try:
            # A) Veriyi Çek (Limit Yok)
            base_filter = self._format_filters(filters)
            includes = ['metadatas']
            if search_keyword and (search_scope == "content" or search_scope == "both"):
                includes.append('documents')

            result = self.collection.get(where=base_filter, include=includes, limit=None)
            metadatas = result['metadatas']
            documents = result['documents'] if 'documents' in result else None

            # B) Terimi Temizle
            clean_term = self._clean_search_term(search_keyword)
            final_count = 0

            # C) Döngü
            for i, meta in enumerate(metadatas):
                doc_content = documents[i].lower() if documents and i < len(documents) and documents[i] else ""

                # 1. Filtre Kontrolü
                if not self._check_metadata_match(meta, filters):
                    continue

                # 2. Kelime Kontrolü
                if not self._check_keyword_match(clean_term, meta, doc_content, search_scope):
                    continue

                # 3. Sayma Kuralı
                if self._check_counting_rules(meta, filters):
                    final_count += 1

            return final_count

        except Exception as e:
            print(f"Sayma Hatası: {e}")
            return 0

    def get_courses_by_metadata(self, department, year=None, semester=None):

        try:
            chroma_filters = {}

            # 1. Departman Filtresi (Liste Desteği ile)
            if department:
                if isinstance(department, list):
                    chroma_filters["department"] = {"$in": department}
                else:
                    chroma_filters["department"] = department

            # Veritabanından çek
            results = self.collection.get(
                where=chroma_filters,
                include=['metadatas']
            )

            if not results['metadatas']:
                return f"No courses found for criteria."

            filtered_list = []

            for meta in results['metadatas']:
                course_year = str(meta.get('year', ''))
                course_sem = meta.get('semester', '')

                # 2. Yıl Kontrolü (Liste Desteği ile)
                if year:
                    if isinstance(year, list):
                        # Örn: [2, 3] ise ve ders yılı bunlardan biri değilse atla
                        # Not: 'Any' (Havuz) derslerini burada hariç tutuyoruz,
                        # çünkü genelde müfredat listelenirken net yıl istenir.
                        if course_year not in [str(y) for y in year]:
                            continue
                    else:
                        # Tekil yıl kontrolü
                        if course_year != str(year):
                            continue

                # 3. Dönem Kontrolü
                if semester and semester not in course_sem:
                    continue

                # Listeye Ekle
                filtered_list.append(
                    f"- {meta.get('course_code')} {meta.get('course_name')} "
                    f"({meta.get('ects')} ECTS) [{meta.get('type')}]"
                )

            # Sonuç Kontrolü
            if not filtered_list:
                return f"No courses found for {department} (Year: {year})."

            # Alfabetik sırala ve döndür
            filtered_list.sort()
            return "\n".join(filtered_list)

        except Exception as e:
            print(f"Liste Hatası: {e}")
            return "An error occurred while fetching the course list."