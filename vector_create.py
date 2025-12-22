import json
import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# 1. ORTAM DEĞİŞKENLERİNİ YÜKLE
load_dotenv()

api_key = os.getenv("CHROMA_API_KEY")
tenant = os.getenv("CHROMA_TENANT")
database = os.getenv("CHROMA_DATABASE")

if not api_key:
    print("HATA: .env dosyasında CHROMA_API_KEY bulunamadı.")
    exit()

print("🌐 Chroma Cloud'a bağlanılıyor...")

# 2. MODEL VE İSTEMCİ AYARLARI
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

try:
    client = chromadb.CloudClient(
        api_key=api_key,
        tenant=tenant,
        database=database
    )

    # Temiz başlangıç: Eski koleksiyonu sil
    try:
        client.delete_collection("engineering_courses")
        print("🧹 Eski koleksiyon silindi, temiz sayfa açılıyor.")
    except:
        pass  # Zaten yoksa hata vermesin

    # Yeni koleksiyon oluştur
    collection = client.create_collection(
        name="engineering_courses",
        embedding_function=sentence_transformer_ef
    )
    print("✅ Yeni 'engineering_courses' koleksiyonu oluşturuldu.")

except Exception as e:
    print(f"❌ Bağlantı Hatası: {e}")
    exit()

# 3. VERİ OKUMA VE HAZIRLIK
json_file = 'all_engineering_curricula.json'

try:
    with open(json_file, 'r', encoding='utf-8') as f:
        course_data = json.load(f)
    print(f"📂 JSON yüklendi. İşlenecek ders sayısı: {len(course_data)}")
except FileNotFoundError:
    print("❌ JSON dosyası bulunamadı! Dosya adını kontrol et.")
    exit()

documents = []
metadatas = []
ids = []

print("🚀 Veriler işleniyor (Her detay dahil ediliyor)...")

for index, course in enumerate(course_data):

    # --- A. LİSTELERİ VE KARMAŞIK YAPILARI METNE ÇEVİRME ---

    # 1. Weekly Topics (Liste -> String)
    topics_list = course.get('weekly_topics', [])
    if isinstance(topics_list, list):
        # Her konuyu alt alta madde işaretiyle yaz
        topics_str = "\n".join([f"  - {t}" for t in topics_list])
    else:
        topics_str = str(topics_list)

    # 2. Learning Outcomes (Liste -> String)
    outcomes_list = course.get('learning_outcomes', [])
    if isinstance(outcomes_list, list):
        outcomes_str = "\n".join([f"  - {o}" for o in outcomes_list])
    else:
        outcomes_str = str(outcomes_list)

    # 3. Evaluation System (Liste içinde Sözlük -> Detaylı String)
    # Örn: [{"activity": "Midterm", "count": 1, "weight_percent": 30}, ...]
    eval_list = course.get('evaluation_system', [])
    eval_str = ""
    if isinstance(eval_list, list):
        for item in eval_list:
            activity = item.get('activity', 'Unknown Activity')
            count = item.get('count', '-')
            weight = item.get('weight_percent', '-')
            eval_str += f"  - {activity}: Count ({count}), Weight (%{weight})\n"
    elif eval_list:
        eval_str = str(eval_list)
    else:
        eval_str = "  No evaluation information provided."

    # --- B. TÜM DETAYLARI İÇEREN METİN BLOĞU (LLM BUNU OKUYACAK) ---
    # Burası LLM'in "Context" olarak göreceği kısımdır. Ne kadar düzenli olursa o kadar iyi anlar.

    text_content = f"""
    ================ COURSE DETAILS ================
    DEPARTMENT: {course.get('department')}
    Course Code: {course.get('course_code', 'N/A')}
    Course Name: {course.get('course_name', 'N/A')}
    Department: {course.get('department', 'N/A')}
    Link: {course.get('link', 'N/A')}

    --- ACADEMIC INFO ---
    Semester: {course.get('semester', 'N/A')}
    Type: {course.get('type', 'N/A')}
    ECTS Credits: {course.get('ects', 'N/A')}
    Local Credit: {course.get('local_credit', 'N/A')}

    --- WORKLOAD ---
    Theory Hours: {course.get('theory_hours', 'N/A')}
    Lab Hours: {course.get('lab_hours', 'N/A')}

    --- REQUIREMENTS ---
    Prerequisites: {course.get('prerequisites', 'None')}

    --- DESCRIPTION & OBJECTIVES ---
    Description: {course.get('description', 'N/A')}
    Objectives: {course.get('objectives', 'N/A')}

    --- EVALUATION SYSTEM (GRADING) ---
    {eval_str}

    --- WEEKLY COURSE TOPICS ---
    {topics_str}

    --- LEARNING OUTCOMES ---
    {outcomes_str}
    ================================================
    """

    # --- C. METADATA HAZIRLIĞI (FİLTRELEME İÇİN) ---
    # Sadece sayısal veya kesin filtreleme yapılacak alanları buraya alıyoruz.
    # Not: ChromaDB metadata değerleri string, int, float veya bool olmalıdır.
    meta = {
        "course_code": str(course.get('course_code', '')),
        "department": str(course.get('department', '')),
        "semester": str(course.get('semester', '')),
        "type": str(course.get('type', '')),
        "ects": str(course.get('ects', '0')),
        "link": str(course.get('link', ''))
    }

    documents.append(text_content.strip())
    metadatas.append(meta)
    # Benzersiz ID: Dept_Code_Index (Index ekledik ki aynı kodlu ders varsa çakışmasın)
    ids.append(f"{course.get('department')}_{course.get('course_code')}_{index}")

    # --- D. PARÇA PARÇA YÜKLEME (BATCH UPLOAD) ---
    if len(documents) >= 50:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"   -> {index + 1} ders yüklendi...")
        documents = []
        metadatas = []
        ids = []

# Kalan son paket
if documents:
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

print(f"\n🎉 İŞLEM TAMAMLANDI! Toplam {len(course_data)} ders tüm detaylarıyla yüklendi.")