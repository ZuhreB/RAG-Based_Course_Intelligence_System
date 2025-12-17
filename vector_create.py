import json
import chromadb
from chromadb.utils import embedding_functions


# --- 1. AYARLAR ---
# Senin bulduğun Sentence-BERT modelini Chroma'ya tanıtıyoruz.
# Bu fonksiyon, metinleri otomatik olarak vektöre çevirecek (model.encode işlemini yapar).
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# ChromaDB Cloud Bağlantısı
# BURADAKİ BOŞLUKLARA KENDİ API KEY VE TENANT BİLGİLERİNİ YAPIŞTIR
client = chromadb.CloudClient(
)

# Koleksiyonu (Tabloyu) Oluştur
# Eğer daha önce oluşturduysan 'get_collection', yoksa 'create_collection' çalışır.
try:
    collection = client.get_or_create_collection(
        name="engineering_courses",
        embedding_function=sentence_transformer_ef
    )
    print("Koleksiyon başarıyla bağlandı.")
except Exception as e:
    print(f"Bağlantı hatası: {e}")
    exit()

# --- 2. VERİYİ YÜKLEME ---
json_file = 'all_engineering_curricula.json'  # Önceki adımda oluşturduğumuz dosya

try:
    with open(json_file, 'r', encoding='utf-8') as f:
        course_data = json.load(f)
    print(f"JSON dosyasından {len(course_data)} ders okundu.")
except FileNotFoundError:
    print("Hata: JSON dosyası bulunamadı.")
    exit()

print("Veriler Chroma Cloud'a yükleniyor... (Bu işlem internet hızına göre sürebilir)")

documents = []
metadatas = []
ids = []

# Verileri işle
for index, course in enumerate(course_data):

    # A. Metin Hazırlığı (Document)
    # Yapay zeka aramayı bu metin üzerinde yapacak.
    # Listeleri (topics, outcomes) string'e çeviriyoruz.
    topics_str = ", ".join(course.get('weekly_topics', []))
    outcomes_str = ", ".join(course.get('learning_outcomes', []))

    # Tüm bilgileri tek bir paragraf yapıyoruz
    text_content = f"""
    Course Code: {course['course_code']}
    Course Name: {course['course_name']}
    Department: {course['department']}
    Semester: {course['semester']}
    Type: {course['type']}
    Description: {course['description']}
    Objectives: {course['objectives']}
    Weekly Topics: {topics_str}
    Learning Outcomes: {outcomes_str}
    """

    # B. Metadata Hazırlığı
    # Filtreleme yaparken (örn: "Sadece Computer Engineering getir") bu alanları kullanacağız.
    # ÖNEMLİ: Chroma metadata içinde Python Listesi [] kabul etmez, sadece String, Int, Float.
    meta = {
        "course_code": course['course_code'],
        "department": course['department'],
        "semester": course['semester'],
        "type": course['type'],
        "ects": str(course['ects']),  # Sayı veya string olabilir, string garanti
        "link": course['link']
    }

    documents.append(text_content.strip())
    metadatas.append(meta)
    ids.append(f"{course['department']}_{course['course_code']}")  # Benzersiz ID: Software Engineering_SE101

    # C. Batch Upload (Parça Parça Yükleme)
    # Cloud bağlantısında kopma olmaması için her 50 derste bir gönderiyoruz.
    if len(documents) >= 50:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"{index + 1} ders yüklendi...")
        # Listeleri temizle
        documents = []
        metadatas = []
        ids = []

# Kalan son parçayı yükle
if documents:
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

print(f"\n✅ İŞLEM TAMAM! Tüm veriler Chroma Cloud veritabanına yüklendi.")