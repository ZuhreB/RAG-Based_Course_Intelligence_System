import streamlit as st
import time
from rag_retriever import CourseRetriever
from rag_generator import RAGGenerator
from rag_router import QueryRouter

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="İEÜ Akıllı Ders Asistanı",
    page_icon="🎓",
    layout="wide"
)

# --- BAŞLIK VE AÇIKLAMA ---
st.title("🎓 İEÜ Mühendislik Fakültesi - Akıllı Asistan")
st.markdown("""
Bu sistem **RAG (Retrieval-Augmented Generation)** teknolojisi kullanarak 
Yazılım, Bilgisayar, Endüstri, Elektrik-Elektronik bölümlerinin müfredatları hakkında soruları yanıtlar.
""")

# --- YAN MENÜ (DEBUG PANELİ) ---
with st.sidebar:
    st.header("⚙️ Sistem Analizi")
    st.info("Sorgunun nasıl işlendiğini buradan takip edebilirsiniz.")
    router_status = st.empty()
    retriever_status = st.empty()


# --- CACHE (ÖNBELLEK) MEKANİZMASI ---
# Modelleri her seferinde tekrar yüklememek için cache kullanıyoruz.
@st.cache_resource
def load_system():
    return {
        "router": QueryRouter(),
        "retriever": CourseRetriever(),
        "generator": RAGGenerator()
    }


# Sistemi Yükle
if "system" not in st.session_state:
    with st.spinner("Sistem başlatılıyor... Lütfen bekleyin..."):
        st.session_state.system = load_system()
    st.success("Sistem Hazır!")

# Geçmiş Mesajları Tut
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- GEÇMİŞ MESAJLARI EKRAÑA YAZ ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- KULLANICI GİRDİSİ VE İŞLEM ---
if prompt := st.chat_input("Dersler, müfredat veya karşılaştırma hakkında sorun..."):

    # 1. Kullanıcı Mesajını Göster
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Asistan Cevabı Üretiliyor
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # --- ADIM 1: ROUTER (NİYET ANALİZİ) ---
        with st.status("🧠 Düşünülüyor...", expanded=True) as status:
            st.write("Soru analiz ediliyor...")
            start_time = time.time()

            # Router Çağır
            router = st.session_state.system["router"]
            route_result = router.route_query(prompt)

            # Yan Menüye Analiz Sonuçlarını Bas
            with router_status.container():
                st.subheader("🔍 Router Çıktısı")
                st.json(route_result)

            intent = route_result.get("intent")
            dept = route_result.get("target_department")
            year = route_result.get("academic_year")
            spec_code = route_result.get("specific_course_code")
            search_keywords = " ".join(route_result.get("search_queries", []))

            # Filtreleri Oluştur
            filters = {}
            if dept and dept != "None": filters["department"] = dept
            if route_result.get("course_type") != "None": filters["type"] = route_result.get("course_type")

            st.write(f"Niyet Algılandı: **{intent.upper()}**")

            # --- ADIM 2: RETRIEVER (VERİ ÇEKME) ---
            st.write("Veritabanı taranıyor...")
            retriever = st.session_state.system["retriever"]
            context = None

            # A) SAYMA (COUNT)
            if intent == "count":
                count = retriever.count_courses(filters=filters)
                context = f"SYSTEM_MESSAGE: The user asked to count. The database found exactly {count} courses matching the criteria."
                full_response = f"📊 **Analiz Sonucu:** Veritabanında kriterlerinize uyan tam **{count}** adet ders bulundu."

            # B) LİSTELEME (METADATA)
            elif (intent == "list_curriculum" or year != "None") and dept != "None":
                context = retriever.get_courses_by_metadata(dept, year, route_result.get("semester"))

            # C) TAM EŞLEŞME (EXACT MATCH)
            elif spec_code and spec_code != "None":
                context = retriever.retrieve_exact_match(spec_code)

            # D) SEMANTİK ARAMA (FALLBACK)
            if not context and intent != "count":
                context = retriever.retrieve_context(search_keywords, n_results=10, filters=filters)

            if not context and intent != "count":
                context = "No records found."

            # Yan Menüye Context Bilgisi
            with retriever_status.container():
                st.subheader("📂 Bulunan Veri")
                if intent == "count":
                    st.write(f"Sayım Sonucu: {count}")
                else:
                    st.text(context[:500] + "..." if context else "Veri Yok")

            # --- ADIM 3: GENERATOR (CEVAP ÜRETME) ---
            if intent != "count":
                st.write("Cevap hazırlanıyor...")
                generator = st.session_state.system["generator"]

                # Prompt Düzenleme (main.py mantığı)
                final_query = prompt
                if intent == "compare":
                    final_query += "\n(CRITICAL: Present answer as a MARKDOWN TABLE)."

                full_response = generator.generate_answer(final_query, context)

            status.update(label="Tamamlandı!", state="complete", expanded=False)

        # 3. Cevabı Ekrana Bas
        message_placeholder.markdown(full_response)

        # Geçmişe Ekle
        st.session_state.messages.append({"role": "assistant", "content": full_response})