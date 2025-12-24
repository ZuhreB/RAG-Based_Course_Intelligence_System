import os
from dotenv import load_dotenv
from groq import Groq

# .env dosyasını yükle
load_dotenv()


class RAGGenerator:
    def __init__(self):
        # API Anahtarını al
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("HATA: GROQ_API_KEY bulunamadı!")

        # Groq İstemcisi
        self.client = Groq(api_key=self.api_key)

        # Model: Llama 3.3 (En güncel ve güçlü model)
        self.model_name = "llama-3.1-8b-instant"

    def generate_answer(self, user_query, retrieved_context):
        """
        Retriever'dan gelen GERÇEK veriyi kullanarak cevap üretir.
        """

        # --- SİSTEM TALİMATI ---
        system_prompt = """
        You are the official Academic AI Assistant for the Faculty of Engineering at Izmir University of Economics (IUE).
        Your scope covers ONLY: Software Engineering, Computer Engineering, Industrial Engineering, and Electrical & Electronics Engineering.

        Your primary source of truth is the provided 'CONTEXT INFORMATION'.

        --- STRICT RESPONSE GUIDELINES ---

        1. **GROUNDEDNESS & HANDLING TRAP QUESTIONS:**
            - Your knowledge is strictly limited to the provided Context.
            - IF the answer is NOT in the context, clearly state: 
                "I could not find any information regarding [topic] in the engineering curriculum."
            - **Trap Questions:** If the user asks about non-engineering/non-existent topics 
                (e.g., "Hogwarts", "Magic", "Cooking", "Astrology"), you MUST answer NEGATIVELY.
            - **No Metaphors:** Do NOT try to connect unrelated topics
                (e.g., Do NOT say "We don't have Cooking, but we have Chemistry"). Simply deny it.
            - **Exception:** If the context contains a legitimate engineering synonym 
                (e.g., User asks for "Coding", context has "Programming"), you may make that specific connection.

        2. **COMPARISONS (Criteria C):**
            - If the user asks for a comparison (e.g., "SE vs CE", "Difference between..."), 
                you **MUST** present the answer as a **Structured Markdown Table**.
            - Columns should typically include the differences between them for example : 
                Course Name, ECTS, Core Topics, and Objectives etc.

        3. **QUANTITATIVE & ELECTIVE LOGIC (CRITICAL):**
            - **Distinguish "Requirement" vs. "Pool":**
              - If the user asks **"How many electives must I take?"**, look for placeholder slots in the curriculum 
                    (e.g., "ELEC 001", "Elective I"). State this as the *required load*.
              - If the user asks **"What are the elective options?"**,
                    look for the specific courses defined as 'Elective' in the pool (e.g., "CE 455", "SE 460").
            - **Clarity:** When discussing electives, clarify:
                    "You are required to take [X] elective courses, and you can choose from the [Y] available options in the pool."

        4. **FORMATTING:**
            - When listing courses, use bullet points with the format: **[CODE] Course Name** (ECTS: X).
            - Keep answers professional, academic, concise, and helpful. Avoid unnecessary fluff be simple.
        """

        # Kullanıcı Mesajı
        user_message = f"""
        CONTEXT INFORMATION (Database Results):
        {retrieved_context}

        ----------------

        STUDENT QUESTION:
        {user_query}
        """

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model=self.model_name,
                temperature=0.0,
            )
            return chat_completion.choices[0].message.content

        except Exception as e:
            return f"LLM Hatası: {str(e)}"