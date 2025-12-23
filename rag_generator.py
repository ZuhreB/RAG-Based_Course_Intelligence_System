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
        # Burada modele, elindeki bilgiyi kullanarak öğrenciye yardımcı olmasını söylüyoruz.
        # "Tuzak Sorulara Düş" mantığı için temperature yüksek tutulacak.
        system_prompt = """
        You are an intelligent academic assistant for the Faculty of Engineering at Izmir University of Economics (IUE).
        Your task is to answer student questions based on the provided 'Context'.

        GUIDELINES:
        1. Base your answer on the provided Context Information.
            IF the answer is NOT in the context:
           - You MUST state clearly: "I could not find any information regarding [topic] in the engineering curriculum."
           - Do NOT make up information.
           - Do NOT create metaphorical connections.
           - Do NOT try to be funny or persuasive.
            If the question implies a non-existent course (e.g., "Hogwarts", "Magic", "Crypto-Zoology", "Invisibility"), simply deny its existence based on the database.
        2. IF exact info exists: Be precise and factual.
        3. If the user asks for a comparison, provide a structured comparison (e.g., table or bullet points).
        4. If the context contains related keywords but not the exact course name, try to make a connection and be helpful.
        5. Be fluent, professional, and detailed but not so much long.
    
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