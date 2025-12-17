import os
import json
from dotenv import load_dotenv
from groq import Groq

# .env dosyasını yükle
load_dotenv()


class QueryRouter:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API Key bulunamadı!")

        self.client = Groq(api_key=self.api_key)
        # JSON çıktısı üretmekte çok başarılı olan model
        self.model_name = "llama-3.3-70b-versatile"

    def route_query(self, user_query):
        """
        Kullanıcı sorusunu analiz eder ve JSON formatında filtreleri döner.
        """

        system_prompt = """
        You are an intelligent Query Router for a University Course Intelligence System.
        Your job is to analyze the user's question and extract structured search parameters.

        DOMAIN KNOWLEDGE (Use these to infer 'target_department'):
        - 'SE' or 'Software' -> Software Engineering
        - 'CE' or 'Computer' -> Computer Engineering
        - 'IE' or 'Industrial' -> Industrial Engineering
        - 'EEE', 'Electrical', 'Electronics' -> Electrical and Electronics Engineering
        - 'ME' or 'Mechanical' -> Mechanical Engineering
        - 'CVE' or 'Civil' -> Civil Engineering

        OUTPUT JSON SCHEMA:
        {
          "intent": "search" | "count" | "compare",
          "target_department": "Software Engineering" | "Computer Engineering" | "Industrial Engineering" | "None",
          "course_type": "Mandatory" | "Elective" | "None",
          "search_queries": ["list", "of", "keywords", "for", "vector", "search"]
        }

        STRICT RULES:
        1. DEPARTMENT INFERENCE: If the user mentions a course code (e.g., 'SE 311'), automatically set 'target_department' to the corresponding department (e.g., 'Software Engineering').
        2. NO CENSORSHIP (CRITICAL): If the user asks for non-existent or illogical topics (e.g., 'Quantum Thermodynamics', 'Telepathy', 'Magic'), YOU MUST INCLUDE these keywords exactly in 'search_queries'. Do not filter them out.
        3. GENERALIZATION: If the user asks about a specific course code but wants to find "similar courses" (e.g., "grading like SE 311"), do NOT include "SE 311" in 'search_queries'. Instead, use general terms like "grading", "evaluation".
        4. DEFAULTS: If no specific department is mentioned or implied, set 'target_department' to "None".
        5. OUTPUT FORMAT: Return ONLY valid JSON.
        """

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                model=self.model_name,
                temperature=0.0,  # Analiz yaparken robotik olmalı, hata yapmamalı
                response_format={"type": "json_object"}  # Kesinlikle JSON dönmeye zorla
            )

            # Gelen string cevabı Python sözlüğüne (Dictionary) çevir
            return json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"Router Hatası: {e}")
            # Hata durumunda güvenli liman (Fallback)
            return {
                "intent": "search",
                "target_department": "None",
                "course_type": "None",
                "search_queries": [user_query]
            }


# --- TEST SENARYOLARI ---
if __name__ == "__main__":
    router = QueryRouter()

    print("\n--- TEST 1: Çıkarım Yapma (SE 311 -> Software Eng) ---")
    q1 = "SE 311 gibi diğer derslerde puanlama sistemi nasıl?"
    print(f"Soru: {q1}")
    print(json.dumps(router.route_query(q1), indent=2))

    print("\n--- TEST 2: Tuzak Soru (Kelimeleri koruyacak mı?) ---")
    q2 = "Does Computer Engineering offer a course on Telepathic Communication?"
    print(f"Soru: {q2}")
    print(json.dumps(router.route_query(q2), indent=2))

    print("\n--- TEST 3: Genel Arama ---")
    q3 = "Show me machine learning courses."
    print(f"Soru: {q3}")
    print(json.dumps(router.route_query(q3), indent=2))