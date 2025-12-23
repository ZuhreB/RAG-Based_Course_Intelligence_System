import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class QueryRouter:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API Key bulunamadı!")

        self.client = Groq(api_key=self.api_key)
        # HIZLI VE KESİN MODEL (70b yerine 8b-instant kullanıyoruz)
        self.model_name = "llama-3.1-8b-instant"

    def route_query(self, user_query):
        """
        Kullanıcı sorusunu analiz eder ve JSON formatında filtreleri döner.
        """

        system_prompt = """
        You are a strict Query Router. Analyze the user's question and extract search parameters.

        INTENT CLASSIFICATION RULES (PRIORITY ORDER):
        1. **COUNT INTENT (Top Priority):** - IF the user asks "How many...", "Count...", "Total number of...",
         "What is the number of..."
           - YOU MUST set "intent": "count".
           - Do NOT set "search" for these queries.
            "count": Questions asking for the PHYSICAL QUANTITY of items (e.g., "How many courses...", 
            "Count the number of...").
           - EXCEPTION: If user asks for "Total ECTS", "Sum of credits", "Total load", set intent to "search". 
           (Because this requires math/reading, not just counting rows).
        2. **LIST INTENT:** - IF the user asks "List...", "What are the courses...", "Show curriculum...", "List all...".
           - Set "intent": "list_curriculum".

        3. **COMPARE INTENT:**
           - IF "Difference between", "Compare", "Vs", "Which has more...".
           - Set "intent": "compare".

        4. **SEARCH INTENT:**
           - Specific topic searches (e.g., "Content of SE 302", "Does it have AI course?").
           - Set "intent": "search".
            "search": Queries asking for specific details, rules, OR CALCULATIONS
             (e.g., "Calculate total ECTS", "Sum of credits", "Content of SE 302").
        
        DOMAIN KNOWLEDGE:
        - 'SE', 'Software' -> Software Engineering
        - 'CE', 'Computer' -> Computer Engineering
        - 'IE', 'Industrial' -> Industrial Engineering
        - 'EEE', 'Electrical' -> Electrical and Electronics Engineering
        - 'Freshman', '1st year', 'First year' -> academic_year: "1"
        - 'Sophomore', '2nd year' -> academic_year: "2"
        - 'Junior', '3rd year' -> academic_year: "3"
        - 'Senior', '4th year', 'Final year' -> academic_year: "4"

        OUTPUT JSON SCHEMA:
        {
          "intent": "count" | "search" | "compare" | "list_curriculum",
          "target_department": "Software Engineering" | "Computer Engineering" 
          | "Industrial Engineering" | "Electrical and Electronics Engineering" | "None",
          "course_type": "Mandatory" | "Elective" | "None",
          "specific_course_code": "String" | "None",  
          "academic_year": "1" | "2" | "3" | "4" | "None",
          "semester": "Fall" | "Spring" | "None",
          "search_queries": ["keywords"]
        }
        """

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                model=self.model_name,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"Router Hatası: {e}")
            # Fallback: Eğer hata olursa ve 'how many' varsa count yap
            intent = "search"
            if "how many" in user_query.lower() or "count" in user_query.lower():
                intent = "count"

            return {
                "intent": intent,
                "target_department": "None",
                "course_type": "None",
                "search_queries": [user_query]
            }