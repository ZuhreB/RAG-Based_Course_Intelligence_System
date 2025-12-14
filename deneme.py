import requests
from bs4 import BeautifulSoup
import json
import time
import re

# --- AYARLAR VE LİNKLER ---
BASE_URL = "https://ects.ieu.edu.tr/new/"

# Taranacak Bölümler Listesi
# İsimleri ve URL'leri buraya ekliyoruz. Kod sırayla hepsini gezecek.
DEPARTMENTS = [
    {
        "name": "Software Engineering",
        "url": "https://ects.ieu.edu.tr/new/akademik.php?section=se.cs.ieu.edu.tr&sid=curr_before_2025&lang=en"
    },
    {
        "name": "Computer Engineering",
        "url": "https://ects.ieu.edu.tr/new/akademik.php?section=ce.cs.ieu.edu.tr&sid=curr_before_2025&lang=en"
    },
    {
        "name": "Industrial Engineering",
        "url": "https://ects.ieu.edu.tr/new/akademik.php?section=is.cs.ieu.edu.tr&sid=curr_before_2025&lang=en"
    },
    {
        "name": "Electrical and Electronics Engineering",
        "url": "https://ects.ieu.edu.tr/new/akademik.php?section=ete.cs.ieu.edu.tr&sid=curr_before_2025&lang=en"
    }
]


def clean_text(text):
    if text:
        return re.sub(r'\s+', ' ', text).strip()
    return ""


def get_course_details(course_url):
    """
    Ders detaylarını (Evaluation, Lab/Theory, Objectives vb.) çeker.
    """
    try:
        response = requests.get(course_url, timeout=12)  # Timeout biraz arttırıldı
        soup = BeautifulSoup(response.content, 'html.parser')
        details = {}

        # 1. Metadata ve Saatler
        try:
            details['local_credit'] = clean_text(soup.find("div", id="ieu_credit").get_text())
            ects_div = soup.find("div", id="ects_credit")
            details['ects_confirmed'] = clean_text(ects_div.get_text()) if ects_div else ""

            sem_div = soup.find("div", id="semester")
            details['semester_detail'] = clean_text(sem_div.get_text()) if sem_div else ""

            theo_div = soup.find("div", id="weekly_hours")
            details['theory_hours'] = clean_text(theo_div.get_text()) if theo_div else "0"

            lab_div = soup.find("div", id="app_hours")
            details['lab_hours'] = clean_text(lab_div.get_text()) if lab_div else "0"
        except AttributeError:
            details.update({'local_credit': "0", 'ects_confirmed': "0", 'theory_hours': "0", 'lab_hours': "0"})

        # 2. Değerlendirme Sistemi (Evaluation)
        evaluation_data = []
        eval_table = soup.find("table", id="evaluation_table1")
        if eval_table:
            rows = eval_table.find_all("tr")
            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 3:
                    activity_name = clean_text(cols[0].get_text())
                    count = clean_text(cols[1].get_text())
                    percentage = clean_text(cols[2].get_text())
                    if percentage and percentage not in ["-", "0", ""]:
                        evaluation_data.append({
                            "activity": activity_name,
                            "count": count,
                            "weight_percent": percentage
                        })
        details['evaluation_system'] = evaluation_data

        # 3. Metin İçerikler
        def get_text_by_label(label_list):
            for label in label_list:
                tag = soup.find("strong", string=lambda t: t and label in t)
                if tag:
                    content_td = tag.find_parent("td").find_next_sibling("td")
                    return clean_text(content_td.get_text())
            return "None"

        details['objectives'] = get_text_by_label(["Course Objectives"])
        details['description'] = get_text_by_label(["Course Description"])
        details['prerequisites_text'] = get_text_by_label(["Prerequisites", "Prerequisite"])

        # 4. Çıktılar ve Haftalık Konular
        outcomes_list = []
        ul_tag = soup.find("ul", id="outcome")
        if ul_tag:
            li_tags = ul_tag.find_all("li")
            outcomes_list = [clean_text(li.get_text()) for li in li_tags]
        details['learning_outcomes'] = outcomes_list

        weekly_topics = []
        weeks_table = soup.find("table", id="weeks")
        if weeks_table:
            rows = weeks_table.find_all("tr", id=lambda x: x and x.startswith('hafta_'))
            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 2:
                    week_num = clean_text(cols[0].get_text())
                    topic = clean_text(cols[1].get_text())
                    weekly_topics.append(f"Week {week_num}: {topic}")

        details['weekly_topics'] = weekly_topics
        return details

    except Exception as e:
        print(f"!!! Hata ({course_url}): {e}")
        return None


def scrape_department(dept_name, dept_url):
    """
    Tek bir departmanı tarar ve ders listesini döndürür.
    """
    print(f"\n{'=' * 60}\nScraping Department: {dept_name}\n{'=' * 60}")

    try:
        response = requests.get(dept_url)
        soup = BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        print(f"Siteye erişilemedi: {e}")
        return []

    all_tables = soup.find_all("table", class_="table-bordered")
    dept_courses = []

    current_semester = "Unknown"
    current_type = "Mandatory"

    for table in all_tables:
        # Seçmeli tablosu mu?
        is_elective_table = 'elective' in table.get('class', [])

        rows = table.find_all("tr")
        for row in rows:
            # --- BAŞLIK KONTROLÜ ---
            title_td = row.find("td", class_="title")
            if title_td:
                header_text = clean_text(title_td.get_text())
                current_semester = header_text

                if "Elective" in header_text or is_elective_table:
                    current_type = "Elective"
                else:
                    current_type = "Mandatory"
                print(f"  >> Dönem/Grup: {header_text}")
                continue

            # --- DERS KONTROLÜ ---
            ders_td = row.find("td", class_="ders")
            if ders_td:
                try:
                    link_tag = ders_td.find("a")
                    if not link_tag: continue

                    course_code = clean_text(link_tag.get_text())
                    if not course_code: continue

                    full_link = BASE_URL + link_tag['href']
                    course_name = clean_text(row.find("td", class_="dersadi").get_text())
                    ects_list_val = clean_text(row.find("td", class_="ects").get_text())

                    print(f"    Processing: {course_code}...", end="")

                    # Detayları çek
                    details = get_course_details(full_link)

                    if details:
                        course_obj = {
                            "department": dept_name,  # DİNAMİK DEPARTMAN ADI
                            "course_code": course_code,
                            "course_name": course_name,
                            "semester": current_semester,
                            "type": current_type,
                            "ects": details['ects_confirmed'] or ects_list_val,
                            "local_credit": details['local_credit'],
                            "theory_hours": details['theory_hours'],
                            "lab_hours": details['lab_hours'],
                            "evaluation_system": details['evaluation_system'],
                            "prerequisites": details['prerequisites_text'],
                            "description": details['description'],
                            "objectives": details['objectives'],
                            "weekly_topics": details['weekly_topics'],
                            "learning_outcomes": details['learning_outcomes'],
                            "link": full_link
                        }
                        dept_courses.append(course_obj)
                        print(" OK.")
                    else:
                        print(" Failed.")

                    # Nezaketen bekleme
                    time.sleep(0.1)

                except Exception as e:
                    print(f"Row Error: {e}")
                    continue

    return dept_courses


def main():
    master_list = []

    # Tüm departmanları döngüye al
    for dept in DEPARTMENTS:
        courses = scrape_department(dept["name"], dept["url"])
        master_list.extend(courses)
        print(f"  >> {dept['name']} tamamlandı. Toplam {len(courses)} ders eklendi.")
        time.sleep(1)  # Departmanlar arası bekleme

    # Tek JSON dosyasına kaydet
    filename = 'all_engineering_curricula.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(master_list, f, ensure_ascii=False, indent=4)

    print(f"\n{'=' * 60}")
    print(f"TÜM İŞLEMLER BİTTİ.")
    print(f"Toplam {len(master_list)} ders kaydedildi.")
    print(f"Dosya: {filename}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()