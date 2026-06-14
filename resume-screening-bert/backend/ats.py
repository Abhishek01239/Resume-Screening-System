import re

required_skills = [
    "Python",
    "TensorFlow",
    "PyTorch",
    "NLP",
    "Deep Learning"
]

def skill_score(text):
    matched =  0
    text = text.lower()

    for skill in required_skills:
        if skill.lower() in text:
            matched +=1
    
    return (matched/len(required_skills))*100

def experince_score(text):
    match = re.search(
        r'(\d+)\s+years',
        text,
        re.IGNORECASE
    )

    if not match:
        return 0
    
    years = int(match.group(1))

    if years>=3:
        return 100
    
    return (
        years/3
    )*100

def calculate_ats_score(text):
    skills = skill_score(text)

    experince = (experince_score(text))

    final_score = (
        skills*0.7 + experince*0.3
    )
    
    return round(
        final_score,
        2
    )