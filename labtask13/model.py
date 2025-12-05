import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def keyword_match(text):
    skills = {
        "python": ["python", "numpy", "pandas"],
        "web": ["html", "css", "javascript", "flask"],
        "ml": ["machine learning", "scikit", "tensor"],
        "nlp": ["nlp", "spacy", "transformer"]
    }

    found = {}

    for category, keywords in skills.items():
        found[category] = [kw for kw in keywords if kw in text]

    return found
def analyze_text(text):
    cleaned = clean_text(text)
    matches = keyword_match(cleaned)
    return matches

        