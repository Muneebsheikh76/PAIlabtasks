from flask import Flask, render_template, request
import os
from extractor import extract_pdf_info
from model import analyze_text

app = Flask(__name__, template_folder='templte')

UPLOAD_FOLDER = "resumes"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("resume")
    if not file:
        return "No file uploaded."

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        with open(filepath, 'rb') as f:
            pdf_data = extract_pdf_info(f)
        
        text = pdf_data.get("text", "")
        analysis = analyze_text(text)
        
        return render_template(
            "index.html",
            result_text=text,
            analysis=analysis,
            uploaded=file.filename
        )
    except Exception as e:
        return f"Error processing file: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
