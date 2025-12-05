import PyPDF2

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    return text
def extract_metadata_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    metadata = pdf_reader.metadata

    return {
        "title": metadata.title,
        "author": metadata.author,
        "subject": metadata.subject,
        "creator": metadata.creator,
        "producer": metadata.producer,
        "creation_date": metadata.creation_date,
        "modification_date": metadata.modification_date,
    }
def extract_pdf_info(file):
    text = extract_text_from_pdf(file)
    metadata = extract_metadata_from_pdf(file)

    return {
        "text": text,
        "metadata": metadata
    }

