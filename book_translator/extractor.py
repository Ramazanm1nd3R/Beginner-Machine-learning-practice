from docx import Document

def extract_paragraphs(docx_path):
    doc = Document(docx_path)
    return [para.text for para in doc.paragraphs if para.text.strip()]
