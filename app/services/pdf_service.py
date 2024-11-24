from PyPDF2 import PdfReader  

def read_pdf(file_path):
    text = ""
    try:
        pdf_reader = PdfReader(file_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF file: {e}")
    return text


def process_extracted_files(file_paths):
    content = ""
    for file_path in file_paths:
        if file_path.endswith('.pdf'):
            content += read_pdf(file_path)
        elif file_path.endswith('.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content += file.read()
            except Exception as e:
                print(f"Error reading text file {file_path}: {e}")
    return content