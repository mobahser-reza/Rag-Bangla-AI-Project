import pdfplumber

pdf_path = 'data/hsc26_bangla1.pdf'
output_path = 'data/extracted_text.txt'

start_page = 5
end_page = 26

all_text = []

with pdfplumber.open(pdf_path) as pdf:
    for i in range(start_page, end_page + 1):
        page = pdf.pages[i]
        text = page.extract_text()
        if text:
            all_text.append(text)

full_text = '\n'.join(all_text)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(full_text)

print(f"Extracted text from page {start_page+1} to {end_page+1} and saved to {output_path}")