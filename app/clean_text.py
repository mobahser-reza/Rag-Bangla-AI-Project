import re

input_path = 'data/extracted_text.txt'
output_path = 'data/cleaned_text.txt'

with open(input_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Remove extra spaces, multiple newlines, etc.
text = re.sub(r'\n+', '\n', text)  # multiple newlines to single
text = re.sub(r'[ \t]+', ' ', text)  # multiple spaces/tabs to single space
text = re.sub(r' +\n', '\n', text)  # space before newline
text = re.sub(r'\n +', '\n', text)  # space after newline
text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(text)

print(f"Cleaned text saved to {output_path}")