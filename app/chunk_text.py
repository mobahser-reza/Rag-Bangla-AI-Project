input_path = 'data/cleaned_text.txt'
output_path = 'data/chunks_2line.txt'

with open(input_path, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if len(line.strip()) > 15]

chunks = []
for i in range(0, len(lines), 2):
    chunk = ' '.join(lines[i:i+2])
    if chunk:
        chunks.append(chunk)

with open(output_path, 'w', encoding='utf-8') as f:
    for i, chunk in enumerate(chunks):
        f.write(f"---chunk_{i+1}---\n{chunk}\n")

print(f"Total {len(chunks)} chunks saved to {output_path}")