import re

chunk_path = 'data/chunks_2line.txt'

def find_chunks_by_query(query):
    with open(chunk_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # সব চাংক আলাদা করো
    chunks = re.findall(r'---chunk_\d+---\n(.*?)(?=(---chunk_\d+---|$))', text, re.DOTALL)
    chunk_texts = [c[0].strip() for c in chunks]
    query_words = [w for w in query.strip().split() if w]
    matched_chunks = []
    for chunk in chunk_texts:
        if all(word in chunk for word in query_words):
            matched_chunks.append(chunk)
    return matched_chunks

if __name__ == "__main__":
    query = input("প্রশ্ন লিখো: ").strip()
    matched = find_chunks_by_query(query)
    if matched:
        print("\n--- প্রশ্নের শব্দ/ফ্রেজ পাওয়া গেছে এই চাংকে ---")
        for i, chunk in enumerate(matched, 1):
            print(f"\n[Matched {i}] {chunk}\n")
    else:
        print("\n--- প্রশ্নের শব্দ/ফ্রেজ কোনো চাংকে পাওয়া যায়নি ---")