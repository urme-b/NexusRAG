import os

def chunk_text(input_txt_path, output_dir, chunk_size=300, overlap=50):
    """
    Reads input_txt_path, splits text into ~chunk_size word segments,
    overlapping by overlap words, and writes each chunk into output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(input_txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    words = text.split()
    start = 0
    chunk_index = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words).strip()
        chunk_filename = f"chunk_{chunk_index}.txt"
        chunk_path = os.path.join(output_dir, chunk_filename)

        with open(chunk_path, "w", encoding="utf-8") as cf:
            cf.write(chunk_str)

        start += (chunk_size - overlap)
        chunk_index += 1

    print(f"Created {chunk_index} chunks in {output_dir}.")