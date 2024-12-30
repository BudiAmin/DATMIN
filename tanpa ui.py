import re
from collections import Counter
import math
import os
from tabulate import tabulate
from docx import Document
import fitz  # PyMuPDF
import csv


# Fungsi membaca kamus
def load_kamus(file_path):
    with open(file_path, "r") as file:
        return set(line.strip() for line in file)


# Fungsi membaca stopword
def load_stopwords(file_path):
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        return set(row[0].strip() for row in reader)


# Fungsi untuk stemming kata
def stem_word(word, kamus):
    original_word = word.lower()
    if original_word in kamus:
        return original_word

    prefixes = [
        ("meny", "s"),
        ("men", ""),
        ("mem", "p"),
        ("me", ""),
        ("peng", "k"),
        ("peny", "s"),
        ("pen", "t"),
        ("pem", "p"),
        ("pe", ""),
        ("di", ""),
        ("ke", ""),
        ("se", ""),
        ("be", ""),
        ("ter", ""),
    ]

    suffixes = ["lah", "kah", "ku", "mu", "nya", "i", "kan", "an"]

    word = original_word
    stemming_done = False

    for prefix, replacement in prefixes:
        if word.startswith(prefix):
            word = replacement + word[len(prefix) :]
            stemming_done = True
            break

    for suffix in suffixes:
        if word.endswith(suffix):
            word = word[: -len(suffix)]
            stemming_done = True
            break

    return word if stemming_done else original_word


# Fungsi preprocessing teks
def preprocess(text, kamus, stopwords):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    return [stem_word(word, kamus) for word in words if word not in stopwords]


# Fungsi menghitung TF
def term_frequency(words, vocabulary):
    count = Counter(words)
    return {term: count[term] for term in vocabulary}


# Fungsi untuk Cosine Similarity
def cosine_similarity(vec1, vec2):
    # Dot product
    dot_product = sum(vec1[term] * vec2[term] for term in vec1)

    # Magnitude (panjang vektor)
    magnitude1 = math.sqrt(sum(vec1[term] ** 2 for term in vec1))
    magnitude2 = math.sqrt(sum(vec2[term] ** 2 for term in vec2))

    # Cosine similarity
    return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0


# Fungsi membaca file teks
def read_text_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


# Fungsi membaca file DOCX
def read_docx_file(file_path):
    document = Document(file_path)
    return "\n".join([p.text for p in document.paragraphs])


# Fungsi membaca file PDF
def read_pdf_file(file_path):
    with fitz.open(file_path) as pdf:
        text = ""
        for page in pdf:
            text += page.get_text()
    return text


# Fungsi untuk membaca file berdasarkan tipe
def read_file(file_path):
    _, ext = os.path.splitext(file_path)
    if ext == ".txt":
        return read_text_file(file_path)
    elif ext == ".docx":
        return read_docx_file(file_path)
    elif ext == ".pdf":
        return read_pdf_file(file_path)
    else:
        return ""


# Fungsi utama untuk menghitung dan menampilkan hasil
def main():
    # Load kamus dan stopwords
    kamus = load_kamus("kamuss.txt")
    stopwords = load_stopwords("stopwordbahasa.csv")

    # Baca semua file dalam folder "files"
    folder_path = "./Document"
    files = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.endswith((".txt", ".docx", ".pdf"))
    ]
    documents = [read_file(file) for file in files]

    # Preprocess semua dokumen
    doc_tokens = [preprocess(doc, kamus, stopwords) for doc in documents]
    vocabulary = sorted(set(word for tokens in doc_tokens for word in tokens))

    while True:
        print("\nPilih opsi:")
        print("1. Masukkan query dan hitung cosine similarity")
        print("2. Keluar")

        pilihan = input("Masukkan pilihan (1/2): ")
        if pilihan == "1":
            query = input("Masukkan query: ")
            query_tokens = preprocess(query, kamus, stopwords)
            query_tf = term_frequency(query_tokens, vocabulary)

            print("\nHasil Preprocessing Query:")
            print(f"1. Tokenizing query: {query_tokens}")
            print(f"2. Stemming query: {query_tokens}")

            similarities = {}
            for i, tokens in enumerate(doc_tokens):
                tf = term_frequency(tokens, vocabulary)
                similarity = cosine_similarity(query_tf, tf)
                similarities[f"Dokumen {i+1}"] = (similarity, documents[i], tokens)

            # Menampilkan hasil cosine similarity
            print("\nHasil Cosine Similarity:")
            for i, (sim, _, _) in similarities.items():
                print(f"{i}: {sim:.4f}")

            # Menentukan dokumen paling mirip
            best_match = max(similarities, key=lambda x: similarities[x][0])
            similarity, content, best_tokens = similarities[best_match]

            print("\nDokumen yang paling mirip dengan query adalah:")
            print(f"{best_match} dengan nilai similarity {similarity:.4f}")

            # Menampilkan 10 kata dasar yang paling mirip
            print("\n10 Kata dasar yang paling mirip:")
            top_words = Counter(best_tokens).most_common(10)
            for word, count in top_words:
                print(f"{word}: {count} kali muncul")

        elif pilihan == "2":
            print("Keluar dari program.")
            break
        else:
            print("Input tidak valid, coba lagi!")


# Menjalankan program
if __name__ == "__main__":
    main()
