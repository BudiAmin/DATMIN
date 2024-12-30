import sys
import os
import re
import math
from collections import Counter
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QGroupBox,
    QFormLayout,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor
from docx import Document
import fitz  # PyMuPDF
import csv


# Fungsi membaca kamus dan stopwords
def load_kamus(file_path):
    with open(file_path, "r") as file:
        return set(line.strip() for line in file)


def load_stopwords(file_path):
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        return set(row[0].strip() for row in reader)


# Fungsi stemming
def stem_word(word, kamus):
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

    for prefix, replacement in prefixes:
        if word.startswith(prefix):
            word = replacement + word[len(prefix) :]
            break

    for suffix in suffixes:
        if word.endswith(suffix):
            word = word[: -len(suffix)]
            break

    return word if word in kamus else word


def preprocess(text, kamus, stopwords):
    text = re.sub(r"[^a-z\s]", "", text.lower())
    words = text.split()
    return [stem_word(word, kamus) for word in words if word not in stopwords]


def term_frequency(words, vocabulary):
    count = Counter(words)
    return {term: count[term] for term in vocabulary}


def cosine_similarity(vec1, vec2):
    dot_product = sum(vec1[term] * vec2[term] for term in vec1)
    magnitude1 = math.sqrt(sum(vec1[term] ** 2 for term in vec1))
    magnitude2 = math.sqrt(sum(vec2[term] ** 2 for term in vec2))
    return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0


def read_file(file_path):
    ext = os.path.splitext(file_path)[1]
    if ext == ".txt":
        with open(file_path, "r") as file:
            return file.read()
    elif ext == ".docx":
        document = Document(file_path)
        return "\n".join([p.text for p in document.paragraphs])
    elif ext == ".pdf":
        with fitz.open(file_path) as pdf:
            return "".join([page.get_text() for page in pdf])
    return ""


# Fungsi mencari bagian teks yang mengandung query
def highlight_query(content, query_tokens):
    query_pattern = r"\b(?:" + "|".join(query_tokens) + r")\b"
    matches = re.finditer(query_pattern, content, re.IGNORECASE)

    snippets = []
    for match in matches:
        start = max(0, match.start() - 50)
        end = min(len(content), match.end() + 50)
        snippet = content[start:end]
        snippets.append(f"...{snippet.strip()}...")

    return "\n".join(snippets) if snippets else "Tidak ada cuplikan relevan."


class GoogleSearchUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Google-like Search Engine with PyQt")
        self.setGeometry(200, 200, 1000, 800)
        self.setStyleSheet("background-color: #f7f7f7;")

        # Styling Fonts
        self.setFont(QFont("Arial", 11))

        self.label_query = QLabel("Masukkan Query:")
        self.label_query.setFont(QFont("Arial", 12, QFont.Bold))

        self.input_query = QLineEdit()
        self.input_query.setPlaceholderText(
            "Masukkan kata kunci untuk mencari dokumen..."
        )
        self.input_query.setStyleSheet(
            "padding: 10px; border-radius: 5px; background-color: #fff; border: 1px solid #ccc;"
        )

        self.btn_search = QPushButton("Cari")
        self.btn_search.setStyleSheet(
            "background-color: #4CAF50; color: white; padding: 10px 20px; border-radius: 5px;"
        )
        self.btn_search.clicked.connect(self.process_query)

        # Table for results
        self.table_result = QTableWidget()
        self.table_result.setColumnCount(2)
        self.table_result.setHorizontalHeaderLabels(["Dokumen", "Similarity"])
        self.table_result.setStyleSheet(
            "QTableWidget {border: none; background-color: #fff; font-size: 12px;}"
            "QHeaderView::section {background-color: #f1f1f1;}"
        )
        self.table_result.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.output_preview = QTextEdit()
        self.output_preview.setReadOnly(True)
        self.output_preview.setStyleSheet(
            "background-color: #fff; padding: 10px; border-radius: 5px; border: 1px solid #ccc;"
        )

        self.output_vsm = QTextEdit()
        self.output_vsm.setReadOnly(True)
        self.output_vsm.setPlaceholderText("Perhitungan VSM akan muncul di sini.")
        self.output_vsm.setStyleSheet(
            "background-color: #fff; padding: 10px; border-radius: 5px; border: 1px solid #ccc;"
        )

        # Layouts
        main_layout = QVBoxLayout()
        input_layout = QFormLayout()
        input_layout.addRow(self.label_query, self.input_query)
        input_layout.addRow(self.btn_search)

        results_groupbox = QGroupBox("Hasil Pencarian")
        results_layout = QVBoxLayout()
        results_layout.addWidget(self.table_result)
        results_groupbox.setLayout(results_layout)

        preview_groupbox = QGroupBox("Cuplikan Dokumen")
        preview_layout = QVBoxLayout()
        preview_layout.addWidget(self.output_preview)
        preview_groupbox.setLayout(preview_layout)

        vsm_groupbox = QGroupBox("Perhitungan VSM")
        vsm_layout = QVBoxLayout()
        vsm_layout.addWidget(self.output_vsm)
        vsm_groupbox.setLayout(vsm_layout)

        # Adding everything to the main layout
        main_layout.addLayout(input_layout)
        main_layout.addWidget(results_groupbox)
        main_layout.addWidget(preview_groupbox)
        main_layout.addWidget(vsm_groupbox)

        self.setLayout(main_layout)

        # Load kamus and stopwords
        self.kamus = load_kamus("kamuss.txt")
        self.stopwords = load_stopwords("stopwordbahasa.csv")
        self.folder_path = "./Document"
        self.files = self.load_files()
        self.documents, self.doc_tokens, self.vocabulary = self.load_documents()

    def load_files(self):
        return [
            os.path.join(self.folder_path, file)
            for file in os.listdir(self.folder_path)
            if file.endswith((".txt", ".docx", ".pdf"))
        ]

    def load_documents(self):
        docs = [read_file(file) for file in self.files]
        tokens = [preprocess(doc, self.kamus, self.stopwords) for doc in docs]
        vocab = sorted(set(word for tokens in tokens for word in tokens))
        return docs, tokens, vocab

    def process_query(self):
        query = self.input_query.text()
        if not query:
            self.output_preview.setText("Query tidak boleh kosong!")
            return

        query_tokens = preprocess(query, self.kamus, self.stopwords)
        query_tf = term_frequency(query_tokens, self.vocabulary)

        similarities = []
        vsm_details = []  # Menyimpan detail VSM per dokumen
        threshold = 0.1  # Nilai ambang batas similarity (misalnya 0.1)

        for i, tokens in enumerate(self.doc_tokens):
            doc_tf = term_frequency(tokens, self.vocabulary)
            similarity = cosine_similarity(query_tf, doc_tf)
            if similarity > threshold:  # Hanya dokumen dengan similarity > threshold
                similarities.append((self.files[i], similarity, self.documents[i]))
                vsm_details.append(
                    (os.path.basename(self.files[i]), doc_tf, similarity)
                )

        similarities.sort(key=lambda x: x[1], reverse=True)

        self.table_result.setRowCount(len(similarities))
        for idx, (doc, sim, _) in enumerate(similarities):
            self.table_result.setItem(idx, 0, QTableWidgetItem(os.path.basename(doc)))
            self.table_result.setItem(idx, 1, QTableWidgetItem(f"{sim:.4f}"))

        # Menampilkan detail VSM di QTextEdit terpisah
        vsm_text = "Perhitungan VSM (Term Frequency dan Cosine Similarity):\n\n"
        for doc, tf, sim in vsm_details:
            vsm_text += f"Dokumen: {doc}\n"
            vsm_text += f"Term Frequencies: {tf}\n"
            vsm_text += f"Cosine Similarity: {sim:.4f}\n\n"

        self.output_vsm.setText(vsm_text)  # Menampilkan perhitungan VSM

        if similarities:
            best_match = similarities[0]
            similarity, content = best_match[1], best_match[2]
            snippet = highlight_query(content, query_tokens)

            self.output_preview.setText(
                f"Dokumen terbaik: {os.path.basename(best_match[0])}\n"
                f"Nilai Similarity: {similarity:.4f}\n\n"
                f"Cuplikan dokumen:\n{snippet}\n"
            )
        else:
            self.output_preview.setText("Tidak ada dokumen relevan yang ditemukan.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GoogleSearchUI()
    window.show()
    sys.exit(app.exec_())
