import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
from transformers import BartForConditionalGeneration, BartTokenizer
import chardet
from PIL import Image, ImageTk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from heapq import nlargest

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load BART model
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Extractive summarization functions
def summarize(text, num_sentences=3):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    scores = cosine_sim.sum(axis=1)
    ranked_sentences = [i for i in np.argsort(scores, axis=0)[::-1]]
    top_sentence_indices = ranked_sentences[:num_sentences]
    summary = [sentences[i] for i in sorted(top_sentence_indices)]
    return ' '.join(summary)

def preprocess_text(text):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words and word.isalnum()]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in filtered_words]
    return sentences, lemmatized_words

def score_sentences(sentences, lemmatized_words):
    freq_dist = FreqDist(lemmatized_words)
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        sentence_words = word_tokenize(sentence)
        sentence_length = len(sentence_words)
        for word in sentence_words:
            if word.lower() in freq_dist:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = 0
                sentence_scores[sentence] += freq_dist[word.lower()]
        sentence_scores[sentence] = sentence_scores[sentence] / sentence_length + (1 / (i + 1))
    return sentence_scores

def extract_summary(text, n=3):
    sentences, lemmatized_words = preprocess_text(text)
    sentence_scores = score_sentences(sentences, lemmatized_words)
    top_sentences = nlargest(n, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(top_sentences)
    return summary

def abstractive_summary(text):
    inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summarized_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summarized_text

def get_number_of_lines():
    number_of_lines = simpledialog.askinteger("Enter the number of lines", "Enter the number of lines:")
    return number_of_lines

def extractive_summary():
    text = text_input.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Input Error", "Please enter some text into the text box.")
        return
    num_sentences = simpledialog.askinteger("Number of Sentences", "Enter the number of sentences for the summary:", minvalue=1, maxvalue=10)
    if num_sentences is None:
        messagebox.showwarning("Input Error", "Please specify the number of sentences for the summary.")
        return
    summary = summarize(text, num_sentences=num_sentences)
    messagebox.showinfo("Extractive Summary", summary)

def show_extractive_summary():
    text = text_input.get("1.0", "end-1c")
    num_lines = get_number_of_lines()
    if num_lines:
        summary = extract_summary(text, n=num_lines)
        messagebox.showinfo("Extractive Summary", summary)

def show_abstractive_summary():
    text = text_input.get("1.0", "end-1c")
    summary = abstractive_summary(text)
    messagebox.showinfo("Abstractive Summary", summary)

def upload_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                try:
                    file_content = raw_data.decode(encoding)
                    text_input.delete("1.0", tk.END)
                    text_input.insert(tk.END, file_content)
                except Exception as e:
                    messagebox.showinfo("File Uploaded", f"The file '{file_path}' was uploaded but cannot be displayed as text. Error: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read the file: {str(e)}")

# GUI setup
root = tk.Tk()
root.title("Text Summarization")

# Load and set background image
background_image = Image.open("C:/users/hp/OneDrive/Desktop/KALYANI/Python/image/img.png")
background_photo = ImageTk.PhotoImage(background_image)
background_label = tk.Label(root, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create a Text widget with a scrollbar for input text
text_input_frame = tk.Frame(root, bg='black', padx=5, pady=5)
text_input_frame.pack(padx=10, pady=10)

scrollbar = tk.Scrollbar(text_input_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

text_input = tk.Text(text_input_frame, wrap="word", width=90, height=30, yscrollcommand=scrollbar.set)
text_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.config(command=text_input.yview)

# Create buttons to trigger the summarization and upload file
btn_upload = tk.Button(root, text="Upload File", bg='lightcyan', command=upload_file, activeforeground='blue', activebackground='grey')
btn_upload.pack(side=tk.TOP, padx=10, pady=10)

btn_extractive = tk.Button(root, text="Extractive Summary", bg='lightblue', command=extractive_summary, activeforeground='blue', activebackground='grey')
btn_extractive.place(relx=0.0, rely=1.0, x=10, y=-10, anchor='sw')

btn_abstractive = tk.Button(root, text="Abstractive Summary", bg='lightblue', command=show_abstractive_summary, activeforeground='blue', activebackground='grey')
btn_abstractive.place(relx=1.0, rely=1.0, x=-10, y=-10, anchor='se')

# Run the application
root.mainloop()
