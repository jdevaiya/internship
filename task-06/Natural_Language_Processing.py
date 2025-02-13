# Install required libraries (if not already installed)
# pip install nltk spacy

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Input text
text = """
Natural Language Processing (NLP) is a fascinating field of Artificial Intelligence. 
It enables computers to understand, interpret, and generate human language.
"""

# --- NLTK Example ---
# 1. Tokenization
word_tokens = word_tokenize(text)
sentence_tokens = sent_tokenize(text)
print("Word Tokens:", word_tokens)
print("Sentence Tokens:", sentence_tokens)

# 2. Stopword Removal
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in word_tokens if word.lower() not in stop_words]
print("Filtered Words (Without Stopwords):", filtered_words)

# --- spaCy Example ---
# Load spaCy English model
nlp = spacy.load('en_core_web_sm')

# Process text with spaCy
doc = nlp(text)

# 1. Part-of-Speech (POS) Tagging
print("\nPart-of-Speech Tagging:")
for token in doc:
    print(f"{token.text}: {token.pos_}")

# 2. Named Entity Recognition (NER)
print("\nNamed Entities:")
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")

# 3. Lemmatization
print("\nLemmatization:")
lemmatized_words = [token.lemma_ for token in doc]
print(lemmatized_words)
