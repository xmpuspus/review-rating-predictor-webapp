import pandas as pd
import joblib

import nltk
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string

import gensim
from gensim import corpora
from gensim.models import CoherenceModel

import streamlit as st

# Header
### Set Title
st.title("Review Rating Predictor")
st.write("""From the input review, predict rating.""")

# Text input
text = st.text_area("Input review text", 'Input text here.')

# Function to Clean text
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

# Actually clean text
text_process = clean(text).split()

# Load dictionary 
loaded_dictionary = corpora.Dictionary.load('models/dictionary.sav') 

# Convert list of words to document term array
doc_term = loaded_dictionary.doc2bow(text_process)

# Load LDA model
lda_load_model = gensim.models.ldamodel.LdaModel.load("models/lda_model.sav")

# Get document topic probabilities
lists = lda_load_model.get_document_topics(doc_term, minimum_probability=0.0)

# Convert to array/list
document_topic =  [i[1] for i in lists]

# Load regression model
regression_model = joblib.load('models/lr_model.sav')

# predict on output document topic
prediction = regression_model.predict([document_topic])[0]

# Predicted rating
st.write(f'Predicted rating is {float("{0:.2f}".format(prediction))}.')