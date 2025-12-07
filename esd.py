import pandas as pd

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from flask import Flask, request

df = pd.read_csv("spam.csv", encoding="latin1")

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['processed_text'] = df['v2'].apply(preprocess)

df.to_csv("preprocessed_spam.csv", index=False)

processed_text = df['processed_text']

vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(processed_text)

labels = df['v1']

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))

print(classification_report(y_test, predictions))

def predict_email(text):
    text = preprocess(text)
    vector = vectorizer.transform([text])
    return model.predict(vector)[0]

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def classify():
    email = request.json['email']
    return {"label": predict_email(email)}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)





