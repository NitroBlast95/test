from flask import Flask, request, jsonify, render_template
import json
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import random
import pickle

# Initialize Flask app
app = Flask(__name__)

# Priority levels
PRIORITY_LEVELS = ["Low", "Medium Low", "Medium High", "High"]

# Load dataset and model
nltk.download('stopwords')
nltk.download('punkt')

# Load Dataset
with open('/Users/srinandham/Downloads/analyzed_dataset_diff_stats_with_impact.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df.dropna(subset=['body', 'impact_score'], inplace=True)

# Convert impact score to categories
def impact_category(score):
    if score <= 25:
        return 'Low'
    elif score <= 50:
        return 'Medium Low'
    elif score <= 75:
        return 'Medium High'
    else:
        return 'High'

df['impact_category'] = df['impact_score'].apply(impact_category)

# Text Preprocessing
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df['cleaned_text'] = df['body'].apply(preprocess_text)

# Vectorize Text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['impact_category']

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model and vectorizer
with open('impact_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Load trained model and vectorizer
with open('impact_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Predict Function
def predict_impact_category(comment):
    cleaned_comment = preprocess_text(comment)
    vectorized_comment = vectorizer.transform([cleaned_comment])
    prediction = model.predict(vectorized_comment)
    return prediction[0]

@app.route('/')
def index():
    return render_template('datathon.html')

@app.route('/commit')
def commit():
    return render_template('commit.html')

@app.route('/pull')
def pull():
    return render_template('pull.html')

@app.route('/process_message', methods=['POST'])
def process_message():
    data = request.json
    message = data.get("message", "").strip()
    
    if not message:
        return jsonify({"error": "No message provided"}), 400  
    
    # Predict impact category based on message
    impact_category = predict_impact_category(message)
    
    # Return the message and predicted impact category
    return jsonify({"message": message, "impact_category": impact_category})

if __name__ == '__main__':
    app.run(debug=True)
