# 📱 SMS Spam Detection Web Application

This is a web-based application that detects whether a given SMS is **Spam** or **Not Spam** using Natural Language Processing (NLP) and Machine Learning techniques.

## 🔍 Features

- Takes user input of SMS text
- Preprocesses the input (tokenization, stopword removal, etc.)
- Uses a trained ML model to classify the message as **Spam** or **Ham (Not Spam)**
- Interactive and user-friendly web interface

## 🛠️ Technologies Used

- **Python**
- **Scikit-learn**
- **NLTK**
- **TfidfVectorizer**
- **Flask**
- **HTML/CSS**

## 📁 Project Structure

sms-spam-detection/
├── app.py
├── model.pkl
├── tfidf_vectorizer.pkl
├── spam.csv
├── requirements.txt
├── templates/
│   └── index.html
└── static/
    └── style.css

