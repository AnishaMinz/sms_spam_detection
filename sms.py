import pandas as pd
import numpy as np
import string
import re
import pickle
from flask import Flask,render_template,request, jsonify
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,accuracy_score


#loading the dataset

df = pd.read_csv(r'D:\sms spam\spam.csv', encoding='latin-1')

df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1) 
df = df.rename(columns={'v1':'label','v2':'text'})  
print(df.head())

#processing and cleaning the dataset
STOPWORDS=set(stopwords.words("english"))
def clean_text(text):
    #convert to lowercase
    text=text.lower()
    #remove special characters
    text=re.sub(r'[^0-9a-zA-Z]', ' ', text)
    #remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    #remove stopwords
    text=' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

print()
'''print("THE CLEANED DATA SET.")
df['clean_text']=df['text'].apply(clean_text)
print(df.head())'''

# Remove urls, mentions and punctuations
def remove_punc(text):
    # Define regex patterns
    url_pattern = r'https?://\S+|www\.\S+'
    mention_pattern = r'@\w+'
    
    # Remove URLs, mentions, punctuations
    text = re.sub(url_pattern, '', text)  
    text = re.sub(mention_pattern, '', text) 
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    
    return text.strip()

# Calculate the length of each message
df['length'] = df['text'].apply(lambda x: len(x.split(' ')))

# Separate the data into ham and spam messages
ham_length = df[df['label'] == 'ham']['length']
spam_length = df[df['label'] == 'spam']['length']
print()
print(f'Max length of ham message: {max(ham_length)}')
print(f'Max length of spam message: {max(spam_length)}')

df['wo_punc'] = df['text'].apply(lambda text: remove_punc(text))
df["wo_stop"] = df["wo_punc"].apply(lambda text: clean_text(text))

# Apply Stemming
stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

df["stemmed_text"] = df["wo_stop"].apply(lambda text: stem_words(text))
print()
print("STEMMED DATA SET.")
print(df.head())

x = df['stemmed_text']
y = df['label']

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Vectorization
vectorizer = TfidfVectorizer()
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)


#modeling
# naive bayes
BayesModel = MultinomialNB()
BayesModel.fit(x_train_vec,y_train)
Bm_pred = BayesModel.predict(x_test_vec)
print("naive bayes")
print("accuracy:",accuracy_score(y_test,Bm_pred))
print("Report:")
print(classification_report(y_test,Bm_pred))
print()

#svm model
S_model = SVC(kernel='linear')
S_model.fit(x_train_vec,y_train)
Sm_pred = S_model.predict(x_test_vec)
print("support vector machine")
print("accuracy:",accuracy_score(y_test,Sm_pred))
print("Report:")
print(classification_report(y_test,Sm_pred))
print()

#Decision tree
Dt_model = DecisionTreeClassifier()
Dt_model.fit(x_train_vec,y_train)
Dm_pred = Dt_model.predict(x_test_vec)
print("Decision Tree")
print("accuracy:",accuracy_score(y_test,Dm_pred))
print("Report:")
print(classification_report(y_test,Dm_pred))
print()

#KNN 
Knn_model = KNeighborsClassifier()
Knn_model.fit(x_train_vec,y_train)
Km_pred = Knn_model.predict(x_test_vec)
print("K-nearest neighbour")
print("accuracy:",accuracy_score(y_test,Km_pred))
print("Report:")
print(classification_report(y_test,Km_pred))
print()

#saving in directory
pickle.dump(S_model,open('model.pkl','wb'))
pickle.dump(vectorizer,open('tfidf_vectorizer.pkl','wb'))


#integration

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))
vector = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
   
    input_message = request.form['msg']

    int_features = vector.transform([input_message])
 
    prediction = model.predict(int_features)
    
    #print(prediction)
    # Output the prediction
    if prediction[0] == 'spam':
        output = "Spam"
    else:
        output = "Not Spam"

    return render_template('index.html', prediction_text='Message is: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
