from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import string
import pickle

app = Flask(__name__)



df = pd.read_csv("Twitter_Data.csv")
df['clean_text']=df['clean_text'].astype("string")
df=df.dropna()

def preprocess_text(text):
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    
    text = text.lower()
    
    
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    filtered_words = [word for word in tokens if word not in stop_words]
    
    return ' '.join(filtered_words)

df['cleaned_review'] = df['clean_text'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['category'], test_size=0.1, random_state=42)


vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = LogisticRegression(multi_class='multinomial')
model.fit(X_train_vec, y_train)





with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('vectorizer.pkl', 'wb') as vect_file:
    pickle.dump(vectorizer, vect_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        
       
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('vectorizer.pkl', 'rb') as vect_file:
            vectorizer = pickle.load(vect_file)

        
        text_tfidf = vectorizer.transform([text])
        
        
        prediction = model.predict(text_tfidf)[0]
        
      
        if prediction == 1:
            sentiment = "Positive"
        elif prediction == 0:
            sentiment = "Neutral"
        else:
            sentiment = "Negative"
        
        return render_template('index.html', prediction_text=f'Sentiment: {sentiment}')

if __name__ == "__main__":
    app.run(debug=True)
