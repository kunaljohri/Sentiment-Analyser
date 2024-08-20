# Sentiment-Analyser

>>Summary

The project involves developing a web application using Flask for sentiment analysis, employing a logistic regression model trained on a sample dataset to classify sentiments as positive, neutral, or negative. Users can interact with the application by entering text directly into a form for sentiment analysis. The application features a user-friendly interface that processes and displays sentiment predictions based on the input text.

>>How to use

Ensure you have Python installed. You can download it from python.org.

1.Install Flask and scikit-learn using pip:
        ->pip install flask scikit-learn pandas(write this on your computer's terminal)
        
2.Create a new directory sentiment-analyser:
        ->mkdir sentiment-analysis-app(to make the directory)
        ->cd sentiment-analysis-app(to navigate to it)
        
3.Inside the sentiment-analysis-app directory:
        ->download Twitter_Data.csv ,model.pkl,vectorizer.pkl,sentiment_analyser.py
        ->Also make a 'templates' folder in the directory and download index.html inside it
        
4.Open a terminal and navigate to the sentiment-analysis-app directory.Flask will start a development server, usually accessible at http://127.0.0.1:5000/.

5.Open Your Web Browser:
        ->Go to http://127.0.0.1:5000/ to see the sentiment analysis web app.        
