from textblob import Blobber
from textblob.en.sentiments import NaiveBayesAnalyzer

from flask import Flask, request
from flask_cors import CORS

import nltk
import os

# download wordnet corpus
nltk.download('movie_reviews')

# initialize app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('secret_key')

CORS(app, resources=r'/*')

# initialize NaiveBayesAnalyzer for sentiment analysis
analyzer = NaiveBayesAnalyzer()

# sanity check at root of app
@app.route('/test', methods=['GET'])
def sanity_check():
    return {
        'response': 'SUCCESS'
    }

# function to analyze the sentiment of the passed text
@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        # print(request.form)

        text = request.form['text']
        tb = Blobber(analyzer=analyzer)

        sentiment = tb(text).sentiment

        p_pos, p_neg = sentiment[1], sentiment[2]
        pred = 'Positive' if p_pos > 0.5 \
            else 'Negative'
        
        return {
            'prediction': pred,
            'p_pos': round(p_pos, 4),
            'p_neg': round(p_neg, 4)
        }

if __name__ == '__main__':
    app.secret_key = os.environ.get('SECRET_KEY')
    # app.secret_key = os.urandom(12)
    app.run(debug=True)