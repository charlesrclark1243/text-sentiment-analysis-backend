from textblob import Blobber
from textblob.en.sentiments import NaiveBayesAnalyzer

from flask import Flask, request, session
from flask_cors import CORS

import os

# initialize app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('secret_key')

CORS(app, resources=r'/*')

# initialize NaiveBayesAnalyzer for sentiment analysis
analyzer = NaiveBayesAnalyzer()

# sanity check at root of app
@app.route('/', methods=['GET'])
def sanity_check():
    return {
        'response': 'SUCCESS'
    }

# function to analyze the sentiment of the passed text
@app.route('/analyze', methods=['POST'])
def analyze(epsilon=0.2):
    if request.method == 'POST':
        text = request.form['text']
        tb = Blobber(analyzer=analyzer)

        sentiment = tb(text).sentiment

        p_pos, p_neg = sentiment[1], sentiment[2]
        pred = 'Positive' if p_pos > 0.5 + epsilon \
            else 'Negative' if p_neg > 0.5 + epsilon \
            else 'Neutral'
        
        return {
            'prediction': pred,
            'p_pos': p_pos,
            'p_neg': p_neg
        }

if __name__ == '__main__':
    app.secret_key = os.environ.get('SECRET_KEY')
    app.run(debug=True)