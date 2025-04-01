from flask import Flask, request, jsonify
import feedparser
from transformers import pipeline

app = Flask(__name__)

# Load sentiment analysis model
sentiment_model = pipeline("sentiment-analysis")

# Default route - Welcome message
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the News Sentiment Analysis API!"})

# News Sentiment Analysis Route
@app.route('/analyze', methods=['GET'])
def analyze_news():
    company = request.args.get('company', 'Tesla')
    url = f"https://news.google.com/rss/search?q={company}+when:1d&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)

    articles = []
    for entry in feed.entries[:10]:  # Get Top 10 news
        result = sentiment_model(entry.title[:512])[0]
        articles.append({
            "title": entry.title,
            "sentiment": result["label"]
        })

    return jsonify(articles)

if __name__ == '__main__':
    app.run(debug=True)
