"""
News Analyzer - Optimized for Hugging Face Spaces
"""

import streamlit as st
import feedparser
from transformers import pipeline
from gtts import gTTS
from datetime import datetime
from deep_translator import GoogleTranslator
import matplotlib.pyplot as plt
import base64
import io 
import os
os.system("pip install httpx==0.24.1 httpcore==0.17.3 --quiet")

# Configure app
st.set_page_config(
    page_title="News Analyzer",
    layout="wide"
)

# Color scheme
COLOR_MAP = {
    "positive": "#4CAF50",  # Green
    "negative": "#F44336",  # Red
    "neutral": "#FF9800"    # Orange
}

# Initialize models (cached properly for Hugging Face)
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english",  # Smaller model for faster inference
        framework="pt"
    )

def fetch_news(company):
    """Fetches news from Google News RSS"""
    try:
        url = f"https://news.google.com/rss/search?q={company}+when:1d&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        return [
            {
                "title": entry.title,
                "summary": entry.title[:200],  # Truncate for model input
                "source": entry.source.title if hasattr(entry, 'source') else "Unknown",
                "url": entry.link,
                "date": datetime.now().strftime("%Y-%m-%d")
            }
            for entry in feed.entries[:5]  # Top 5 articles
        ]
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

def create_sentiment_chart(sentiment_counts):
    """Creates matplotlib chart with thread-safe approach"""
    fig, ax = plt.subplots()
    sentiments = ["Positive", "Negative", "Neutral"]
    colors = [COLOR_MAP["positive"], COLOR_MAP["negative"], COLOR_MAP["neutral"]]
    ax.bar(sentiments, sentiment_counts.values(), color=colors)
    ax.set_title("Sentiment Distribution")
    ax.set_ylabel("Number of Articles")
    plt.close()  # Prevents memory leaks
    return fig


def text_to_hindi_audio(text):
    try:
        # Translation
        hindi_text = GoogleTranslator(source='en', target='hi').translate(text)
        
        # Audio generation
        audio_bytes = io.BytesIO()
        tts = gTTS(text=hindi_text, lang='hi', slow=False)
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
        
    except Exception as e:
        st.error(f"Audio generation failed: {e}")
        return None

def main():
    st.title("📰 News Sentiment Analyzer")
    
    model = load_model()
    company = st.text_input("Enter company name:", "Tesla").strip()
    
    if st.button("Analyze News") or company:
        with st.spinner("Analyzing news..."):
            articles = fetch_news(company)
            
            if not articles:
                st.error("No news found. Try: Tesla, Apple, Microsoft")
                return
            
            # Analyze sentiment
            for article in articles:
                try:
                    result = model(article["summary"])[0]
                    article["sentiment"] = result["label"].lower()
                except:
                    article["sentiment"] = "neutral"
            
            # Display articles
            st.subheader(f"📰 Latest News About {company}")
            for article in articles:
                with st.expander(f"{article['title']} ({article['source']})"):
                    st.write(f"**Date:** {article['date']}")
                    st.markdown(f"[Read Full Article ↗]({article['url']})")
                    st.markdown(
                        f"**Sentiment:** <span style='color:{COLOR_MAP[article['sentiment']]};'>"
                        f"{article['sentiment'].title()}</span>",
                        unsafe_allow_html=True
                    )
            
            # Create sentiment counts BEFORE using them
            sentiment_counts = {
                "positive": sum(1 for a in articles if a["sentiment"] == "positive"),
                "negative": sum(1 for a in articles if a["sentiment"] == "negative"),
                "neutral": sum(1 for a in articles if a["sentiment"] == "neutral")
            }
            
            st.pyplot(create_sentiment_chart(sentiment_counts))
            
            if st.button("Generate Hindi Audio Summary"):
                summary_text = (
                    f"{company} समाचार सारांश. "
                    f"सकारात्मक: {sentiment_counts['positive']}, "
                    f"नकारात्मक: {sentiment_counts['negative']}, "
                    f"तटस्थ: {sentiment_counts['neutral']}."
                )
    
                audio_data = text_to_hindi_audio(summary_text)
                if audio_data:
                    st.audio(audio_data, format="audio/mp3")
                else:
                    st.warning("ऑडियो जनरेशन विफल हुआ | (Audio generation failed)")

if __name__ == "__main__":
    main()
