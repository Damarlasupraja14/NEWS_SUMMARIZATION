# NEWS_SUMMARIZATION
# News Sentiment Analyzer

## Overview
The **News Sentiment Analyzer** fetches news articles from Google News RSS, analyzes their sentiment using a transformer-based model, and provides visual insights. It also supports audio summaries in Hindi using text-to-speech.

## Features
- Fetches news articles from Google News RSS based on a company name.
- Analyzes sentiment using a transformer-based model.
- Displays a sentiment distribution chart.
- Provides an audio summary in Hindi.
- User-friendly Streamlit interface.

## Installation

### Prerequisites
Ensure you have **Python 3.10+** installed.

### Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/news-analyzer.git
   cd news-analyzer
   ```
2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # For macOS/Linux
   venv\Scripts\activate  # For Windows
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
Run the Streamlit application:
```sh
streamlit run app.py
```

## Dependencies
- `streamlit`
- `feedparser`
- `transformers`
- `gtts`
- `deep-translator`
- `matplotlib`
- `pandas`

## Troubleshooting
**Error: `AttributeError: module 'httpcore' has no attribute 'SyncHTTPTransport'`**
- This issue occurs due to an incompatible version of `httpcore`.
- Solution:
  ```sh
  pip uninstall httpcore
  pip install httpcore==0.15.0
  ```

## License
This project is licensed under the MIT License.

