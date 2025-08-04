# News-Sentiment Dashboard  
_VADER vs DistilBERT headline sentiment, in real time_

![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-orange)
![License](https://img.shields.io/badge/license-MIT-blue)

An interactive web app that tracks daily mood swings in news headlines for any keyword or topic, then compares a fast **lexicon-based** engine (VADER) with a context-aware **Transformer** model (DistilBERT).  
Built as the term project for **Boston University – MET CS 688 Web Analytics & Mining**.

# Features
* **Live data** – pull headlines straight from *The Guardian* Open-Platform API  
* **Dual engines** – choose VADER, DistilBERT, or compare both side-by-side  
* **Visual insights** – time-series line charts (with 7-day smoothing) and pie-chart breakdowns  
* **Keyword & date filters** – zoom in on any period or sub-topic instantly  
* **Download-ready** – export sentiment-scored headlines as CSV for offline analysis  

## Quick start (local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GUARDIAN_KEY="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
streamlit run app.py
```
Automatically open http://localhost:8501 in your browser.

# Clean BBC sample
A 5000-row, already-cleaned BBC headline sample is included at
data/bbc_clean_5k.csv (≈1 MB).
If you ever re-clean the full dataset (data/bbc_news.csv), run
python data/clean.py --in data/bbc_news.csv --out data/bbc_clean_5k.csv

# File layout
news-sentiment-dashboard/
├─ app.py
├─ requirements.txt
├─ .gitignore
└─ data/
   ├─ clean_bbc_5k.csv
   └─ clean.py
License
MIT – see LICENSE.

Yitian Qian, “News Sentiment Trend Dashboard – CS688 Term Project”, Boston University, 2025.
