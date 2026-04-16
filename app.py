from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import re
import nltk
import feedparser
import requests
from bs4 import BeautifulSoup
from youtubesearchpython import VideosSearch
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://topic-modeling-theta.vercel.app",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# REQUEST MODEL
class CompareRequest(BaseModel):
    leaders: list

# Cache
cache = {}

# GOOGLE NEWS
def get_google_news(name):
    url = f"https://news.google.com/rss/search?q={name.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)

    texts = []
    for entry in feed.entries[:10]:
        title = entry.title.lower()
        summary = entry.summary.lower() if 'summary' in entry else ""
        texts.append(title + " " + summary)

    return texts

# BING NEWS
def get_bing_news(name):
    url = f"https://www.bing.com/news/search?q={name.replace(' ', '+')}&format=rss"
    feed = feedparser.parse(url)

    texts = []
    for entry in feed.entries[:10]:
        title = entry.title.lower()
        summary = entry.summary.lower() if 'summary' in entry else ""
        texts.append(title + " " + summary)

    return texts

# DUCKDUCKGO
def get_ddg_news(name):
    url = f"https://duckduckgo.com/html/?q={name.replace(' ', '+')}+news"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")

        results = []
        for a in soup.select(".result__a")[:10]:
            results.append(a.get_text().lower())

        return results
    except:
        return []

# YOUTUBE METADATA
def get_youtube_text(name, limit=5):
    try:
        results = VideosSearch(f"{name} speech", limit=limit).result()

        texts = []
        for v in results['result']:
            title = v.get('title', '')
            desc = ""

            if v.get('descriptionSnippet'):
                desc = " ".join([d['text'] for d in v['descriptionSnippet']])

            texts.append((title + " " + desc).lower())

        return texts
    except:
        return []

# Mering
def get_combined_data(name):
    google = get_google_news(name)
    bing = get_bing_news(name)
    ddg = get_ddg_news(name)
    yt = get_youtube_text(name)

    all_texts = google + bing + ddg + yt

    scored = []

    for text in all_texts:
        score = 0

        if name.lower() in text:
            score += 3

        score += len(text.split()) / 50

        scored.append((score, text))

    scored.sort(reverse=True, key=lambda x: x[0])

    top_texts = [text for score, text in scored[:15]]

    return pd.DataFrame({"speech": top_texts})

# CLEANING
def clean(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text.lower())

    return " ".join([
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words and len(w) > 2
    ])

# TOPIC MODELING
def get_topics(df, n_topics=3):
    if len(df) < 3 or df['clean'].str.strip().eq('').all():
        return []

    vec = CountVectorizer(max_df=0.9, stop_words='english')

    try:
        X = vec.fit_transform(df['clean'])
    except:
        return []

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    words = vec.get_feature_names_out()

    topics = []
    for i, t in enumerate(lda.components_):
        topics.append({
            "topic_id": i,
            "keywords": [words[j] for j in t.argsort()[-5:]]
        })

    return topics

# MAIN PROCESS
def process_leader(name):
    if name in cache:
        return cache[name]

    df = get_combined_data(name)

    if df.empty:
        return {"leader": name, "error": "No data found"}

    df['clean'] = df['speech'].apply(clean)

    if df['clean'].str.strip().eq('').all():
        return {"leader": name, "error": "No valid text"}

    df['sentiment'] = df['speech'].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )

    result = {
        "leader": name,
        "speech_count": len(df),
        "avg_sentiment": df['sentiment'].mean(),
        "topics": get_topics(df)
    }

    cache[name] = result
    return result

# APIs
@app.get("/")
def health():
    return {"status": "Running"}

@app.post("/compare")
def compare(req: CompareRequest):
    results = {}

    for name in req.leaders:
        try:
            results[name] = process_leader(name)
        except Exception as e:
            results[name] = {"error": str(e)}

    return results