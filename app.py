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
from wordcloud import WordCloud
import io
import base64
import os
import matplotlib
from collections import Counter, defaultdict
import itertools

matplotlib.use("Agg")
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

# FastAPI setup
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

def setup_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except:
        nltk.download('stopwords')

    try:
        nltk.data.find('corpora/wordnet')
    except:
        nltk.download('wordnet')

setup_nltk()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# REQUEST MODEL
class CompareRequest(BaseModel):
    leaders: list[str]
    n_topics: int = 3

# DATA SOURCES

def get_google_news(name):
    url = f"https://news.google.com/rss/search?q={name.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)

    return [
        (entry.title + " " + entry.get("summary", "")).lower()
        for entry in feed.entries[:10]
    ]

def get_bing_news(name):
    url = f"https://www.bing.com/news/search?q={name.replace(' ', '+')}&format=rss"
    feed = feedparser.parse(url)

    return [
        (entry.title + " " + entry.get("summary", "")).lower()
        for entry in feed.entries[:10]
    ]

def get_ddg_news(name):
    try:
        url = f"https://duckduckgo.com/html/?q={name.replace(' ', '+')}+news"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")

        return [a.get_text().lower() for a in soup.select(".result__a")[:10]]
    except:
        return []

def get_youtube_text(name, limit=5):
    try:
        results = VideosSearch(f"{name} speech", limit=limit).result()
        texts = []

        for v in results['result']:
            title = v.get('title', '')
            desc = " ".join([d['text'] for d in v.get('descriptionSnippet', [])])
            texts.append((title + " " + desc).lower())

        return texts
    except:
        return []

# PROCESSING
def get_combined_data(name):
    texts = (
        get_google_news(name)
        + get_bing_news(name)
        + get_ddg_news(name)
        + get_youtube_text(name)
    )

    scored = []
    for text in texts:
        score = (3 if name.lower() in text else 0) + len(text.split()) / 50
        scored.append((score, text))

    scored.sort(reverse=True)
    return pd.DataFrame({"speech": [t for _, t in scored[:15]]})

def clean(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text.lower())

    return " ".join([
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words and len(w) > 2
    ])

def get_topics(df, n_topics=3):
    if df.empty:
        return []

    vec = CountVectorizer(max_df=0.9, stop_words='english')

    try:
        X = vec.fit_transform(df['clean'])
    except:
        return []

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    words = vec.get_feature_names_out()

    return [
        {
            "topic_id": i,
            "keywords": [words[j] for j in t.argsort()[-5:]]
        }
        for i, t in enumerate(lda.components_)
    ]

def generate_wordcloud(texts):
    if not texts:
        return None

    text = " ".join(texts)

    wc = WordCloud(
        width=800,
        height=400,
        background_color='white',
        font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ).generate(text)

    img = io.BytesIO()
    wc.to_image().save(img, format='PNG')
    img.seek(0)

    return base64.b64encode(img.getvalue()).decode()

def generate_knowledge_graph(texts, top_n=20):
    if not texts:
        return {"nodes": [], "edges": []}

    words = []
    for t in texts:
        words.extend(t.split())

    common = [w for w, _ in Counter(words).most_common(top_n)]
    nodes = [{"id": w, "label": w} for w in common]

    # co-occurrence edges
    edge_weights = defaultdict(int)

    for t in texts:
        tokens = set(t.split()) & set(common)
        for a, b in itertools.combinations(tokens, 2):
            edge_weights[(a, b)] += 1

    edges = [
        {"source": a, "target": b, "weight": w}
        for (a, b), w in edge_weights.items()
        if w > 1
    ]

    return {"nodes": nodes, "edges": edges}

def process_leader(name, n_topics):
    df = get_combined_data(name)

    if df.empty:
        return {"leader": name, "error": "No data found"}

    df['clean'] = df['speech'].apply(clean)

    df['sentiment'] = df['speech'].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )

    return {
        "leader": name,
        "speech_count": len(df),
        "avg_sentiment": df['sentiment'].mean(),
        "topics": get_topics(df, n_topics),
        "wordcloud": generate_wordcloud(df['clean'].tolist()),
        "graph": generate_wordcloud(df['clean'].tolist()),
    }

#  APIs
@app.get("/")
def health():
    return {"status": "Running"}

@app.post("/compare")
def compare(req: CompareRequest):
    return {
        name: process_leader(name, req.n_topics)
        for name in req.leaders
    }