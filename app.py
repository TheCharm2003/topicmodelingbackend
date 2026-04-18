from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import re
import nltk
import feedparser
import requests
from bs4 import BeautifulSoup
from youtubesearchpython import VideosSearch, ChannelsSearch
from youtube_transcript_api import YouTubeTranscriptApi
import trafilatura
from concurrent.futures import ThreadPoolExecutor
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
import gc

matplotlib.use("Agg")
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

# FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://topic-modeling-theta.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def setup_nltk():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except:
        nltk.download('stopwords')
        nltk.download('wordnet')

setup_nltk()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

_DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_PATH = _DEJAVU if os.path.exists(_DEJAVU) else None

# REQUEST MODEL
class CompareRequest(BaseModel):
    leaders: list[str]
    n_topics: int = 3

# DATA SOURCES

def fetch_article_body(url, max_words=500):
    if not url:
        return ""
    try:
        res = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5,
            allow_redirects=True,
        )
        if res.status_code != 200:
            return ""
        text = trafilatura.extract(res.text) or ""
        return " ".join(text.split()[:max_words])
    except Exception:
        return ""

def fetch_bodies_parallel(urls, max_workers=2):
    if not urls:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(fetch_article_body, urls))

def get_google_news(name):
    url = f"https://news.google.com/rss/search?q={name.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    entries = feed.entries[:8]

    bodies = fetch_bodies_parallel([e.get("link", "") for e in entries])

    return [
        (e.title + " " + e.get("summary", "") + " " + body).lower()
        for e, body in zip(entries, bodies)
    ]

def get_bing_news(name):
    url = f"https://www.bing.com/news/search?q={name.replace(' ', '+')}&format=rss"
    feed = feedparser.parse(url)
    entries = feed.entries[:8]

    bodies = fetch_bodies_parallel([e.get("link", "") for e in entries])

    return [
        (e.title + " " + e.get("summary", "") + " " + body).lower()
        for e, body in zip(entries, bodies)
    ]

def get_ddg_news(name):
    try:
        url = f"https://duckduckgo.com/html/?q={name.replace(' ', '+')}+news"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")

        return [a.get_text().lower() for a in soup.select(".result__a")[:10]]
    except:
        return []

def fetch_transcript(video_id, max_words=2000):
    if not video_id:
        return ""
    try:
        entries = YouTubeTranscriptApi.get_transcript(
            video_id, languages=["en", "hi"]
        )
        words = " ".join(e["text"] for e in entries).split()
        return " ".join(words[:max_words])
    except Exception:
        return ""

def get_youtube_text(name, limit=5):
    try:
        results = VideosSearch(f"{name} speech", limit=limit).result()
        texts = []

        for v in results['result']:
            title = v.get('title', '')
            desc = " ".join([d['text'] for d in v.get('descriptionSnippet', [])])
            transcript = fetch_transcript(v.get('id', ''))
            texts.append((title + " " + desc + " " + transcript).lower())

        return texts
    except:
        return []

def get_official_channel_videos(name, limit=6):
    try:
        channels = ChannelsSearch(name, limit=1).result()
        if not channels.get("result"):
            return []
        channel_id = channels["result"][0]["id"]
    except Exception:
        return []

    feed = feedparser.parse(
        f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    )

    texts = []
    for entry in feed.entries[:limit]:
        video_id = getattr(entry, "yt_videoid", None)
        if not video_id and entry.get("link"):
            video_id = entry.link.split("v=")[-1].split("&")[0]

        title = entry.get("title", "")
        summary = entry.get("summary", "")
        transcript = fetch_transcript(video_id) if video_id else ""
        texts.append((title + " " + summary + " " + transcript).lower())
    return texts

def get_wikiquote(name):
    try:
        url = f"https://en.wikiquote.org/wiki/{name.replace(' ', '_')}"
        res = requests.get(
            url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6
        )
        if res.status_code != 200:
            return []

        soup = BeautifulSoup(res.text, "html.parser")
        content = soup.select_one(".mw-parser-output")
        if not content:
            return []

        quotes = []
        for li in content.select("ul > li"):
            parts = []
            for child in li.children:
                if getattr(child, "name", None) in ("ul", "ol"):
                    break
                parts.append(
                    child.get_text(" ", strip=True)
                    if hasattr(child, "get_text")
                    else str(child).strip()
                )
            text = " ".join(p for p in parts if p)
            text = re.sub(r"\[\d+\]", "", text).strip()

            if 5 < len(text.split()) < 80:
                quotes.append(text.lower())

        return quotes[:20]
    except Exception:
        return []

# PROCESSING
def get_combined_data(name):
    texts = (
        get_google_news(name)
        + get_bing_news(name)
        + get_ddg_news(name)
        + get_youtube_text(name)
        + get_official_channel_videos(name)
        + get_wikiquote(name)
    )

    scored = []
    for text in texts:
        score = (3 if name.lower() in text else 0) + len(text.split()) / 50
        scored.append((score, text))

    scored.sort(reverse=True)
    return pd.DataFrame({"speech": [t for _, t in scored[:20]]})

def clean(text, extra_stopwords=frozenset()):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text.lower())

    out = []
    for w in text.split():
        if w in stop_words or len(w) <= 2:
            continue
        w = lemmatizer.lemmatize(w)
        if w in extra_stopwords:
            continue
        out.append(w)
    return " ".join(out)

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
        font_path=FONT_PATH,
    ).generate(text)

    img = io.BytesIO()
    wc.to_image().save(img, format='PNG')
    img.seek(0)

    return base64.b64encode(img.getvalue()).decode()

def generate_knowledge_graph(texts, top_n=20):
    # Nodes = the top_n most frequent words across all snippets.
    # Edges = co-occurrence: two words appearing in the same snippet.
    # Weight = how many snippets they both appear in.
    if not texts:
        return {"nodes": [], "edges": []}

    words = []
    for t in texts:
        words.extend(t.split())

    common = [w for w, _ in Counter(words).most_common(top_n)]
    nodes = [{"id": w, "label": w} for w in common]

    edge_weights = defaultdict(int)

    for t in texts:
        tokens = set(t.split()) & set(common)
        for a, b in itertools.combinations(sorted(tokens), 2):
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

    name_tokens = set()
    for word in name.lower().split():
        if len(word) > 2:
            name_tokens.add(word)
            name_tokens.add(lemmatizer.lemmatize(word))
    extra_stop = name_tokens | {
        "said", "say", "says", "told", "speech", "speeches",
        "video", "watch", "youtube", "news",
    }

    df['clean'] = df['speech'].apply(lambda t: clean(t, extra_stop))

    df['sentiment'] = df['speech'].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )

    return {
        "leader": name,
        "speech_count": len(df),
        "avg_sentiment": df['sentiment'].mean(),
        "topics": get_topics(df, n_topics),
        "wordcloud": generate_wordcloud(df['clean'].tolist()),
        "graph": generate_knowledge_graph(df['clean'].tolist()),
    }

#  APIs
@app.get("/")
def health():
    return {"status": "Running"}

@app.post("/compare")
def compare(req: CompareRequest):
    results = {}
    for name in req.leaders:
        results[name] = process_leader(name, req.n_topics)
        gc.collect()   # free memory between leaders
    return results
