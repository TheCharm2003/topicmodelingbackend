from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import re
import nltk
from youtube_transcript_api import YouTubeTranscriptApi
from youtubesearchpython import VideosSearch
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Setup
app = FastAPI()

nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# REQUEST MODELS
class LeaderRequest(BaseModel):
    name: str

class CompareRequest(BaseModel):
    leaders: list


# -------- CACHE (IMPORTANT FOR SPEED) --------
cache = {}


# -------- SCRAPING --------
def get_transcript(video_id):
    try:
        t = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([x['text'] for x in t])
        return text if len(text.split()) > 100 else None
    except:
        return None


def search_videos(name, limit=5):
    results = VideosSearch(f"{name} full speech", limit=limit).result()
    return [v['id'] for v in results['result']]


def scrape(name):
    rows = []
    for vid in search_videos(name):
        text = get_transcript(vid)
        if text:
            rows.append({"politician": name, "speech": text})
    return pd.DataFrame(rows)


# -------- CLEANING --------
def clean(text):
    text = re.sub(r'[^a-zA-Z ]', '', text.lower())
    return " ".join([
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words
    ])


# -------- TOPIC MODELING --------
def get_topics(df, n_topics=3):
    vec = CountVectorizer(max_df=0.9, stop_words='english')
    X = vec.fit_transform(df['clean'])

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


# -------- PIPELINE --------
def process_leader(name):
    if name in cache:
        return cache[name]

    df = scrape(name)

    if df.empty:
        return {"error": "No data found"}

    df['clean'] = df['speech'].apply(clean)
    df['sentiment'] = df['speech'].apply(lambda x: TextBlob(x).sentiment.polarity)

    result = {
        "leader": name,
        "speech_count": len(df),
        "avg_sentiment": df['sentiment'].mean(),
        "topics": get_topics(df)
    }

    cache[name] = result
    return result


# -------- ROUTES --------

# Health check
@app.get("/")
def health():
    return {"status": "alive"}


# Single leader analysis
@app.post("/analyze")
def analyze(req: LeaderRequest):
    return process_leader(req.name)


# Topics only
@app.post("/topics")
def topics(req: LeaderRequest):
    data = process_leader(req.name)
    return {"leader": req.name, "topics": data.get("topics", [])}


# Compare leaders
@app.post("/compare")
def compare(req: CompareRequest):
    results = {}

    for name in req.leaders:
        results[name] = process_leader(name)

    return results