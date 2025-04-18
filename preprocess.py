import re
from stopwords import tl_stopwords, en_stopwords

SENTIMENT_MAP = {
    "Positive": 1,
    "Neutral": 0,
    "Negative": -1
}

def clean_text(text: str):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) 
    words = text.split()
    filtered = []
    stopwords = en_stopwords.union(tl_stopwords);
    for word in words:
        if word not in stopwords:
            filtered.append(word)
    return ' '.join(filtered)

def preprocess_data(df):
    df = df.drop_duplicates()
    df = df.dropna(subset=['sentiment', 'text'])

    df["text"] = df["text"].fillna("").apply(clean_text)
    df["sentiment"] = df["sentiment"].map(SENTIMENT_MAP)

    return df
