import re
from pathlib import Path
from typing import List

import nltk
import numpy as np
import pandas as pd
from nltk import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from project.util import ROOT_DIR, DATA_PATH

analyzer = SentimentIntensityAnalyzer()
df_sentiment = pd.read_csv(DATA_PATH / "Data_Set_S1.txt", sep="	")
punctuations = "?:!.,;"

LYRICS_PATH = ROOT_DIR / "data" / "lyrics"
wordnet_lemmatizer = WordNetLemmatizer()


# Function which computes the average sentiment in dialogue sentences
def compute_sentiment_VADER(dialogues):
    sentiment = analyzer.polarity_scores(str(dialogues))['compound']
    return sentiment


def compute_sentiment_LabMT(words):
    words_series = pd.Series(words).rename('word')
    sentiments = df_sentiment.merge(words_series, how='inner', on="word")
    return sentiments["happiness_average"].mean()


def preprocess(lyrics: str) -> List[str]:  # List of sentences to lists of processed words
    sentence_words = nltk.word_tokenize(lyrics)
    words_lemmatized = [wordnet_lemmatizer.lemmatize(w, pos='v').lower()
                        for w in sentence_words
                        if w not in punctuations]
    return words_lemmatized


def scale(x, out_range=(-1, 1)):
    domain = min(x), max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def filename_to_album_artist(f):
    album = f.stem
    return album.split("_")[0], album.split("_")[1]


def main():
    album_to_raw_text = {
        filename_to_album_artist(f): f.open().read() for f in LYRICS_PATH.glob("*.txt")
    }
    album_to_preprocessed_words = {
        album: preprocess(lyrics)
        for album, lyrics in album_to_raw_text.items()
        if lyrics != "not found"
    }
    album_to_sentiment = {
        album: compute_sentiment_LabMT(preprocessed_words)
        for album, preprocessed_words in album_to_preprocessed_words.items()
    }
    df = pd.DataFrame([{"album": album[0], "artist": album[1], "sentiment": sentiment}
                       for album, sentiment in album_to_sentiment.items()])
    df["sentiment"] = scale(df["sentiment"].values, out_range=(0, 1))
    df.to_csv(DATA_PATH / "album_to_sentiment.csv", index=False)


if __name__ == '__main__':
    main()
