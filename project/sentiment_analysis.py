import sys
from os import listdir

from typing import List
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
import scipy as sp
import tqdm
from bs4 import BeautifulSoup
from community import community_louvain
from sklearn.linear_model import LinearRegression
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from scipy.stats import pearsonr
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os
import nltk
from nltk.corpus import stopwords
from os.path import isfile, join
import regex as re
from tqdm import tqdm
import seaborn as sns
#reference to code file

#UNCOMMENT
#from project.spotify.get_spotify_access import refresh_spotify_access
# save the path of the wiki texts
script_dir = os.getcwd()

sys.setrecursionlimit(3000)


#function to plot in and out-degree distribution
def plot_degree_distribution(degrees, title1, title2):
    sns.set()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    max_degree = np.max(degrees)
    min_degree = np.min(degrees)
    vector = np.arange(min_degree, max_degree)
    hist_pois, bin_edges = np.histogram(degrees, bins=vector)
    hist, bin_edges = np.histogram(degrees, bins=vector)
    bin_means = [0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]
    ax1.bar(bin_means, hist, width=bin_edges[1] - bin_edges[0], color='b', edgecolor='blue')
    ax1.set_title(title1)
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("Frequency")

    plt.loglog(bin_means, hist, marker='.', linestyle='None')
    plt.title(title2)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()

#lyrics preprocessing by removing stop words, tokenization and lemmatization
def pre_process_lyrics(folder_dir):
    path_txts = os.path.join(script_dir, folder_dir)
    # get list of txt file names
    files = [f for f in os.listdir(path_txts) if isfile(join(path_txts, f))]
    tk = nltk.WordPunctTokenizer()
    wnl = nltk.WordNetLemmatizer()
    # pattern to remove headers
    pattern_headers = r'=+[\w\s]+=+'
    # pattern to remove new lines
    pattern_nl = r'\\n'
    # Define set of stopwords in english
    stopWords = set(stopwords.words('english'))
    unfiltered_texts = [open(path_txts+filename,'r',encoding='Latin1').read() for filename in files]
    temp = [re.sub(pattern_headers,'',text) for text in unfiltered_texts]
    temp = [re.sub(pattern_nl,' ',text).lower() for text in temp]
    preprocessed_tokens = [wnl.lemmatize(token) for text in tqdm(temp)
                           for token in tk.tokenize(text)
                           if token.isalpha() and token not in stopWords]
    return preprocessed_tokens

# function to scrape lyrics from the Genius API
def scrape_lyrics(album_id):
    sp = refresh_spotify_access()
    album = sp.album(album_id)
    artists = tuple([artist['name'] for artist in album['artists']])
    artist = artists[0]

    album_track_names = [a['name'] for a in album['tracks']['items']]
    artistname2 = str(artist.replace(' ', '-')) if ' ' in artist else str(artist)
    songname2 = [str(songname.replace(' ', '-')) if ' ' in songname else str(songname) for songname in
                 album_track_names[1:3]]
    lyrics_l = []
    for song in songname2:
        page = requests.get('https://genius.com/' + artistname2 + '-' + song + '-' + 'lyrics')

        html = BeautifulSoup(page.text, 'html.parser')
        lyrics1 = html.find("div", class_="lyrics")
        lyrics2 = html.find("div", class_="Lyrics__Container-sc-1ynbvzw-2 jgQsqn")
        if lyrics1:
            lyrics = lyrics1.get_text()
        elif lyrics2:
            lyrics = lyrics2.get_text()
        elif lyrics1 == lyrics2 == None:
            lyrics = None
        # print(lyrics)
        lyrics_l.append(lyrics)
    return lyrics_l

#function to load lyrics locally from dataframe
def load_lyrics(df_albums, index):
    for artist, album_name, album_uri in zip(df_albums["artist"][index:], df_albums["album"][index:], df_albums["album_uri"][index:]):
        try:
            lyrics = scrape_lyrics(album_uri)
            with open("../data/lyrics/" + str(album_name) + "_" + str(artist) + ".txt", "w") as text_file:
                lyr = " ".join([lyr for lyr in lyrics if lyr ])
                if lyr:
                    text_file.write(lyr)
                else:
                    text_file.write("not found")
        except:
                pass

#function which returns the album, artist and lyrics
def get_album_artist(files, path):
    df = pd.DataFrame()
    df["album"] = [f.split("_")[0] for f in files]
    df["artist"] = [f.split("_")[1] for f in files]
    files_f = [path+f for f in files]

    #list_lyrics=[open(f, "r").read() for f in files_f]
    list_lyrics = [open(f, "r").readlines() for f in files_f]
    list_lyrics = ["".join(l) for l in list_lyrics]
    #df["lyrics"] = list_lyrics
    df["lyrics"] = list_lyrics
    return df

# function which transforms list of sentences to lists of processed words (tokenized)
def preprocess(sentences: List[str]) -> List[str]:
    wordnet_lemmatizer = WordNetLemmatizer()
    punctuations = "?:!.,;"
    delimiter = "\n"
    sentences = " ".join(sentences)
    sentence_words = nltk.word_tokenize(sentences)
    words_lemmatized = [wordnet_lemmatizer.lemmatize(w, pos='v').lower()
                        for w in sentence_words
                        if w not in punctuations or delimiter]
    return words_lemmatized

def preprocess(sentences: List[str]) -> List[str]: # List of sentences to lists of processed words
    sentences = " ".join(sentences)
    sentence_words = nltk.word_tokenize(sentences)
    words_lemmatized = [wordnet_lemmatizer.lemmatize(w, pos='v').lower()
                        for w in sentence_words
                        if w not in punctuations]
    return words_lemmatized

#function which computes the average VADER sentiment
def compute_sentiment_VADER(dialogues):
    df = pd.DataFrame()
    try:
        out = np.mean(analyzer.polarity_scores(str(dialogues))['compound'])
    except Exception as e:
        print()
    df["VADER"]=out
    return df

#tokenization function
def tokenize(file):
    tk = nltk.WordPunctTokenizer()
    wnl = nltk.WordNetLemmatizer()
    punctuations = "?:!.,;"
    # pattern to remove headers
    pattern_headers = r'=+[\w\s]+=+'
    # pattern to remove new lines
    pattern_nl = r'\\n'
    # Define set of stopwords in english
    stopWords = set(stopwords.words('english'))
    #filter out pattens, punctuation

    temp = [re.sub(pattern_headers,'',text) for text in file]
    temp = [re.sub(pattern_nl,' ',text).lower() for text in temp]
    #tokenize
    preprocessed_tokens = [wnl.lemmatize(token) for token in tk.tokenize(file)
                           if token.isalpha() and token not in stopWords]
    return preprocessed_tokens

#function to plot the average sentiment per community
def show_bar_plot(df, sentiment_source, ylim=None):

    df.groupby("community")\
            .agg(sentiment=(sentiment_source, np.mean), std=(sentiment_source, np.std))\
            .sort_values("sentiment")\
            .plot.bar(y="sentiment", ylim=ylim)
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.legend(loc='upper left')
    #plt.title(title)
    plt.show()

#function to plot the average sentiment per community
def show_bar_plot_dual(df, sentiment_source, sentiment_source2, ylim=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    df1 = df.groupby("community")\
            .agg(sentiment=(sentiment_source, np.mean), std=(sentiment_source, np.std))\
            .sort_values("sentiment")
    df2 = df.groupby("community") \
        .agg(sentiment=(sentiment_source2, np.mean), std=(sentiment_source2, np.std)) \
        .sort_values("sentiment")

    p1 = ax1.bar(df1.index, df1["sentiment"], ylim=ylim, label=str(sentiment_source))
    p2 = ax2.bar(df2.index, df2["sentiment"], ylim=ylim, label=str(sentiment_source2))

    plt.rcParams['figure.figsize'] = [20, 8]
    plt.legend(loc='upper left')
    #plt.title(title)
    plt.show()

# Function to find 3 most connected nodes within a partition and concatenating them to obtain a partition name
def partition_to_top3_names(G, node_to_partition_id, partition_id):
    nodes_in_partition = [node for node in G.nodes()
                          if node_to_partition_id[node] == partition_id]
    sub_G = G.subgraph(nodes_in_partition)
    node_to_connectivity = [(node, sub_G.degree(node)) for node in sub_G.nodes()]
    top3_nodes = sorted(node_to_connectivity, key=lambda t: t[1], reverse=True)[:3]
    return ", ".join([n for n, _ in top3_nodes])

# Function to find 3 most connected nodes within a partition and concatenating them to obtain a partition name
def partition_to_top3_genre(G, node_to_partition_id, partition_id):
    nodes_in_partition = [node for node in G.nodes()
                          if node_to_partition_id[node] == partition_id]
    sub_G = G.subgraph(nodes_in_partition)
    node_to_connectivity = [(node, sub_G.degree(node)) for node in sub_G.nodes()]
    top3_nodes = sorted(node_to_connectivity, key=lambda t: t[1], reverse=True)[:3]
    return ", ".join([n for n, _ in top3_nodes])

#function which computes the average LabMT sentiment
def compute_sentiment_LabMT(words):
    words_series = pd.Series(words).rename('word')
    sentiments = df_sentiment.merge(words_series, how='inner', on="word")
    return sentiments["happiness_average"].mean()

#scale to (-1, 1) range
def scale(x, out_range=(-1, 1)):
    domain = min(x), max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

#function which converts filename to album and artist
def filename_to_album_artist(f):
    album = f.stem
    return album.split("_")[0], album.split("_")[1]

#function used to return the dominant genre of an artist
def return_genre(community, dict_genre_artist):
    list_artist = community.split(', ')
    list_genre = []
    for a in list_artist:
        try:
            list_genre.append(dict_genre_artist[a])
        except:
            pass
    # remove duplicates
    list_genre = list(set(list_genre))
    l = ", ".join([n for n in list_genre])
    l = l.replace('[', '') \
        .replace("'", "") \
        .replace("[", "") \
        .replace("]", "") \
        .replace('"', '') \
        .replace(",", "") \
        .replace("album rock", " ")

    return l

#function used to return the main record label of an artist
def return_label(community, dict_genre_label):
    list_artist = community.split(', ')
    list_genre = []
    for a in list_artist:
        try:
            list_genre.append(dict_genre_label[a])
        except:
            pass
    # return list_genre
    l = ", ".join([n for n in list_genre])

    return l

#LINEAR REGRESSOR
def linear_prediction(x, y):
    model_pipe = make_pipeline(StandardScaler(),
                               TransformedTargetRegressor(
                                   regressor=LinearRegression(),
                                   func=np.log10,
                                   inverse_func=sp.special.exp10
                               )
                               )

    params = {}

    lin_model = GridSearchCV(model_pipe, param_grid=params)
    CV = KFold(n_splits=5, shuffle=True, random_state=42)


    lin_predicted = cross_val_predict(
        lin_model,x, y, cv=CV, verbose=1)
    return model_pipe, lin_predicted

#LINEAR REGRESSION PLOT: Actual vs Predicted Y value
def plot_linear_prediction(y, lin_predicted):
    corr, p = pearsonr(y, lin_predicted)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y, lin_predicted, edgecolors=(0, 0, 0), s=60)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', c="red", alpha=0.6)
    ax.set_xlabel('Actual LabMT Sentiment', fontsize=14)
    ax.set_ylabel('Predicted LabMT Sentiment', fontsize=14)

    plt.title("Predicted and Actual LabMT Sentiment by the Linear Regressor \nPearson correlation: " + str(
        np.round(corr, 3)) + ", " + "P-value: " + str(np.round(p, 4)), fontsize=16)
    plt.show()

#LINEAR REGRESSION: plot of the regressor's coefficient by importance
def linear_coef_imp(model_pipe, x, y):
        model_pipe, lin_predicted = linear_prediction(x, y)
        params = {}
        lin_model = GridSearchCV(model_pipe, param_grid=params)
        CV = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_model = cross_validate(
            lin_model, x, y, cv=CV,
            return_estimator=True, scoring=['neg_mean_absolute_error'], verbose=1)


        coefs = pd.DataFrame(
            [est.best_estimator_.named_steps['transformedtargetregressor'].regressor_.coef_
             for est in cv_model['estimator']],
            columns=x.columns
        )


        coefs.plot(kind="barh", figsize=(9, 7))
        plt.title("Linear Model Coefficient Importance")
        plt.axvline(x=0, color=".5")
        plt.subplots_adjust(left=0.3)
        plt.show()

#RIDGE REGRESSOR
def ridge_prediction(x, y):
    ridge_model_pipe = make_pipeline(StandardScaler(),
                                     TransformedTargetRegressor(
                                         regressor=Ridge(alpha=1e-10),
                                         func=np.log10,
                                         inverse_func=sp.special.exp10
                                     ))
    model = ridge_model_pipe
    params = {}

    CV = KFold(n_splits=11, shuffle=True, random_state=42)
    ridge = GridSearchCV(ridge_model_pipe,
                         param_grid=params,
                         cv=CV,
                         scoring=['neg_mean_absolute_error'],
                         refit='neg_mean_absolute_error')

    ridge_predicted = cross_val_predict(
        ridge, x, y, cv=CV, verbose=1)
    corr, p = pearsonr(y, ridge_predicted)
    return ridge_model_pipe, ridge_predicted


#RIDGE REGRESSION PLOT: Actual vs Predicted Y value
def plot_ridge_prediction(y, ridge_predicted):
    corr, p = pearsonr(y, ridge_predicted)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y, ridge_predicted, edgecolors=(0, 0, 0), s=60)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', c="red", alpha=0.6)
    ax.set_xlabel('Actual LabMT Sentiment', fontsize=14)
    ax.set_ylabel('Predicted LabMT Sentiment', fontsize=14)

    plt.title("Predicted and Actual LabMT Sentiment by the Ridge Regressor \nPearson correlation: " + str(
        np.round(corr, 3)) + ", " + "P-value: " + str(np.round(p, 4)), fontsize=16)
    plt.show()

#RIDGE REGRESSION: plot of the regressor's coefficient by importance
def ridge_coef_importance(ridge_model_pipe, x, y):
    ridge_model_pipe, _ = ridge_prediction(x, y)
    cv_model_ridge = cross_validate(
        ridge_model_pipe,
        x,
        y,
        cv=KFold(n_splits=5),
        return_estimator=True,
    )

    coefs = pd.DataFrame(
        [
            est.named_steps["transformedtargetregressor"].regressor_.coef_
            for est in cv_model_ridge["estimator"]
        ],
        columns=x.columns,
    )

    coefs.plot(kind="barh", figsize=(9, 7))
    plt.title("Ridge Model Coefficient Importance")
    plt.axvline(x=0, color=".5")
    plt.subplots_adjust(left=0.3)
    plt.show()

#RANDOM FOREST REGRESSOR
def random_forest_prediction(x, y):
    rfr_model_pipe = make_pipeline(StandardScaler(),
                                   TransformedTargetRegressor(
                                       regressor=RandomForestRegressor(),
                                       func=np.log10,
                                       inverse_func=sp.special.exp10
                                   )
                                   )

    params = {}

    CV = KFold(n_splits=7, shuffle=True, random_state=42)
    rfr = GridSearchCV(rfr_model_pipe, param_grid=params, cv=CV, scoring=["r2", 'neg_mean_absolute_error'], refit='r2')
    cv_model = cross_validate(
        rfr, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=42),
        return_estimator=True, scoring=["r2", 'neg_mean_absolute_error'], verbose=1)

    rf_predicted = cross_val_predict(rfr, x, y, cv=CV, verbose=1)

    R2 = sorted(cv_model['test_r2'])
    MAE = sorted(cv_model['test_neg_mean_absolute_error'])
    #print("Mean r2 & std: ", np.mean(R2), "+/-", np.std(R2))
    #print("Mean MAE & std: ", -np.mean(MAE), "+/-", np.std(MAE))
    return cv_model, rf_predicted

#Random Forest PLOT: Actual vs Predicted Y value
def plot_random_forest(y, rf_predicted):
    fig, ax = plt.subplots()
    ax.scatter(y, rf_predicted, edgecolors=(0, 0, 0), s=60)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', c="red", alpha=0.6)
    ax.set_xlabel('Actual LabMT Sentiment', fontsize=14)
    ax.set_ylabel('Predicted LabMT Sentiment', fontsize=14)

    corr, p = pearsonr(y, rf_predicted)
    plt.title("Predicted and Actual LabMT Sentiment by the Random Forest Regressor \nPearson correlation: " + str(
        np.round(corr, 3)) + ", " + "P-value: " + str(np.round(p, 4)), fontsize=16)
    ax.set_title
    plt.show()

#Random Forest REGRESSION: plot of the regressor's coefficient by importance
def rf_coef_imp(x, y, cv_model):
    importances = []

    for i, est in enumerate(cv_model["estimator"]):
        importances.append(est.best_estimator_['transformedtargetregressor'].regressor_.feature_importances_)

    importances = pd.DataFrame(importances, columns=x.columns)
    importances.plot(kind="barh", figsize=(9, 7))
    plt.title("Random Forest Model Coefficient Importance")
    plt.show()

#PLOT X AND Y VALUES USING COLORBAR
def plot_colorbar(df, x, y, ylabel, title, index=False, cb_orientation="vertical", color=None):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    cmap = matplotlib.cm.get_cmap('viridis')


    for i in range(len(df)):
        if index:
            if color:
                norm = matplotlib.colors.Normalize(vmin=df[color].min(), vmax=df[color].max())
                plt.bar(df.index[i],
                        df[y][i],
                        color=cmap(norm(df[color][i])),
                        linestyle='None')
            else:
                norm = matplotlib.colors.Normalize(vmin=df[x].min(), vmax=df[x].max())
                plt.bar(df.index[i],
                        df[y][i],
                        color=cmap(norm(df[x][i])),
                        linestyle='None')
        else:
            if color:
                norm = matplotlib.colors.Normalize(vmin=df[color].min(), vmax=df[color].max())
                plt.bar(df[x][i],
                        df[y][i],
                        color=cmap(norm(df[color][i])),
                        linestyle='None')
            else:
                norm = matplotlib.colors.Normalize(vmin=df[x].min(), vmax=df[x].max())
                plt.bar(df[x][i],
                        df[y][i],
                        color=cmap(norm(df[x][i])),
                        linestyle='None')


    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), label='LabMT', orientation=cb_orientation)

    plt.grid()
    # plt.yscale('log')
    plt.xticks(rotation=90)
    plt.ylabel(ylabel, fontsize=14)
    plt.suptitle(
        title,
        fontsize=16)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    #DO NOT UNCOMMENT
    # df_albums = pd.read_csv("albums.csv")
    # load_lyrics(df_albums, 309)

    # LYRICS_FOLDER = "../data/lyrics.zip/"
    LYRICS_PATH = "data/lyrics/"
    # with zipfile.ZipFile(LYRICS_FOLDER, 'r') as zip_ref:
    #     zip_ref.extractall(LYRICS_PATH)

    files = [f for f in listdir(LYRICS_PATH) if isfile(join(LYRICS_PATH, f))]
    x = pd.read_csv("data/spotify_final.csv")
    df = get_album_artist(files, LYRICS_PATH)

    final_df = pd.read_csv("Spotify/final_vader.csv")

    G = nx.read_gpickle("graph_building/G_sa.pickle")
    LC = max(nx.weakly_connected_components(G), key=len)
    # Save only the largest component as G
    GG = nx.DiGraph(G.subgraph(LC))
    G_u = GG.to_undirected()
    #returns a dic with the names of the artist and the assignedc community
    partition_ids = community_louvain.best_partition(G_u)
    print(final_df.columns)

    keep_artists = {k for k in final_df["artist_x"] if k in partition_ids.keys()}
    final_df= final_df[final_df["artist_x"].isin(keep_artists)]

    # Assign partition name to each partition_id
    partition_id_to_name = {partition_id: partition_to_top3_names(G_u, partition_ids, partition_id)
                            for partition_id in set(partition_ids.values())}
    final_df["community"] = final_df["artist_x"].apply(lambda name: partition_id_to_name[partition_ids[name]])

    # Dataframe for the average sentiment in Vader and LabMT for each community
    community_to_avg_sentiments = [{"community": community,
                                    "avg VADER sentiment": round(df_comm["sentiment_VADER"].mean(), 3)}
                                   for community, df_comm
                                   in final_df.groupby("community")]

    community_to_avg_sentiments = pd.DataFrame(community_to_avg_sentiments)
    community_to_avg_sentiments.to_csv("data/community_to_avg_sentiments.csv")
    show_bar_plot(community_to_avg_sentiments, sentiment_source="avg VADER sentiment")
    print(community_to_avg_sentiments)



    analyzer = SentimentIntensityAnalyzer()
    df_sentiment = pd.read_csv("data/Data_Set_S1.txt", sep="	")
    punctuations = "?:!.,;"

    #LYRICS_PATH = ROOT_DIR / "data" / "lyrics"
    wordnet_lemmatizer = WordNetLemmatizer()

    album_to_preprocessed_words = {
        album: preprocess(lyrics)
        for album, lyrics in zip(final_df["album"], final_df["lyrics"])
        if lyrics != "not found"
    }
    album_to_sentiment = {
        album: compute_sentiment_LabMT(preprocessed_words)
        for album, preprocessed_words in album_to_preprocessed_words.items()
    }

    labMT = pd.read_csv("Spotify/album_to_sentiment.csv")
    final_df2 = pd.merge(final_df, labMT, on="album")

    collaborations = pd.read_csv("Spotify/collaboration_clean.csv")
    final_final_df = pd.merge(final_df2, collaborations, on="energy")

    final_final_df.to_csv("data/final_df.csv", index=False)
    print(final_final_df.columns)






