# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 09:30:11 2021

@author: Nikolaj
"""

import os
import nltk
from nltk.corpus import stopwords
from os.path import isfile, join
import regex as re
from tqdm import tqdm
from community import community_louvain
import networkx as nx
from nltk.probability import FreqDist
import math
import matplotlib.pyplot as plt
from wordcloud import WordCloud
#from textblob import TextBlob
import json


# save the path of the wiki texts
script_dir = os.path.dirname(os.path.abspath(__file__))
folder_dir = "data\\lyrics\\"
path_txts = os.path.join(script_dir, folder_dir)

# Define set of stopwords in english
stopWords = set(stopwords.words('english'))

G = nx.read_gpickle(script_dir + "/graph_building/G.pickle")
# Extract the largest component LC
LC = max(nx.weakly_connected_components(G),key=len)
# Save only the largest component as G
G = nx.DiGraph(G.subgraph(LC))
G_u = G.to_undirected()

"""
partition = community_louvain.best_partition(G_u)
        
with open(script_dir+'\\data\\louvain_partition.txt','w') as x:
        json.dump(partition,x)
"""
with open(os.path.join(script_dir,'data\\louvain_partition.txt'),'r') as x:
        partition = json.load(x)
        
def get_file_names(path_txts = path_txts):
    # get list of txt file names
    filenames = [f for f in os.listdir(path_txts) if isfile(join(path_txts, f))]
    return filenames

def get_file_name_single(path_txt):
    return path_txt.split('\\')[-1]

def pre_process_lyrics(path_txts = path_txts):
    # get list of txt file names
    files = get_file_names(path_txts)
    tk = nltk.WordPunctTokenizer()
    wnl = nltk.WordNetLemmatizer()
    # pattern to remove headers
    pattern_headers = r'=+[\w\s]+=+'
    # pattern to remove new lines
    pattern_nl = r'\\n'
    unfiltered_texts = [open(path_txts+filename,'r',encoding='Latin1').read() for filename in files]
    temp = [re.sub(pattern_headers,'',text) for text in unfiltered_texts]
    temp = [re.sub(pattern_nl,' ',text).lower() for text in temp]
    preprocessed_tokens = [wnl.lemmatize(token) for text in tqdm(temp)
                           for token in tk.tokenize(text)
                           if token.isalpha() and token not in stopWords]
    return preprocessed_tokens


def pre_process_lyrics_single(path_txt):
    tk = nltk.WordPunctTokenizer()
    wnl = nltk.WordNetLemmatizer()
    # pattern to remove headers
    pattern_headers = r'=+[\w\s]+=+'
    # pattern to remove new lines
    pattern_nl = r'\\n'
    unfiltered_text = open(path_txt,'r',encoding='Latin1').read()
    # check if lyrics are english
    #lyrics_language = TextBlob(unfiltered_text[:30]).detect_language()
    #if lyrics_language == 'en':
    temp = re.sub(pattern_nl,' ',re.sub(pattern_headers,'',unfiltered_text)).lower()
    preprocessed_tokens = [wnl.lemmatize(token) for token in tk.tokenize(temp)
                           if token.isalpha() and token not in stopWords]
    return preprocessed_tokens
    

def lyrics_to_artists(path_txts = path_txts):
    filenames = get_file_names(path_txts)
    try:
        artists = [re.sub('__','_',filename).split('_')[-1].split('.txt')[0]
                   for filename in filenames]
    except:
        pass
      
    return artists

def lyrics_to_community_genres (path_txts = path_txts, partition = partition):
    artists_with_lyrics = lyrics_to_artists(path_txts)
    
    communities_louvain = {community:[artist for artist in partition
                                      if (artist in artists_with_lyrics and partition[artist] == community)]
                           for community in set(partition.values())}
    # remove empty communities
    communities_louvain = {community:artists
                           for community,artists in communities_louvain.items()
                           if len(artists) > 0}
    community_to_genres = {community:[genre.capitalize() for artist in artists
                                      for genre in G_u.nodes[artist]['genres']]
                           for community,artists in communities_louvain.items()}
    return community_to_genres


def calculate_tfidf(G_u, partition = partition, path_txts = path_txts, as_TCIDF = False):
    filenames = get_file_names(path_txts)
    artists_with_lyrics = lyrics_to_artists(path_txts)

    communities_louvain = {community:[artist for artist in partition
                                      if (artist in artists_with_lyrics and partition[artist] == community)]
                           for community in set(partition.values())}
    # remove empty communities
    communities_louvain = {community:artists
                           for community,artists in communities_louvain.items()
                           if len(artists) > 0}    
    
    unique_coms = [com for com in communities_louvain.keys()]
    
    text_all_coms = []
    # fill in communities with lyrics, according to their artists
    text_coms = {com:[] for com in unique_coms}
    for i,artist in enumerate(artists_with_lyrics):
        
        try:
            
            # extract lyrics for each artist
            tokenstemp = pre_process_lyrics_single(path_txts+filenames[i])
            text_coms[partition[artist]] += tokenstemp
            text_all_coms += tokenstemp
            """
            with open(script_dir+'\\data\\lyrics_preprocessed\\'+artist+'.txt','w',encoding='Latin1') as x:
                json.dump(tokenstemp,x)
            """
            """
            with open(script_dir+'\\data\\lyrics_preprocessed\\'+artist+'.txt','r',encoding='Latin1') as x:
                tokenstemp = json.load(x)
                text_coms[partition[artist]] += tokenstemp
                text_all_coms += tokenstemp
            """
        except:
            pass
    text_all_coms = nltk.Text(text_all_coms)
    text_all_coms_unique = set(text_all_coms)
    
    # Find term frequency (TF) for each word within each community (local search)
    TF_coms = {com:[] for com in unique_coms}
    for com in unique_coms:
        # normalize
        com_to_tf = {node: count /len(text_coms[com]) for node, count in FreqDist(text_coms[com]).items()}
        # sort the result
        TF_coms[com] += sorted(com_to_tf.items(), key=lambda x: x[1], reverse=True)
        TF_coms[com] = dict(TF_coms[com])
        
    # Find Inverse Document Frequency (IDF) #
    # number of documents N=|D|. D is the communities!
    numdocs = len(unique_coms)
    # number of documents where term t appears (adjusted with +1 t avoid division by zero) |{d \in D: t \in d}|
    numdocs_tinD = {word:len([word for com in unique_coms if word in text_coms[com]])
                    for word in tqdm(text_all_coms_unique)}
    # IDF = log(N/|{d \in D: t \in D}|)
    IDF_coms = {word:math.log(numdocs/(numdocs_tinD[word]+1)) for word in text_all_coms_unique}
    
    # calculate TFIDF for each community and each word in each community
    TFIDF_coms = {com:{word:TF_coms[com][word]*IDF_coms[word] for word in set(text_coms[com])}
                  for com in unique_coms}
    if as_TCIDF:
        TCIDF_coms = get_TCIDF(TFIDF_coms,text_coms)
        return TCIDF_coms
    return TFIDF_coms

def get_TCIDF(TFIDF,text_races):
    return {race:{word:tf_idf*len(text_races[race]) for word,tf_idf in race_dict.items()} for race,race_dict in TFIDF.items()}


def plot_wordclouds(Category,TCIDF,unique_races,mask=None):
    # prepare text by category for the wordcloud function. Add each word a number of times according to their TC-IDF scores
    onestring_races = {race:"" for race in unique_races}
    for race in unique_races:
        for word in TCIDF[race]:
            for _ in range(round(TCIDF[race][word])):
                onestring_races[race] += word+" "
        if len(onestring_races[race]) > 0:
            # Generate wordcloud object to be plotted
            wc = WordCloud(width=1500,height=750,collocations=False,background_color='white',mask=mask).generate(onestring_races[race])
            # plot each wordcloud
            plt.figure(figsize=(10,8))
            plt.imshow(wc,interpolation="bilinear")
            plt.axis('off')
            plt.title(f'{Category}: {race}', fontdict = {'fontsize' : 20})
            plt.show()

def main():
    """
    
    tcidf = calculate_tfidf(G_u, as_TCIDF=True)
    
    with open(script_dir+'\\data\\TCIDF_lyrics.txt','w') as x:
        json.dump(tcidf,x)
    
    tfidf = calculate_tfidf(G_u, as_TCIDF=False)
    with open(script_dir+'\\data\\TFIDF_lyrics.txt','w') as x:
        json.dump(tfidf,x)

    #com_genres = lyrics_to_community_genres()
    #print(com_genres)
    #print(com_genres.keys())
    """
    
    """
    with open(script_dir+'\\data\\TCIDF_lyrics.txt','r') as x:
        tcidf = json.load(x)
    unique_coms = list(tcidf.keys())
    print([len(i) for i in tcidf.values()])
#    plot_wordclouds('Community',tcidf,unique_coms)
    """
if __name__ == '__main__':
    main()