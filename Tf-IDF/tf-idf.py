import pandas as pd
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

df2 = pd.read_csv('movies2.csv')

# function to join a string of keywords and genres
# because TfIdf accepts data in that format


def genres_and_keywords_to_string(row):
    genre = json.loads(row['genres'])
    genre = ' '.join(''.join(j['name'].split()) for j in genre)

    keywords = json.loads(row['keywords'])
    keywords = ' '.join(''.join(j['name'].split()) for j in keywords)
    return "%s %s " % (genre, keywords)


# creates a new string for each movie and string is the new columns
df2['string'] = df2.apply(genres_and_keywords_to_string, axis=1)


# TfIdf instance, max cols = 2000
tfidf = TfidfVectorizer(max_features=2000)

# inserts data to the matrix from the string column and keeps track of the most frequent terms
X = tfidf.fit_transform(df2['string'])


# generating a map from movie title -> index
movie2idx = pd.Series(df2.index, index = df2['title'])


idx = movie2idx['Scream 3']

query = X[idx]
query = query.toarray()

scores = cosine_similarity(query, X)

scores = scores.flatten()

scores = (-scores).argsort()

recommended_idx = (-scores).argsort()[1:6]

df2['title'].iloc[recommended_idx]

def recommended(title):
    # getting the index of the title from the data set
    idx = movie2idx[title]


    if type(idx) == pd.Series:
        idx = idx.iloc[0]

    # new row ka data from query
    query = X[idx]

    # getting the cosine similarity
    scores = cosine_similarity(query, X)

    # array ko 1D banana
    scores = scores.flatten()

    # array ko sort karna
    recommended_idx = (-scores).argsort()[1:6]
    
    return df2['title'].iloc[recommended_idx]

print(recommended('Avatar'))
