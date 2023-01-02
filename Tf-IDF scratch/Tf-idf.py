import pandas as pd
import numpy as np


from nltk import word_tokenize

df = pd.read_csv('bbc_text.csv')

idx = 0
word2idx = {}
tokenized_docs = []

for doc in df['text']:
    words = word_tokenize(doc)
    doc_as_int = []
    
    for word in words:
        if word not in word2idx:
            word2idx[word] = idx
            idx = idx+1
        doc_as_int.append(word2idx[word])
    tokenized_docs.append(doc_as_int)

# print(tokenized_docs)

idx2word = {v:k for k,v in word2idx.items()}

# print(idx2word)

N = len(df['text'])

V = len(word2idx)

tf = np.zeros((N,V))

for i, doc_as_int in enumerate(tokenized_docs):
    for j in doc_as_int:
        tf[i, j]= tf[i,j]+1


document_freq = np.sum(tf>0, axis=0)
idf = np.log(N / document_freq)

tf_idf = tf*idf

np.random.seed(193)

i = np.random.choice(N)
row = df.iloc[i]

print('Label: ', row['labels'])
print('Text', row['text'].split("\n", 1)[0])

scores = tf_idf[i]

indices = (-scores).argsort()


for j in indices[:5]:
    print(idx2word[j])





