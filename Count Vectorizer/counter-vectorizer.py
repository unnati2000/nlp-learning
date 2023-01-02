import numpy as numpy
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet 

# nltk.download('wordnet')
# nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

df = pd.read_csv('bbc_text.csv')
print(df.head())

inputs  = df['text']
labels = df['labels']


labels.hist(figsize=(10, 5))

# splits data set into training data and test data

inputs_train, inputs_test, Ytrain, Ytest = train_test_split(
    inputs, labels, random_state=123
)

vectorizer = CountVectorizer()

Xtrain = vectorizer.fit_transform(inputs_train)
Xtest = vectorizer.transform(inputs_test)

# print(Xtrain)

(Xtrain != 0).sum() / numpy.prod(Xtrain.shape)

model = MultinomialNB()
model.fit(Xtrain, Ytrain)

print(model.score(Xtrain, Ytrain))
print(model.score(Xtest, Ytest))

# using stop words

vectorizer = CountVectorizer(stop_words="english")
Xtrain = vectorizer.fit_transform(inputs_train)
Xtest = vectorizer.transform(inputs_test)
model = MultinomialNB()
model.fit(Xtrain, Ytrain)

print('stop word train', model.score(Xtrain, Ytrain))
print('stop word test', model.score(Xtest, Ytest))


# pos tag
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        tokens = word_tokenize(doc)
        words_and_tags = nltk.pos_tag(tokens)
        return [self.wnl.lemmatize(word, pos=get_wordnet_pos(tag))  for word, tag in words_and_tags]


vectorizer = CountVectorizer(tokenizer=LemmaTokenizer())
Xtrain = vectorizer.fit_transform(inputs_train)
Ytrain = vectorizer.transform(inputs_test)
model = MultinomialNB()
model.fit(Xtrain, Ytrain)

print('lemma wala', model.score(Xtrain, Ytrain))
print('lemma wala', model.score(Xtest, Ytest))



print('train score')