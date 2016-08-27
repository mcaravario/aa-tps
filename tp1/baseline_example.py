# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016

import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, f1_score
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter

# Leo los mails (poner los paths correctos).
# ham_raw= json.load(open('./data/ham_txt.json'))
# spam_raw= json.load(open('./data/spam_txt.json'))
ham_txt= json.load(open('./data/ham_txt.json'))
spam_txt= json.load(open('./data/spam_txt.json'))

# # Imprimo un mail de ham y spam como muestra.
# print ham_txt[0]
# print "------------------------------------------------------"
# print spam_txt[0]
# print "------------------------------------------------------"

# Armo un dataset de Pandas 
# http://pandas.pydata.org/

# ham_txt = ham_raw[:int(0.5*len(ham_raw))]
# ham_test = ham_raw[int(0.5*len(ham_raw)):]
# spam_txt = spam_raw[:int(0.5*len(spam_raw))]
# spam_test = spam_raw[int(0.5*len(spam_raw)):]
df = pd.DataFrame(ham_txt+spam_txt, columns=['text'])
df['class'] = ['ham' for _ in range(len(ham_txt))]+['spam' for _ in range(len(spam_txt))]

# vectorizer = CountVectorizer(max_features=100)
# X = vectorizer.fit_transform(df['text'].values)
# print vectorizer.get_feature_names()

# words = ['3d', 'br', 'font']
# def count_3d(txt): return txt.count('3d')
# def count_br(txt): return txt.count('br')
# def count_font(txt): return txt.count('font')

# df['3d'] = map(count_3d, df.text)
# df['br'] = map(count_br, df.text)
# df['font'] = map(count_font, df.text)	
# X = df[['3d', 'br', 'font']].values
# # Extraigo dos atributos simples: 
# # 1) Longitud del mail.
# df['len'] = map(len, df.text)

# # 2) Cantidad de espacios en el mail.
# def count_spaces(txt): return txt.count(" ")
# df['count_spaces'] = map(count_spaces, df.text)

# # 3) Tokenizo y lemmatizo
# stoplist = stopwords.words('english')
# def preprocess(sentence):
# 	lemmatizer = WordNetLemmatizer()
# 	return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence)]

# def get_features(text, setting):
#     if setting=='bow':
#         return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}
#     else:
#         return {word: True for word in preprocess(text) if not word in stoplist}

# all_features = [(get_features(email, 'bow'), label) for (email, label) in all_emails]

# df[word] = map()
# print all_features[0][0]

# # Preparo data para clasificar
# X = df[['len', 'count_spaces']].values
y = df['class']

# Elijo mi clasificador.
clf = DecisionTreeClassifier()

pipeline = Pipeline([
	('count_vectorizer',	CountVectorizer(max_features=100)),
	('classifier', 			DecisionTreeClassifier()) ])

k_fold = KFold(n=len(df), n_folds=10, shuffle=True)
scores = []
confusion = np.array([[0, 0], [0, 0]])
for train_indices, test_indices in k_fold:
    train_text = df.iloc[train_indices]['text'].values
    train_y = df.iloc[train_indices]['class'].values.astype(str)

    test_text = df.iloc[test_indices]['text'].values
    test_y = df.iloc[test_indices]['class'].values.astype(str)

    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)

    confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions, pos_label='spam')
    scores.append(score)

print('Total emails classified:', len(df))
print('Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)

# # Ejecuto el clasificador entrenando con un esquema de cross validation
# # de 10 folds.
# res = cross_val_score(clf, X, y, cv=2, scoring='f1')

# print res
# print np.mean(res), np.std(res)
# # salida: 0.783040309346 0.0068052434174  (o similar)
