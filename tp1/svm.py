#!/usr/bin/python
# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016
import json
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_selection import SelectKBest, f_classif, chi2


# Leo los mails (poner los paths correctos).)
ham_txt = json.load(open('./data/ham_dev.json'))
spam_txt = json.load(open('./data/spam_dev.json'))

# Me quedo con la mitad del dataset para entrenar
ham_txt_train = ham_txt[:len(ham_txt)/2]
spam_txt_train = spam_txt[:len(spam_txt)/2]

print "Cargando data frame..."

X = ham_txt_train+spam_txt_train
y = [0 for _ in range(len(ham_txt_train))]+[1 for _ in range(len(spam_txt_train))]


pipeline = Pipeline([
	('tfidf_vectorizer',	TfidfVectorizer(max_features=1000, stop_words="english", lowercase=False)),
    ("selection",           SelectKBest(k=100)),
	('classifier', 			SVC(decision_function_shape="ovr")) ])

print "Creo pipeline"

# Configuracion de Grid search
param_grid = 	{"classifier__kernel": ["linear", "poly", "rbf"],
				 "classifier__C": (1.0, 1.5, 2, 2.5, 3, 3.5) 
}


grid_search = GridSearchCV(pipeline, n_jobs=1, pre_dispatch=1,scoring="f1", cv=10, param_grid=param_grid, verbose=10)
grid_search.fit(X, y)
print "Termine de entrenar"
parameters = grid_search.best_params_
print parameters

best_estimator = grid_search.best_estimator_

# Me quedo con la mitad del dataset para testear
ham_txt_test = ham_txt[len(ham_txt)/2:]
spam_txt_test = spam_txt[len(ham_txt)/2:]

predictions = best_estimator.predict(spam_txt_test+ham_txt_test)

# Creo vector con las clases correctas
test_y = len(spam_txt_test)*[1] + len(ham_txt_test)*[0]

confusion = confusion_matrix(test_y, predictions)

score = f1_score(test_y, predictions, pos_label=1)


print 'Total emails classified:', len(test_y)
print 'Score:', score
print 'Confusion matrix:'
print confusion
