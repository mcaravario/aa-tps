#!/usr/bin/python
# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import TfidfVectorizer


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
	('extraction',			TfidfVectorizer(max_features=1000, stop_words="english", lowercase=False)),
	('selection',    		SelectKBest(k=100)),
	('classifier', 			RandomForestClassifier())])

print "Creo pipeline"

# Configuracion de Grid search
param_grid = 	{"classifier__n_estimators": [2, 5, 7, 10, 15],
                "classifier__criterion": ["gini", "entropy"],
				"classifier__max_features": ["sqrt", None]}

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
