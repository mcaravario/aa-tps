#!/usr/bin/python
# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016
import json
import pickle
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import TfidfVectorizer


# Leo los mails (poner los paths correctos).)
print "Cargando data frame..."
ham_train = json.load(open('./data/ham_dev.json'))
spam_train = json.load(open('./data/spam_dev.json'))

#ham_train = ham_train[:len(ham_train)/2]
#spam_train = spam_train[:len(spam_train)/2]


X = ham_train+spam_train
y = [0 for _ in range(len(ham_train))]+[1 for _ in range(len(spam_train))]

clf = Pipeline([
	('extraction',			TfidfVectorizer(max_features=1000, stop_words="english", lowercase=False)),
	('selection',    		SelectKBest(k=100)),
	('classifier', 			RandomForestClassifier(max_features="sqrt",n_estimators=1,criterion="entropy"))])

print "Entrenando"
clf.fit(X,y)
print "Guardando"

fout = open('mejor_modelo.pickle','w')
pickle.dump(clf,fout)
fout.close()
