#!/usr/bin/python
import pickle
import sys
import json

clf = pickle.load( open('mejor_modelo.pickle') )
X = json.load(open('./data/spam_dev.json'))
predicted = clf.predict(X)
for y in predicted:
	if y == 1:
		print "spam"
	else:
		print "ham"
