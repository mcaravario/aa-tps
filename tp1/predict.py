import pickle
import sys
import json

args = sys.argv
if len(args) != 2:
	print "Falta filename de conjunto a testear"
	sys.exit()	
test_set = args[1]
clf = pickle.load(open("mejor_modelo.pickle"))
X = json.load(open('test.json'))
predicted = clf.predict(X)
for y in predicted:
	if y == 1:
		print "spam"
	else:
		print "ham"