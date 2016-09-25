#!/usr/bin/python
# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
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
	('extraction',			TfidfVectorizer(max_features=1000, stop_words="english", lowercase=False)),
	#("features", 			FeatureUnion([("pca", PCA(n_components=50)), ("univ_select", SelectKBest(k=50))])),
	('selection', 			SelectKBest(k=100)),
	('classifier', 			DecisionTreeClassifier()) ])

print "Creo pipeline"

# Configuracion de Grid search
param_grid = {}
# param_grid = 	{"classifier__max_depth": [3, 5, 10],
# 				"classifier__max_features": ["sqrt", 10, 50, 100, None],
#               	"classifier__criterion": ["gini", "entropy"],
#               	'selection__score_func': [f_classif, chi2],
#               	"classifier__min_samples_leaf": [1, 5, 10]}
# grid_search = GridSearchCV(pipeline, n_jobs=1, pre_dispatch=1,scoring="f1", cv=2, param_grid=param_grid, verbose=10)
# grid_search.fit(X, y)
print "Termine de entrenar"
# parameters = grid_search.best_params_
# print parameters

# best_estimator = grid_search.best_estimator_
best_estimator = pipeline
best_estimator.fit(X,y)

# Me quedo con la mitad del dataset para testear
ham_txt_test = ham_txt[len(ham_txt)/2:]
spam_txt_test = spam_txt[len(ham_txt)/2:]

predictions = best_estimator.predict(spam_txt_test+ham_txt_test)
pvalues = best_estimator.named_steps["selection"].pvalues_
pvalues = sorted(pvalues)
scores = -np.log10(pvalues[-1:-50:-1])
scores /= scores.max()
# X_indices = np.arange(X.shape[-1])
plt.bar(np.arange(len(scores))-.45,scores)
plt.title("Seleccion de atributos")
plt.xlabel("Numero de atributo")
plt.ylabel("Score")
plt.show()


# Creo vector con las clases correctas
test_y = len(spam_txt_test)*[1] + len(ham_txt_test)*[0]

confusion = confusion_matrix(test_y, predictions)

score = f1_score(test_y, predictions, pos_label=1)

print 'Total emails classified:', len(test_y)
print 'Score:', score
print 'Confusion matrix:'
print confusion
