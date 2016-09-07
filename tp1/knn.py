#!/usr/bin/python
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter


# Leo los mails (poner los paths correctos).)
ham_txt = json.load(open('./data/ham_dev.json'))
spam_txt = json.load(open('./data/spam_dev.json'))

# Me quedo con la mitad del dataset para entrenar
ham_txt_train = ham_txt[:len(ham_txt)/2]
spam_txt_train = spam_txt[:len(spam_txt)/2]

# Me quedo con la mitad del dataset para testear
ham_txt_test = ham_txt[len(ham_txt)/2 + 1:]
spam_txt_test = spam_txt[len(ham_txt)/2 + 1:]

################ Forma de plotear los graficos de KNN ##########################
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# from sklearn import neighbors, datasets
#
# n_neighbors = 15
#
# # import some data to play with
# iris = datasets.load_iris()
# X = iris.data[:, :2]  # we only take the first two features. We could
#                       # avoid this ugly slicing by using a two-dim dataset
# y = iris.target
#
# h = .02  # step size in the mesh
#
# # Create color maps
# cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
# cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
#
# for weights in ['uniform', 'distance']:
#     # we create an instance of Neighbours Classifier and fit the data.
#     clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
#     clf.fit(X, y)
#
#     # Plot the decision boundary. For that, we will assign a color to each
#     # point in the mesh [x_min, m_max]x[y_min, y_max].
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#
#     # Put the result into a color plot
#     Z = Z.reshape(xx.shape)
#     plt.figure()
#     plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
#
#     # Plot also the training points
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
#     plt.xlim(xx.min(), xx.max())
#     plt.ylim(yy.min(), yy.max())
#     plt.title("3-Class classification (k = %i, weights = '%s')"
#               % (n_neighbors, weights))
#
# plt.show()

################################################################################

# Armo un dataset de Pandas
# http://pandas.pydata.org/
df = pd.DataFrame(ham_txt+spam_txt, columns=['text'])
df['class'] = ['ham' for _ in range(len(ham_txt))]+['spam' for _ in range(len(spam_txt))]

# Preparo data para clasificar
y = df['class']
X = df['text']

pipeline = Pipeline([
	('count_vectorizer',	CountVectorizer(max_features=100)),
	('classifier', 			KNeighborsClassifier()) ])

print "Creo pipeline"

# Configuracion de Grid search
param_grid = {"n_neighbors": [1, 3, 5, 7, 10],
              "weights": ["uniform", "distance"]}
grid_search = GridSearchCV(pipeline, n_jobs=3, scoring="f1", cv=10, param_grid=param_grid, verbose=5)
grid_search.fit(X, y)
print "Termine de entrenar"
parameters = grid_search.best_params_
print parameters

best_estimator = grid_search.best_estimator_

predictions = best_estimator.predict(spam_txt_test+ham_txt_test)

# Creo vector con las clases correctas
test_y = len(spam_txt_test)*["spam"] + len(ham_txt_test)*["ham"]

confusion = confusion_matrix(test_y, predictions)

score = f1_score(test_y, predictions, pos_label='spam')


print('Total emails classified:', len(df))
print('Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)
