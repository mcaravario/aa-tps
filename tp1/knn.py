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
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter

# Leo los mails (poner los paths correctos).)
ham_txt= json.load(open('./data/ham_dev.json'))
spam_txt= json.load(open('./data/spam_dev.json'))

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

# Elijo mi clasificador.
clf = KNeighborsClassifier(n_neighbors=5, weights="uniform")

pipeline = Pipeline([
	('count_vectorizer',	CountVectorizer(max_features=100)),
	('classifier', 			clf) ])

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
