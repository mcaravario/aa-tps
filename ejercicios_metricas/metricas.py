#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

def confusion_matrix(y_actual, y_predicted, positive_label):
	# Construye una matriz de confusión (binaria)
	# y_actual es la secuencia de etiquetas reales
	# y_predicted es la secuencia de etiquetas predichas por el clasificador
	# positive_label indica cuál es la etiqueta considerada positiva.
	n = len(y_actual)

	tp = len([i for i in range(0, n) if y_actual[i] == positive_label and y_predicted[i] == positive_label])    # verdaderos positivos
	tn = len([i for i in range(0, n) if y_actual[i] != positive_label and y_predicted[i] != positive_label]) 	# verdaderos negativos
	fp = len([i for i in range(0, n) if y_actual[i] != positive_label and y_predicted[i] == positive_label])	# falsos positivos
	fn = len([i for i in range(0, n) if y_actual[i] == positive_label and y_predicted[i] != positive_label])	# falsos negativos

	return tp, tn, fp, fn


def accuracy_score(tp, tn, fp, fn):
	return (tp + tn) / (tp + tn + fp + fn)


def precision_score(tp, tn, fp, fn):
	return tp / (tp + fp)


def recall_score(tp, tn, fp, fn):
	return tp / (tp + fn)


def f_beta_score(tp, tn, fp, fn, beta):
	prec = precision_score(tp, tn, fp, fn)
	recl = recall_score(tp, tn, fp, fn)
	return (1+beta*beta)*(prec * recl)/((beta*beta*prec) + recl)


def f1_score(tp, tn, fp, fn):
	return f_beta_score(tp, tn, fp, fn, beta=1)


def probas_to_labels(y_predicted_probas, positive_label, negative_label, threshold = 0.5):
	return [positive_label if p >= threshold else negative_label for p in y_predicted_probas]


def summary(y_actual, y_pred, positive_label):
	tp, tn, fp, fn = confusion_matrix(y_actual=y_actual, y_predicted=y_pred, positive_label=positive_label)
	print "C-matrix: \n", np.array([["tp: {}".format(tp), "fn: {}".format(fn)],["fp: {}".format(fp), "tn: {}".format(tn)]])

	accuracy = round(accuracy_score(tp, tn, fp, fn), 2)
	precision = round(precision_score(tp, tn, fp, fn), 2)
	recall = round(recall_score(tp, tn, fp, fn), 2)
	f1 = round(f1_score(tp, tn, fp, fn), 2)

	print "(accuracy, precision, recall, f1) = ", (accuracy, precision, recall, f1)
	print
