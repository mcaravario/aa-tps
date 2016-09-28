#! /usr/bin/env python
# -*- coding: utf-8 -*-
import metricas as metricas

## Primera parte (matriz de confusión)

y_actual = ["spam"] * 10 + ["no-spam"] * 978 + ["spam"] * 2
y_pred = ["spam"] * 2 + ["no-spam"] * 900 + ["spam"] * 20 + ["no-spam"] * 68
tp, tn, fp, fn = metricas.confusion_matrix(y_actual=y_actual, y_predicted=y_pred, positive_label="spam")

print "\nTest 1"
print "(tp, tn, fp, fn) = ", (tp, tn, fp, fn)
assert((tp, tn, fp, fn) == (2, 958, 20, 10))
print "OK\n"

### Segunda parte (métricas)

accuracy = round(metricas.accuracy_score(tp, tn, fp, fn), 2)
print accuracy
precision = round(metricas.precision_score(tp, tn, fp, fn), 2)
print precision
recall = round(metricas.recall_score(tp, tn, fp, fn), 2)
print recall
f1 = round(metricas.f1_score(tp, tn, fp, fn), 2)
print f1

print "Test 2"
print "(accuracy, precision, recall, f1) = ", (accuracy, precision, recall, f1)
assert((accuracy, precision, recall, f1) == (0.97, 0.09, 0.17, 0.12))
print "OK\n"
