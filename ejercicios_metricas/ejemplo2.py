#! /usr/bin/env python
# -*- coding: utf-8 -*-
import metricas as metricas

y_actual = ["perro"] * 18 + ["gato"] * 980 + ["perro"] * 5
y_pred_1 = ["gato"] * 980 + ["perro"] * 20 + ["gato"] * 3
y_pred_2 = ["perro"] * 40 + ["gato"] * 900 + ["perro"] * 60 + ["gato"] * 3

print "Clasificador A"
metricas.summary(y_actual, y_pred_1, positive_label="gato")

print "Clasificador B"
metricas.summary(y_actual, y_pred_2, positive_label="gato")

##
## COMPLETAR
print "Clasificador A (perro)"
print "completar"

print "Clasificador B (perro)"
print "completar"
##

exit(0)  # Comentar para parte 2.

y_actual = ["perro"] * 100 + ["gato"] * 900 + ["perro"] * 80
y_pred =   ["perro"] * 80 + ["gato"] * 800 + ["perro"] * 200

tns_gato = []
f1s_gato = []
f1s_perro = []
f1s_avg = []

for i in range(0, 10000, 100):
	y_actual_2 = y_actual + ["perro"] * i
	y_pred_2 = y_pred + ["perro"] * i

	tp1, tn1, fp1, fn1 = metricas.confusion_matrix(y_actual=y_actual_2, y_predicted=y_pred_2, positive_label="gato")
	tp2, tn2, fp2, fn2 = metricas.confusion_matrix(y_actual=y_actual_2, y_predicted=y_pred_2, positive_label="perro")

	f1_gato = metricas.f1_score(tp1, tn1, fp1, fn1)
	f1_perro = metricas.f1_score(tp2, tn2, fp2, fn2)
	f1_avg = (f1_gato + f1_perro) / 2

	tns_gato.append(tn1)
	f1s_gato.append(f1_gato)
	f1s_perro.append(f1_perro)
	f1s_avg.append(f1_avg)

import matplotlib.pyplot as plt
plt.plot(tns_gato, f1s_gato, "*-", label="f1_gato")
plt.plot(tns_gato, f1s_perro, "o-", label="f1_perro")
plt.plot(tns_gato, f1s_avg, "x-", label="f1_avg")
plt.xlabel("True Negatives (Gato)")
plt.ylabel("F1 score")
plt.ylim([0,1])
plt.legend()
plt.show()
