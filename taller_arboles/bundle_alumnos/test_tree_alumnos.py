from tree_aa import *
import pylab

# Armamos dataset tennis 
X=np.array([["Sol","Calor","Alta","Debil"],
["Sol","Calor","Alta","Fuerte"],
["Nublado","Calor","Alta","Debil"],
["Lluvia","Templado","Alta","Debil"],
["Lluvia","Frio","Normal","Debil"],
["Lluvia","Frio","Normal","Fuerte"],
["Nublado","Frio","Normal","Fuerte"],
["Sol","Templado","Alta","Debil"],
["Sol","Frio","Normal","Debil"],
["Lluvia","Templado","Normal","Debil"],
["Sol","Templado","Normal","Fuerte"],
["Nublado","Templado","Alta","Fuerte"],
["Nublado","Calor","Normal","Debil"],
["Lluvia","Templado","Alta","Fuerte"]])
y=np.array('No No Si Si Si No Si No Si Si Si Si Si No'.split())
attrs_names=np.array('Cielo Temperatura Humedad Viento'.split())


# Creo un DecisionTree
dt= DecisionTree(information_gain)

# Entrenamos
dt.fit(X,y,attrs_names)

# Pruebo una prediccion arbitraria
prediccion_ejemplo= dt.predict(np.array([["Lluvia","Frio","Normal","Fuerte"]]))
print 'Predigo',prediccion_ejemplo

# Ploteo el arbol para mirarlo
fig= dt.plot_graph()
pylab.show()

