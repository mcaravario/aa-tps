import numpy as np
from collections import Counter
import pylab
from itertools import groupby

def log2(v): return np.log(v)/np.log(2)

def entropy(y): 
    y.sort()
    pk = [len(list(group))/len(y) for key, group in groupby(y)]
    return np.stats.entropy(pk)

def information_gain(column,y):
    hs = entropy(y)
    suma = 0
    atrs_val = [len(list(group)) for key, group in groupby(column)]
    for a in atrs_value:
        sv = [y_i for (c_i, y_i) in zip(column,y) if c_i==a]
        suma+=(len(sv)/len(y))*entropy(sv)
    return hs - suma

class DecisionTree():
	""" Inicializo el arbol, tengo que tomar la funcion que decida que 
	atributo elegir """
	def __init__(self, attr_select_func):
		self.attr_select_func = attr_select_func
		self.predicted_class=None
		self.attr=""
		self.sub_trees={}
		self._fit_called=False
	
	"""Funcion para entrenar el arbol"""
	def fit(self,X,y,attrs_names):
		self.attrs_names=attrs_names
		self._fit_called=True
		self._fit_y= y
		
		# Si no tengo mas atributos, guardo la clase con mayor cantidad
		if X.shape[1]==0:
			self.predicted_class = Counter(y).most_common(1)[0][0]
			return None
			
		# Si "y" tiene una sola clase, termine seteo la clase para prediccion
		if all(y[0]==y):
			self.predicted_class= y[0]
			return None
		
		# No termine tengo que seguir fiteando recursivamente
		
		# Obtengo el "mejor" atributo, usando la funcion
		index_best_attr= np.argmax([self.attr_select_func(X[:,ci],y) for ci in range(X.shape[1])])
		
		# Armo las matrices para entrenamiento que van a recibir los 
		# arboles hijos
		different_values = sorted(set(X[:,index_best_attr]))
		X_without_column =np.delete(X,index_best_attr,axis=1)
		attrs_names_without_column= np.delete(attrs_names,index_best_attr,axis=0)
		self.attr=attrs_names[index_best_attr]
		
		
		self.sub_trees={}
				
		for dv in different_values:
			# Armo las submatrices para cada arbol hijo
			mask= X[:,index_best_attr]==dv 
			newX = X_without_column[mask,:]
			newy = y[mask]
			self.sub_trees[dv] = DecisionTree(self.attr_select_func)
			self.sub_trees[dv].fit(newX,newy,attrs_names_without_column)
	
	def predict(self,Xtest):
		assert(self._fit_called)
		res =[]
		dt=self
		
		for irow in range(Xtest.shape[0]):
			row= Xtest[irow,:]
			# Si entreno con una muestra no total, esta es una forma de
			# mitigar el problema
			try: res.append(self._predict(row))
			except: res.append( Counter(self._fit_y).most_common(1)[0][0] )
			
		return res
	
	def _predict(self,row):
		if self.attr=='': return self.predicted_class
		
		index_node_attr= np.where(self.attrs_names==self.attr)[0][0]
		return self.sub_trees[row[index_node_attr]]._predict(np.delete(row,index_node_attr))	
	
			
	""" Para plotear el arbol """
	def plot_graph(self):
		# Devuelve las posiciones para que se ploteee en forma de arbol
		def _tree_layout(xcenter,ycenter,dt):
			res=[( id(dt),[xcenter,ycenter])]
			i=0
			for k,sdt in dt.sub_trees.iteritems():
				res+=_tree_layout((xcenter+ np.linspace(-.5/float(abs(ycenter-1)**2),0.5/float(abs(ycenter-1)**2),len(dt.sub_trees)))[i],ycenter-1,sdt)
				i+=1
			return res	
		
		# Bindea object id con label para plotting
		def _create_dict_id_name(dt):
			res =[]
			if dt.attr=="": res.append((id(dt),dt.predicted_class))
			else: res.append((id(dt),dt.attr))
			for k,v in dt.sub_trees.iteritems(): 
				res+=_create_dict_id_name(v)
			return res	
			
		# Obteng los edges		
		def _get_edges_to_plot(dt):
			edges=[]
			if dt.attr=='': return []
			for k,v in dt.sub_trees.iteritems(): 
				edges.append((id(dt),id(v),k))
				if not v.attr=='': edges+= _get_edges_to_plot(v)
			return edges
		
		dict_node_id_to_label = dict(_create_dict_id_name(self))
		edges=_get_edges_to_plot(self)

		pos =dict(_tree_layout(0,0,self))

		fig =pylab.figure(figsize=[ 14.7,   6. ])
		# Plot nodes
		for k,coord in pos.iteritems():
			pylab.scatter(coord[0],coord[1],s=500,alpha=0.3)
			pylab.text(coord[0],coord[1],dict_node_id_to_label[k],horizontalalignment='center', verticalalignment='center',fontsize=14)
		# Plot edges and labels
		for src,dst,label in edges:
			src_point = pos[src]	
			dst_point = pos[dst]
			pylab.plot([src_point[0],dst_point[0]],[src_point[1],dst_point[1]],c='black',alpha=0.5)
			pylab.text(src_point[0]+(dst_point[0]-src_point[0])/2.,src_point[1]+(dst_point[1]-src_point[1])/2.,label,
			horizontalalignment='center', verticalalignment='center',fontsize=12)
		pylab.xticks([])
		pylab.yticks([])
		fig.tight_layout()
		return fig
		
 
	
