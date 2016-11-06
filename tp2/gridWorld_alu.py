import numpy as np
import random
from collections import defaultdict
import pylab


""" Clase que implementa un gridWorld"""
class gridWorld():
	def __init__(self,width=3, height=2,goals=None,positive=100,select_action='random',alpha=0.9,gamma=0.9):
		"""Seteo los parametros"""
		self.height=height
		self.width=width
		self.positive=positive
		self.select_action=select_action
		self.alpha=alpha
		self.gamma=gamma
		self.learning_step=0
		
		"""Represento Q como una lista de filas de celdas de Q
		Es decir, en self.Q[0][1] esta la celda que corresponde al estado
		[0,1]. La celdas son diccionarios de acciones posibles y sus 
		valores. 
		
		Por ejemplo, self.Q[0][0] podria ser: {'right':0.8, 'down':0.3'}
		o si el estado [0,2] es terminal, self.Q[0][2]: {'stay':100}
		
		Esta definido con un defaultdict para facilidad, podria 
		inicialiazarse explicitamente
		"""
		
		self.Q = [ [ defaultdict(lambda :random.random()) for _ in 
		range(self.width)] for _ in range(self.height)]
		
		if goals==None:	self.goals=[[0,self.width-1]]
		else: self.goals=list(goals)
		
			
		
	""" Devuelve las posibles acciones a hacer teniendo en cuenta los
	bordes y los terminales """
	def possibleActions(self,state):
		row,col = state 
		res=[]
		if row>0: res.append('up')
		if row<self.height-1: res.append('down')
		# TO DO ...
		
		
	""" Dado un estado y una accion devuelve un estado nuevo despues de
	hacer hecho la accion"""
	def move(self,state,action):
		new_state= list(state)
		if action=='up': new_state[0]-=1
		if action=='down': new_state[0]+=1
		# TO DO ...
		
		
		
	""" Me dice si state es terminal"""	
	def isTerminal(self,state):
		return state in self.goals
		
		
	""" Funcion recompenza, tengo una reward positiva si desde state
	usando action llego a un estado terminal """	
	def reward(self,state, action):
		if self.isTerminal(self.move(state,action)): 
			return self.positive
		return 0
		
	
	""" Funcion que aprende, implementa Qlearning (diapos teorica)"""
	def learn(self,state):
		self.learning_step+=1
		
		# Repito hasta que state sea terminal
		while not self.isTerminal(state) :
			
			# 1) Listo las posibles acciones que puedo hacer teniendo
			# encuenta el estado de donde estoy
			
			# 2) Elijo alguna accion dentro de las posibles (segun algun criterio)
			
			# 3) Calculo el nuevo valor de Q(s,a)
			
	
			# 4) Actualizo s
			state = new_state
	
		
	
	""" Funcion para plotear la Q"""
	def draw(self):
		def ifExceptReturnNan(dic,k):
			try: return dict(dic)[k]
			except: return np.nan


		matrix_right= np.array([[ ifExceptReturnNan(cel,'right') for cel in row] for row in self.Q])
		matrix_left= np.array([[ ifExceptReturnNan(cel,'left') for cel in row] for row in self.Q])
		matrix_up= np.array([[ ifExceptReturnNan(cel,'up') for cel in row] for row in self.Q])
		matrix_down= np.array([[ ifExceptReturnNan(cel,'down') for cel in row] for row in self.Q])
		matrix_stay= np.array([[ ifExceptReturnNan(cel,'stay') for cel in row] for row in self.Q])

		fig = pylab.figure(figsize=2*np.array([self.width,self.height]))
		for i in range(matrix_right.shape[0]):
			for j in range(matrix_right.shape[1]):
				if not np.isnan(matrix_stay[i][j]): pylab.text(j-.5, self.height- i-.5,'X')
				else:
					pylab.text(j+0.2-.5, self.height-  i-.5,str(matrix_right[i][j])[1:4]+">")
					pylab.text(j-0.3-.5, self.height-i-.5,"<"+str(matrix_left[i][j])[1:4])

					pylab.text(j-.5, self.height- i+.1-.5,str(matrix_up[i][j])[1:4])
					pylab.text(j-.5, self.height- i-.1-.5,str(matrix_down[i][j])[1:4])

		pylab.xlim(-1,self.width-1)
		pylab.ylim(0,self.height)
		pylab.xticks(range(self.width),map(str,range(self.width)))
		pylab.yticks(range(self.height+1),reversed(map(str,range(self.height+1))))
		pylab.grid(color='r', linestyle='-', linewidth=2)
		pylab.title('Q (learning_step:%d)' % self.learning_step,size=16)
		fig.tight_layout()		



if __name__ == "__main__":
	
	# Ejemplo de 4x4 con goal en el medio
	# gw =gridWorld(height=4,width=4,goals=[[2,2]])
	
	
	# Ejemplo de gridWorld  de 2x3 
	gw =gridWorld(height=4,width=4,goals=[[2,2]])

	# Entreno 1K veces
	for epoch in range(1000):
		# Ploteo la matrix a los 10,200, y 999 epochs
		if epoch==10: gw.draw()
		if epoch==200: gw.draw()
		if epoch==999: gw.draw()

		# Elijo un state random para empezar
		start_state = [ random.randint(0,gw.height-1),random.randint(0,gw.width-1)]
		
		# Entreno
		gw.learn(start_state)
		
		if epoch%1000==0: print  epoch
	pylab.show()

