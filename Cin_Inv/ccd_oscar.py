#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Robótica Computacional - 
# Grado en Ingeniería Informática (Cuarto)
# Práctica: Resolución de la cinemática inversa mediante CCD
#           (Cyclic Coordinate Descent).

import sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
import colorsys as cs

# ******************************************************************************
# Declaración de funciones

def muestra_origenes(O,final=0):
  # Muestra los orígenes de coordenadas para cada articulación
  print('Origenes de coordenadas:')
  for i in range(len(O)):
    print('(O'+str(i)+')0\t= '+str([round(j,3) for j in O[i]]))
  if final:
    print('E.Final = '+str([round(j,3) for j in final]))

def muestra_robot(O,obj):
  # Muestra el robot graficamente
  plt.figure()
  plt.xlim(-L,L)
  plt.ylim(-L,L)
  T = [np.array(o).T.tolist() for o in O]
  for i in range(len(T)):
    plt.plot(T[i][0], T[i][1], '-o', color=cs.hsv_to_rgb(i/float(len(T)),1,1))
  plt.plot(obj[0], obj[1], '*')
  plt.pause(0.0001)
  plt.show()
  
#  input()
  plt.close()

def matriz_T(d,th,a,al):
   # Los angulos son en radianes
  return [[cos(th), -sin(th)*cos(al),  sin(th)*sin(al), a*cos(th)]
         ,[sin(th),  cos(th)*cos(al), -sin(al)*cos(th), a*sin(th)]
         ,[      0,          sin(al),          cos(al),         d]
         ,[      0,                0,                0,         1]
         ]

def cin_dir(th,a):
  #Sea 'th' el vector de thetas
  #Sea 'a'  el vector de longitudes
  # Devuelve lista de coordenadas x e y de cada uno de los
  # puntos o respecto al referencial 0
  # o = [[x00, y00], [x10, y10], [x20, y20].....]
  T = np.identity(4)
  origenes = [[0,0]]
  for i in range(len(th)):
    T = np.dot(T,matriz_T(0,th[i],a[i],0))
    tmp=np.dot(T,[0,0,0,1])
    origenes.append([tmp[0],tmp[1]])
  return origenes

# ******************************************************************************
# Cálculo de la cinemática inversa de forma iterativa por el método CCD

# valores articulares arbitrarios para la cinemática directa inicial
th=[0.,0.,0.]
a =[0.,5.,5.]
prismatica = [False, True, False]
# Si la 
to_radian = pi/180
thMax = np.array([90 * to_radian, 10, 90 * to_radian])
thMin = np.array([-90 * to_radian, 0, -90 * to_radian])


L = sum(a) # variable para representación gráfica
EPSILON = .01

#plt.ion() # modo interactivo

# introducción del punto para la cinemática inversa
if len(sys.argv) != 3:
  sys.exit("python " + sys.argv[0] + " x y")
objetivo=[float(i) for i in sys.argv[1:]]
O=cin_dir(th,a)
#O=zeros(len(th)+1) # Reservamos estructura en memoria
 # Calculamos la posicion inicial
print ("- Posicion inicial:")
muestra_origenes(O)

numArti = len(th)

dist = float("inf")
prev = 0.
iteracion = 1
while (dist > EPSILON and abs(prev-dist) > EPSILON/100.):
  prev = dist
  posiciones=[cin_dir(th,a)]
  # Posiciones = [
  # [[x00, y00], [x10, y10], [x20, y20].....] posicion inicial
  # [[x00, y00], [x10, y10], [x20, y20].....] tras una corrección
  # [[x00, y00], [x10, y10], [x20, y20].....] tras dos correcciones
  # ]
  # Para cada combinación de articulaciones:
  for i in range(numArti):
    artActual = numArti - i - 1

    if not prismatica[artActual]:
      # Es articulacion de rotacion
      # cálculo de la cinemática inversa:

      posiciones_actuales = posiciones[-1]

      # Posiciones de los puntos
      P_i = posiciones_actuales[artActual]

      P_e = posiciones_actuales[-1]

      P_t = objetivo

      # Creamos los vectores
      V_i_e = np.subtract(P_e, P_i)

      V_i_t = np.subtract(P_t, P_i)

      # Calculamos los ángulos
      ang_v_e = atan2(V_i_e[1], V_i_e[0])
      ang_v_t = atan2(V_i_t[1], V_i_t[0])

      # Restamos los ángulos
      delta_theta = ang_v_t - ang_v_e

      th[artActual] = th[artActual] + delta_theta 

      # Normalizamos el angulo
      # th[artActual] = atan2(sin(th[artActual]), cos(th[artActual]))
      th[artActual] = (th[artActual] + pi) % (2*pi) - pi

      # Limitar el valor nuevo
      if th[artActual] >= thMax[artActual]:
        th[artActual] = thMax[artActual]
      elif th[artActual] <= thMin[artActual]:
        th[artActual] = thMin[artActual]
    else:
      # Es prismática
      # Calcular omega = sumatorio de titas anteriores
      j = 0
      titaP = 0
      while j < len(th) - i:
        titaP += th[j]
        j += 1

      # Calculamos d = unitario - (R - Ox)
      vectorUnitario = [cos(titaP), sin(titaP)]

      vectorExtension = np.subtract(objetivo, posiciones[artActual][numArti])

      # L = L + d

      d = np.dot(vectorUnitario, vectorExtension)

      print(d)

      a[artActual] += d

      # Limitar L segun gMin y gMax
      if a[artActual] >= thMax[artActual]:
        a[artActual] = thMax[artActual]
      elif a[artActual] <= thMin[artActual]:
        a[artActual] = thMin[artActual]

    posiciones.append(cin_dir(th,a))
  dist = np.linalg.norm(np.subtract(objetivo, posiciones[-1][-1]))
  print ("\n- Iteracion " + str(iteracion) + ':')
  muestra_origenes(posiciones[-1])
  muestra_robot(posiciones,objetivo)
  print ("Distancia al objetivo = " + str(round(dist,5)))
  iteracion+=1
  posiciones[0]=posiciones[-1]

if dist <= EPSILON:
  print ("\n" + str(iteracion) + " iteraciones para converger.")
else:
  print ("\nNo hay convergencia tras " + str(iteracion) + " iteraciones.")
print ("- Umbral de convergencia epsilon: " + str(EPSILON))
print ("- Distancia al objetivo:          " + str(round(dist,5)))
print ("- Valores finales de las articulaciones:")
for i in range(len(th)):
  print ("  theta" + str(i+1) + " = " + str(round(th[i],3)))
for i in range(len(th)):
  print ("  L" + str(i+1) + "     = " + str(round(a[i],3)))
