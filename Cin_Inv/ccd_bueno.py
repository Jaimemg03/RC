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
   
  return [[cos(th), -sin(th)*cos(al),  sin(th)*sin(al), a*cos(th)]
         ,[sin(th),  cos(th)*cos(al), -sin(al)*cos(th), a*sin(th)]
         ,[      0,          sin(al),          cos(al),         d]
         ,[      0,                0,                0,         1]
         ]

def cin_dir(th,a):
  #Sea 'th' el vector de thetas
  #Sea 'a'  el vector de longitudes
  T = np.identity(4)
  o = [[0,0]]
  for i in range(len(th)):
    T = np.dot(T,matriz_T(0,th[i],a[i],0))
    tmp=np.dot(T,[0,0,0,1])
    o.append([tmp[0],tmp[1]])
  return o

def to_radian(angel):
  return angel * pi/180

# ******************************************************************************
# Cálculo de la cinemática inversa de forma iterativa por el método CCD

# valores articulares arbitrarios para la cinemática directa inicial
th=[0.,0.,0.]
a =[0.,5.,5.]
tipo_articulacion = [0, 1, 0] # 0 = rotacion, 1 = prismática
limite_sup = [to_radian(90), 10, to_radian(90)]
limite_inf = [to_radian(-90), 0, to_radian(-90)]

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

dist = float("inf")
prev = 0.
iteracion = 1
while (dist > EPSILON and abs(prev-dist) > EPSILON/100.):
  prev = dist
  O=[cin_dir(th,a)]
  # Para cada combinación de articulaciones:
  for i in range(len(th)):
    if tipo_articulacion[len(th) - 1 - i] == 0: # Si la articulación analizada es de rotación

      # ==================================================================== #
      #                  [ Cálculo de la cinemática inversa ]                #
      # ==================================================================== #
      Px, Py = O[i][len(th) - 1 - i] # coordenadas (x, y) de la articulacion que se esta estudiando
      EFx, EFy = O[i][-1] # coordenadas (x, y) del EF
      Tx, Ty = objetivo # coordenadas (x, y) del punto objetivo

      alpha1 = atan2(EFy - Py, EFx - Px)
      alpha2 = atan2(Ty - Py, Tx - Px)
      inc_th = alpha2 - alpha1
      new_th = th[len(th) - 1 - i] + inc_th

      # ==================================================================== #
      #                      [ Normalización del ángulo ]                    #
      # ==================================================================== #
      # Normalizar new_th (entre -pi y pi)y después comprobar que no se pase de los límites para los límites hacer un vector del mismo tamaño que th y poner ahi los límites
      new_th = (new_th + pi) % (2*pi) - pi

      # ==================================================================== #
      #                         [ Límites excedidos ]                        #
      # ==================================================================== #
      if new_th > limite_sup[len(th) - 1 - i]:
        new_th = limite_sup[len(th) - 1 - i]
      elif new_th < limite_inf[len(th) - 1 - i]:
        new_th = limite_inf[len(th) - 1 - i]

      th[len(th) - 1 - i] = new_th
    elif tipo_articulacion[len(th) - 1 - i] == 1: # Si la articulación analizada es prismática
      # ==================================================================== #
      #                  [ Cálculo de la cinemática inversa ]                #
      # ==================================================================== #
      # Calcular d = v * u, v = (Xt - Xp, Yt - Yp), u = (cos(w), sen(w)), w =  Sumarorio de los tita anteriores, incluido el de la propia articulacion
      w = sum(th[len(th) - 1 - i: + 1])
      d = np.dot(np.subtract(objetivo, O[i][-1]), [cos(w), sin(w)])
      a[len(th) - 1 - i] += d

      # ==================================================================== #
      #                         [ Límites excedidos ]                        #
      # ==================================================================== #
      if a[len(th) - 1 - i] > limite_sup[len(th) - 1 - i]:
        a[len(th) - 1 - i] = limite_sup[len(th) - 1 - i]
      elif a[len(th) - 1 - i] < limite_inf[len(th) - 1 - i]:
        a[len(th) - 1 - i] = limite_inf[len(th) - 1 - i]
  
    O.append(cin_dir(th,a))

  dist = np.linalg.norm(np.subtract(objetivo,O[-1][-1]))
  print ("\n- Iteracion " + str(iteracion) + ':')
  muestra_origenes(O[-1])
  muestra_robot(O,objetivo)
  print ("Distancia al objetivo = " + str(round(dist,5)))
  iteracion+=1
  O[0]=O[-1]

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
