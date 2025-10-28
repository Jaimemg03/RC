# -*- coding: utf-8 -*-
from math import *

# resolución de la cinemática directa mediante Denavit-Hartenberg


# parametros D-H primera articulación
a1=10
d1=0
alfa1=0


# parametros D-H segunda articulación
a2=5
d2=0
alfa2=0


# origenes
o11=[0,0,0,1]
o22=[0,0,0,1]


def matriz_T(theta,alpha,a,d):
        # calcula la matriz T
        f0=[cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)]
        f1=[sin(theta),cos(theta)*cos(alpha), -sin(alpha)*cos(theta), a*sin(theta)]
        f2=[0, sin(alpha), cos(alpha), d]
        f3=[0,0,0,1]
        
        T=[f0,f1,f2,f3]
        return T

def multiplica(matriz,vector_entrada):
        # multiplica una matriz 4x4 por un vector 4x1

        vector_salida=[]

        for i in range(4):
                suma=0
                for j in range (4):
                        suma+=matriz[i][j]*vector_entrada[j]
                vector_salida.append(suma)
        
        return vector_salida


# introducción de los valores de las articulaciones
print('')
t1=input('valor de theta1 en grados  ')
t1=t1*pi/180
t2=input('valor de theta2 en grados  ')
t2=t2*pi/180


# calculo matrices transformación
T01=matriz_T(t1,alfa1,a1,d1)
T12=matriz_T(t2,alfa2,a2,d2)


# calculo punto uno del robot
o10=multiplica(T01,o11)


# calculo punto dos del robot
o21=multiplica(T12,o22)
o20=multiplica(T01,o21)


print('')
print('punto uno del robot')
o10=[round(o10[0]),round(o10[1]),round(o10[2]),round(o10[3])]
print(o10)

print('')
print('punto dos del robot')
o20=[round(o20[0]),round(o20[1]),round(o20[2]),round(o20[3])]
print(o20)




