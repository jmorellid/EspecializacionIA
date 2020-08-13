import numpy as np
import os

import matplotlib.pyplot as plt


os.chdir('C:/Users/jota_/00_Especialización_IA/EspecializacionIA')

POSITION_PATH = '00_Datasets/01_Raw/KALMANposicion.dat'
VELOCITY_PATH = '00_Datasets/01_Raw/KALMANvelocidad.dat'
ACCELERATION_PATH = '00_Datasets/01_Raw/KALMANaceleracion.dat'

pos = np.loadtxt(POSITION_PATH)[:,1:4]
vel = np.loadtxt(VELOCITY_PATH)[:,1:4]
acc = np.loadtxt(ACCELERATION_PATH)[:,1:4]

def kalman_filter(X_0, P_0, med, A, C, Q, R, x=1, v=0, a=0):

    #registro los tamaños
    R_size = R.shape[0]
    m = len(med)

    #inicializo los parametros
    X_nn1 = X_0
    X_nn = X_0

    Kn = np.zeros(shape=(1, 9, R_size))

    P_nn1 = np.zeros(shape=(1, 9, 9))
    P_nn = P_0.reshape((1, 9, 9))

    for i in range(m):

        X_nn1 = np.vstack([X_nn1, np.zeros(X_0.shape)])
        X_nn1[i, :] = A @ X_nn[i, :]
        P_nn1 = np.append(P_nn1, (A @ P_nn[i] @ A.T + Q).reshape((1, 9, 9)), axis=0)

        Kn = np.append(Kn, (P_nn1[i + 1] @ C.T @ np.linalg.inv(C @ P_nn1[i + 1] @ C.T + R)).reshape(1, 9, R_size), axis=0)

        X_nn = np.vstack([X_nn, np.zeros(X_0.shape)] )
        X_nn[i + 1] = X_nn1[i] + Kn[i + 1] @ (med[i, 0:1 + 3 * (x + v + a)] - C @ X_nn1[i])

        P_nn = np.append(P_nn, ((np.eye(9) - Kn[i + 1] @ C) @ P_nn1[i + 1]).reshape((1, 9, 9)), axis=0)

    return X_nn


X_0 = np.array([10.7533, 36.6777, -45.1769, 1.1009, -17.0, 35.7418, -5.7247, 3.4268, 5.2774]).reshape(1, -1)
P_0 = np.diag([100, 100, 100, 1, 1, 1, 0.01, 0.01, 0.01])
Q = 0.3 * np.eye(9)
H = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
R = np.diag([100, 100, 100])
C = np.hstack((np.eye(3), np.zeros(shape=(3,3)), np.zeros(shape=(3,3))))


dT = 1 #Delta de tiempo medido
# Filas de la matriz de la dinámica del sistema
p_row = np.hstack((np.eye(3),np.eye(3)* dT, np.eye(3)* (dT**2)/2) )
v_row = np.hstack((np.zeros(shape=(3, 3)), np.eye(3), np.eye(3)* dT) )
a_row = np.hstack((np.zeros(shape=(3, 3)), np.zeros(shape=(3, 3)), np.eye(3)))

A = np.vstack((p_row,v_row,a_row)) # Matriz con la dinámica del sistema


ruido_normal = np.random.normal(loc=0,scale=100,size=(351,3))
pos_gauss = pos + ruido_normal

pred_Gauss = kalman_filter(X_0, P_0, pos_gauss, A, C, Q, R)

Axes = {0:'X', 1:'Y', 2:'Z'}

for i,j in [(0,1), (1,2), (0,2)]:
    fig0, ax = plt.subplots(figsize=(10,10))
    ax.plot(pred_Gauss[:,i],pred_Gauss[:,j],color='blue',label='Prediction',zorder=3,lw=2)
    ax.plot(pos_gauss[:,i], pos_gauss[:,j], color='grey',ls='--',alpha=0.5,label='Measurement',zorder=1)
    ax.plot(pos[:,i], pos[:,j],color='green',ls='-',alpha=0.5,label='Real',lw=1,zorder=2)
    ax.legend()

    axes = [Axes.get(Ax) for Ax in [i, j]]
    ax.set_title('Comparison between {} predicted, measured and real trajectories'.format(axes))



# Ejercicio nro 2

ruido_uni = np.random.uniform(-5*12**0.5, 5*12**0.5, size=(351, 3))
pos_uni = pos+ ruido_uni

R = np.eye(3) * 10 * 12**0.5

pred_uni = kalman_filter(X_0, P_0, pos_uni, A, C, Q, R)

for i,j in [(0,1), (1,2), (0,2)]:
    fig1, ax = plt.subplots(figsize=(10,10))
    ax.plot(pred_uni[:,i],pred_uni[:,j],color='blue',label='Prediction',zorder=3,lw=2)
    ax.plot(pos_uni[:,i], pos_uni[:,j], color='grey',ls='--',alpha=0.5,label='Measurement',zorder=1)
    ax.plot(pos[:,i], pos[:,j],color='green',ls='-',alpha=0.5,label='Real',lw=1,zorder=2)
    ax.legend()

    axes = [Axes.get(Ax) for Ax in [i, j]]
    ax.set_title('Comparison between {} predicted, measured and real trajectories'.format(axes))



for i,j in [(0,1), (1,2), (0,2)]:
    fig2, ax = plt.subplots(figsize=(10,10))
    ax.plot(pred_uni[:,i],pred_uni[:,j],color='blue',label='Prediction',zorder=3,lw=2)
    ax.plot(pos_uni[:,i], pos_uni[:,j], color='grey',ls='--',alpha=0.5,label='Measurement',zorder=1)

    ax.plot(pred_Gauss[:,i],pred_Gauss[:,j],color='red',label='Prediction',zorder=3,lw=2)
    ax.plot(pos_gauss[:,i], pos_gauss[:,j], color='violet',ls='--',alpha=0.5,label='Measurement',zorder=1)

    ax.plot(pos[:,i], pos[:,j],color='green',ls='-',alpha=0.5,label='Real',lw=1,zorder=2)
    ax.legend()

    axes = [Axes.get(Ax) for Ax in [i, j]]
    ax.set_title('Comparison between {} predicted, measured and real trajectories'.format(axes))

plt.show()