import numpy as np
import math as math
from time import perf_counter
from scipy.integrate import odeint
import pandas as pd
from google.colab import files

#Runge Kutta Method Regular Approach
#Updated September 10
#Test using pythagorean problem

#Setup
G = 1
timestep = 0.00001 #seconds
sims = 1000000
pi = math.pi

#masses in kg
m = [1., 1., 1.]

#relative velocity to center of mass in (x,y)
vel = np.array([0.46620,	0.43237,
0.46620,	0.43237,
-0.93241, -0.86473])

#row,column zero index
#relative position to center of mass in (x,y)
pos = np.array([0.970040,	-0.24309,
-0.97004,	0.24309,
0, 0])


# dy / dt = f(t, U)
U0 = np.hstack((pos,vel))
U = U0
accel = np.zeros((1,6))
for i in range(0,3): 
    j = (i+1)%3
    k = (i+2)%3
    ijr2 = np.dot((U[j*2:j*2+2]-U[i*2:i*2+2]),(U[j*2:j*2+2]-U[i*2:i*2+2]))
    ijr3 = ijr2 * math.sqrt(ijr2)
    accel[0,i*2:i*2+2] = np.add(m[j]*(U[j*2:j*2+2]-U[i*2:i*2+2])/ijr3,accel[0,i*2:i*2+2])

def model(U,t): #U: positions,velocities
  accel = np.zeros((6))
  for i in range(0,3): 
    j = (i+1)%3
    k = (i+2)%3
    ijr2 = np.dot((U[j*2:j*2+2]-U[i*2:i*2+2]),(U[j*2:j*2+2]-U[i*2:i*2+2]))
    ijr3 = ijr2 * math.sqrt(ijr2)
    accel[i*2:i*2+2] = np.add(m[j]*(U[j*2:j*2+2]-U[i*2:i*2+2])/ijr3,accel[i*2:i*2+2])
    ikr2 = np.dot((U[k*2:k*2+2]-U[i*2:i*2+2]),(U[k*2:k*2+2]-U[i*2:i*2+2]))
    ikr3 = ikr2 * math.sqrt(ikr2)
    accel[i*2:i*2+2] = np.add(m[k]*(U[k*2:k*2+2]-U[i*2:i*2+2])/ikr3,accel[i*2:i*2+2])
  dUdt = np.hstack((U[6:12],accel))
  return dUdt
print(pos)
print(vel)


# time points
t = np.linspace(0,20)


results = np.zeros((40,4))

dt = 1e-3

# solve ODE
for i in range(3,7):
  t_end = dt*(10**i)
  t = np.linspace(0,t_end,(10**i)+1)
  for j in range(40):
    start = perf_counter() 
    y = odeint(model,U0,t)
    end = perf_counter()
    execution_time = end-start
    print("runtime (seconds) =",execution_time)
    results[j,i-3] = execution_time
  df = pd.DataFrame(data=results)
  df.to_csv("ScipySpeed.csv")


import matplotlib.pyplot as plt

