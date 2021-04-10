import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# TODO These solutions look really strange, I believe there is something wrong with the differential equation, somehow.

N = 2
dim = 2
G = 1
m1 = 500
m2 = 1
q1 = np.array([0,0])
q2 = np.array([2,0])
p1 = [0,0]
p2 = [0,np.sqrt(G*m1*0.5)]

ic = np.ravel([q1,q2,p1,p2])
ms = np.array([m1,m2])

def right_hand_side(ic,ms,N):
    assert len(ic)%N == 0
    state = np.reshape(ic, (2*N, dim)) # reshape so that it is a vector of momenta and positions
    ps = state[N:]
    qs = state[:N]
    q_dot = ps/ms
    p_dot = G*get_sum(qs,ms)
    # return the flattened output with [pos| momentum]
    return np.append(q_dot.reshape(-1), p_dot.reshape(-1))

def get_sum(qs, ms):
    out = np.zeros_like(qs, dtype=np.float32)
    for k,qk in enumerate(qs):
        for i, qi in enumerate(qs):
            if i == k:
                continue
            out[k] += ms[i]*ms[k]/np.linalg.norm(qi-qk)**3 * (qi - qk)
    return np.ravel(out)

print(right_hand_side(ic, ms, N))

def impl_mpr(T,num, right_hand_side, initial_conditions, ms,n):
    t, dt = np.linspace(0,T, num, retstep=True)
    sol = np.empty((num,) + np.shape(initial_conditions))
    sol[0] = initial_conditions
    for i in range(num-1):
        # TODO check formula
        sol[i+1] = fsolve(lambda x: x - (sol[i] + dt*right_hand_side(0.5*(sol[i] + x), ms, n)) ,sol[i])
    return sol

def expl_euler(T,num, right_hand_side, initial_conditions, ms,n):
    t, dt = np.linspace(0,T, num, retstep=True)
    sol = np.empty((num,) + np.shape(initial_conditions))
    sol[0] = initial_conditions
    for i in range(num-1):
        # TODO check formula
        sol[i+1] = sol[i] + dt*right_hand_side(sol[i], ms, n)
    return sol

T = 3
num = 3000
y = expl_euler(T,num,right_hand_side,ic, ms, N)
print(y)
y_t = np.transpose(y)
print(np.shape(y_t), np.shape(y))
x1, y1, x2, y2, _,_,_,_ = np.transpose(y)
print(x2)
plt.plot(x1,y1)
plt.plot(x2,y2)
plt.show()