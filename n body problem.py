
# %%
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
# from pandas import DataFrame
# from functools import lru_cache
# %%
def mpr_solve(x, sol, i, dt, right_hand_side):
    # fsolve is only able to solve vector valued functions, so I will have to reshape everything
    x = np.reshape(x, np.shape(sol[0]))
    out = x - (sol[i] + dt*(right_hand_side(0.5*(sol[i] + x))))
    return np.reshape(out, (-1))

def impl_mpr(T,N, right_hand_side, ic):
    t, dt = np.linspace(0,T, N, retstep=True)
    sol = np.empty((N,) + np.shape(ic))
    sol[0] = ic
    for i in range(N-1):
        assert sol[0].shape == right_hand_side(sol[0]).shape,f"sol[i] shape: {sol[i].shape}, sys shape:{right_hand_side(sol[0]).shape}"
        print(sol[i].shape)
        sol[i+1] = np.reshape( fsolve(mpr_solve,sol[i], args=(sol, i, dt, right_hand_side)), np.shape(ic))
    return t, sol
# %%
N = 2
dim = 2
G = 1
# [planet, mass(0) or momentum (1) or position (2)]
ic = np.empty((N,3,dim))
m1 = 500
m2 = 1
p1 = [0,0]
p2 = [0,np.sqrt(G*m1*0.5)]
pos1 = np.array([0,0])
pos2 = np.array([2,0])
# the internet says that initializing an np.array with objects doesnt result in computational speedups
ic = np.array([
    [[m1]*dim, p1, pos1],
    [[m2]*dim, p2, pos2]
])
# ic = np.array([ [1,[0,0], [1,0]], [1,[0,0],[0,0]] ],object)
# ic = np.array([ [[m1]*dim,[0,0], [3,0]], [[m2]*dim,[0,0],[0,0]] ])
# %%
def rhs(state):
    "right hand side"
    p_dot = G*get_sum(state)
    q_dot = state[:,2] / state[:, 0]
    out = np.array([state[:,0], p_dot, q_dot]) # (variable, body, dimension)
    out = np.transpose(out, (1,0,2)) # (body, variable, dimension)
    assert out.shape == state.shape, f"shape in {state.shape} != shape out {out.shape}"
    return out
# %%
def get_sum(state):
    "gets the sum for computing p_dot, satisfying the condition that i!=j"
    n = np.shape(state)[0]
    sum_ = np.zeros((n, dim))
    for k in range(n):
        mk = state[k,0]
        qk = state[k, 2]
        for i in range(n):
            if k == i:
                continue
            mi = state[i,0]
            qi = state[i,2]
            sum_[k,:] += mi*mk/(np.linalg.norm(qi - qk)**3) * (qi - qk)
    return sum_

# %%
t, sol = impl_mpr(3, 100, rhs, ic)
print("---------------------------------------")
print(sol.shape, ic.shape)
print(sol)
print("----------------------------------------------")
x1,y1 = np.transpose(sol[:,0,2,0:2], (-1,0))
x2,y2 = np.transpose(sol[:,1,2,0:2], (-1,0))
print(x1,y1)
print(x2,y2)
plt.plot(x1,y1)
plt.plot(x2,y2)
# plt.show()
# %%
