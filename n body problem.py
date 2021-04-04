
# %%
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
# from pandas import DataFrame
# from functools import lru_cache
# %%
def impl_mpr(T,N, sys, ic):
    t, dt = np.linspace(0,T, N, retstep=True)
    sol = np.empty((N,) + np.shape(ic))
    sol[0,...] = ic
    for i in range(N-1):
        print(f"sol[i] shape: {sol[i].shape}", f"sys shape:{sys(sol[0]).shape}")
        sol[i+1] = fsolve(
            lambda x : x - (sol[i] + dt*(sys(0.5*(sol[i] + x)))), sol[i])
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
    sum = np.zeros((n, dim))
    for k in range(n):
        mk = state[k,0]
        qk = state[k, 2]
        for i in range(n):
            if k == i:
                continue
            mi = state[i,0]
            qi = state[i,2]
            sum[k,:] += mi*mk/(np.linalg.norm(qi - qk)**3) * (qi - qk)
    # assert all(sum != 0)
    print(sum)
    return sum

# %%
t, y = impl_mpr(3, 100, rhs, ic)

plt.plot(*y[0,2])
plt.plot(*y[1,2])
plt.show()
# %%
