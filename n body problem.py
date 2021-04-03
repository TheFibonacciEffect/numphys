
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
    sol[0] = ic
    for i in range(N-1):
        sol[i+1] = fsolve(
            lambda x : x - (sol[i] + dt*(sys(0.5*(sol[i] + x)))), sol[i]
            )
    return t, sol
# %%
N = 2
dim = 2
# [planet, mass(0) or momentum (1) or position (2)]
ic = np.empty((N,3,dim))
m1 = 1
m2 = 2
p1 = p2 = [3,4]
pos1 = np.array([5,6])
pos2 = np.array([7,8])
ic = np.array([
    [[m1]*dim, p1, pos1],
    [[m2]*dim, p2, pos2]
])
# ic = np.array([ [1,[0,0], [1,0]], [1,[0,0],[0,0]] ],object)
# ic = np.array([ [[m1]*dim,[0,0], [3,0]], [[m2]*dim,[0,0],[0,0]] ])

G = 1
# %%
state = ic
qs = state[:, 2, :]
# qs = np.swapaxes(qs,1,0)
qs_rep = np.repeat(qs[..., np.newaxis], N, -1)
assert all(qs_rep[0,...,0]==pos1)
# qs_swap = np.swapaxes(qs_rep, 1,-1)
qs_transp = np.swapaxes(qs_rep, 0,-1)
# qs_diff = qs_swap - qs_transp
# assert all( qs_diff[0,0] == pos1 - pos1)
qs_diff = qs_rep - qs_transp
# put the coordinates into the last axis
qs_diff = np.swapaxes( qs_diff, 1,-1)

# np.ma.MaskedArray(qs_diff, np.identity(N, dtype=bool))
# %%
def rhs(state):
    p_dot = G*np.sum(
        get_sum(state), axis=1
    )
    q_dot = ic[:,2] / ic[:, 0]

def get_sum(state):
    # x = np.reshape([1,2,3]*3, (3,3))
    # x[np.identity(3, dtype=bool)] = 0
    # np.sum(x, 1)
    # np.ma.MaskedArray(x,np.identity(n, dtype=bool))
    # m = state[:,0]* np.reshape(state[:,0], (-1,1))
    # q = state[:,2]* np.reshape(state[:,2], (-1,1))
    # mi =np.ma.MaskedArray(m, np.identity(N))
    # mi =np.ma.MaskedArray(q, np.identity(N))
    qs = state[:, 2, :]
    qs_rep = np.repeat(qs[..., np.newaxis], N, -1)
    np.transpose(qs_rep,0)
    return 