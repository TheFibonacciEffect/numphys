
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
G = 1
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
# %%
def rhs(state):
    "right hand side"
    p_dot = G*np.sum(
        get_sum(state), axis=1
    )
    q_dot = ic[:,2] / ic[:, 0]
# %%
def get_sum(state):
    "gets the sum for computing p_dot, satisfying the condition that i!=j"
    ms = state[:, 0, :]
    qs = state[:, 2, :]
    ms, ms_T = permute(ms)
    qs, qs_T = permute(qs)
    m_product = ms*ms_T
    qs_diff = qs - qs_T
    qs_norm = np.linalg.norm(qs_diff, 1)
    out = m_product * qs_diff / qs_norm**3
    return mask(out)

def permute(data):
    """input with three axis
    computes the transpose of the repeated input containing vectors
    """
    print(data)
    data = np.repeat(data[:,np.newaxis,:,:], data.shape[0], 1)
    data_transp = np.swapaxes(data, 0, 1)
    return data, data_transp
    
def mask(inp):
    assert inp.shape[0] == inp.shape[1]
    n = inp.shape[0]
    return np.ma.MaskedArray(inp, np.identity(n, bool))

# %% 
get_sum(ic)