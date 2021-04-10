
# %%
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# This code doesnt work, one reason is that fsolve cant solve tensor valued funcions. 
# I worked arround it by reshapeing everything into vectors and then into tensors and back into vectors
# However that doesnt seem to be the only problem with this code. 
# Somehow the y axis of the position seems to be missing. The collumn only contains zeros.
# I guess that there may be some mistake with reshapeing and indecies. (or maybe I am stupid and just plotting the wrong data)
# I dont even really know how to start finding this issue, those tensors make my head hurt XD

# %%
def mpr_solve(x, sol, i, dt, right_hand_side):
    # fsolve is only able to solve vector valued functions, so I will have to reshape everything
    x = np.reshape(x, np.shape(sol[0]))
    out = x - (sol[i] + dt*(right_hand_side(0.5*(sol[i] + x))))
    return np.reshape(out, (-1))

def impl_mpr(T,N, right_hand_side, intital_conditions):
    t, dt = np.linspace(0,T, N, retstep=True)
    sol = np.empty((N,) + np.shape(initial_conditions))
    sol[0] = initial_conditions
    assert sol[0].shape == right_hand_side(sol[0]).shape,f"sol[i] shape: {sol[0].shape} doesnt match sys shape:{right_hand_side(sol[0]).shape}"
    for i in range(N-1):
        sol[i+1] = np.reshape( fsolve(mpr_solve,sol[i], args=(sol, i, dt, right_hand_side)), np.shape(initial_conditions))
    return t, sol
# %%
N = 2
dim = 2
G = 1
m1 = 500
m2 = 1
p1 = [0,0]
p2 = [0,np.sqrt(G*m1*0.5)]
pos1 = np.array([0,0])
pos2 = np.array([2,0])
# the internet says that initializing an np.array with objects doesnt result in computational speedups
initial_conditions = np.array([
    [[m1]*dim, p1, pos1],
    [[m2]*dim, p2, pos2]
])
# %%
def rhs(state):
    "right hand side"
    p_dot = G*get_sum(state)
    q_dot = state[:,1] / state[:, 0]
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
t, sol = impl_mpr(100, 100, rhs, initial_conditions)
print("---------------------------------------")
print("This is the whole solution")
print(sol.shape, initial_conditions.shape)
print(sol)
print("----------------------------------------------")
x1,y1 = np.transpose(sol[:,0,2,0:2], (-1,0))
x2,y2 = np.transpose(sol[:,1,2,0:2], (-1,0))
print(x1,y1)
print(x2,y2)
plt.plot(x1,y1)
plt.plot(x2,y2)
# plt.show()
plt.clf()
plt.plot(x2)
plt.show()
# %%
