import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# %%
task = input("which task should be solved?")
if task == "d":
    N = 2
    dim = 3
    G = 1
    m1 = 500
    m2 = 1
    q1 = np.array([0,0,0])
    q2 = np.array([2,0,0])
    p1 = [0,0,0]
    p2 = [0,np.sqrt(G*m1*0.5),0]
    ic = np.ravel([q1,q2,p1,p2])
    ms = np.array([m1,m2])
elif task=="e":
    N = 3
    dim = 3
    G = 1
    q1 = np.array([0.97000436,-0.24308753 ,0])
    q2 = np.array((-0.97000436,0.24308753,0))
    q3 = np.zeros(3)
    p1 = (0.46620368,0.43236573,0)
    p2 = (0.46620368,0.43236573,0)
    p3 = (-0.93240737,-0.86473146,0)
    ic = np.ravel([q1,q2, q3,p1,p2, p3])
    ms = np.array([1,1,1])

def right_hand_side(ic,ms,N):
    assert len(ic)%N == 0
    state = np.reshape(ic, (2*N, dim)) # reshape so that it is a vector of momenta and positions
    ps = state[N:]
    qs = state[:N]
    q_dot = ps/ms.reshape(-1,1)
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

print(f"first right hand side {right_hand_side(ic, ms, N)}")

def impl_mpr(T,num, right_hand_side, initial_conditions, ms,n):
    t, dt = np.linspace(0,T, num, retstep=True)
    sol = np.empty((num,) + np.shape(initial_conditions))
    sol[0] = initial_conditions
    for i in range(num-1):
        sol[i+1] = fsolve(lambda x: x - (sol[i] + dt*right_hand_side(0.5*(sol[i] + x), ms, n)) ,sol[i])
    return sol

def expl_euler(T,num, right_hand_side, initial_conditions, ms,n):
    t, dt = np.linspace(0,T, num, retstep=True)
    sol = np.empty((num,) + np.shape(initial_conditions))
    sol[0] = initial_conditions
    for i in range(num-1):
        sol[i+1] = sol[i] + dt*right_hand_side(sol[i], ms, n)
    return sol

T = 3
num = 1000
y = impl_mpr(T,num,right_hand_side,ic, ms, N)
y_t = np.transpose(y)
x1, y1, z1, x2, y2, z2, x3,y3,z3, *_ = y_t # _ is used as a dummy variable for all the momenta, see Extended Iterable Unpacking
plt.plot(x1,y1)
plt.plot(x2,y2)
if N == 3:
    plt.plot(x3,y3)
plt.show()
# plt.savefig("3body.png")

# %%

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x1,y1,z1)
ax.plot(x2,y2,z2)
if N == 3:
    ax.plot(x3,y3,z3)

# %%
