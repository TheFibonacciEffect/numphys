import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from nbody import planet_ic
# %%

def right_hand_side(ic,ms,N):
    assert len(ic)%N == 0
    dim = 3
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

def plot_orbits(three_dim, y):
    N = len(y[0])/(3*2)
    assert N.is_integer(), "is your problem not in 3 dimensions?"
    N = int(N)
    plt.clf()
    if three_dim:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    y_t = np.transpose(y)
    for i in range(N):
        xi = y_t[3*i]
        yi = y_t[3*i+1]
        zi = y_t[3*i+2]
        if three_dim:
            ax.plot(xi,yi,zi)
        else:
            plt.plot(xi,yi)
    plt.show()
    return

if __name__== "__main__":
    task = input("which task should be solved? ")
    if task == "d":
        N = 2
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
        G = 1
        q1 = np.array([0.97000436,-0.24308753 ,0])
        q2 = np.array((-0.97000436,0.24308753,0))
        q3 = np.zeros(3)
        p1 = (0.46620368,0.43236573,0)
        p2 = (0.46620368,0.43236573,0)
        p3 = (-0.93240737,-0.86473146,0)
        ic = np.ravel([q1,q2, q3,p1,p2, p3])
        ms = np.array([1,1,1])
    elif task=="f":
        N = 9
        G = 2.95912208286e-4
        ms, ic = planet_ic()
        print(ic)
        print(ms)
        ic = np.ravel(ic)
    else:
        print("try again")
        quit()

    method = input("mehod (expl_euler, mpr, vvv): ")
    translator = {
        "expl_euler": expl_euler,
        "mpr": impl_mpr,
        # "vvv": vvv
    }
    T = float(input("endtime: "))
    num = int(input("steps: "))
    d = {"Y" : True,"N": False}
    output = d[input("3D output (Y/N)")]

    y = translator[method](T,num,right_hand_side,ic, ms, N)

    plot_orbits(output, y)