import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from nbody import planet_ic
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation


try: from tqdm import tqdm
except ImportError:
    print("tqdm not found, install tqdm for an awesome progress bar :D")
    def tqdm(x):
        return x
# %%

def right_hand_side(ic,ms,N):
    assert len(ic)%(2*N) == 0, f"the number of planets {N} doesnt match the input with shape {np.shape(ic)}"
    dim = 3
    state = np.reshape(ic, (2*N, dim)) # reshape so that it is a vector of momenta and positions
    ps = state[N:]
    qs = state[:N]
    q_dot = ps/ms.reshape(-1,1)
    p_dot = G*get_sum(qs,ms)
    # return the flattened output with [pos| momentum]
    return np.append(q_dot.reshape(-1), p_dot.reshape(-1))

def acc_vvv(positions,ms,N):
    assert len(positions)%N == 0, f"the number of planets {N} doesnt match the input with shape {np.shape(positions)}, ie {np.shape(positions)}%{N} != 0"
    dim = 3
    state = np.reshape(positions, (N, dim)) # reshape so that it is a vector of positions
    F = G*get_sum(state,ms)# p_dot/m = p_double_dot
    q_dd = np.array([F[k] / ms[k%3] for k in range(len(F))]) #TODO optimize
    # return the flattened output with [q_dd]
    print(f"q_dd shape: {np.shape(q_dd)}")
    return q_dd[:,0] #TODO why is this shape (18,1) instead of (18, )

def get_sum(qs, ms): # (6, 3), (6, 1)
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
    for i in tqdm(range(num-1)):
        sol[i+1] = fsolve(lambda x: x - (sol[i] + dt*right_hand_side(0.5*(sol[i] + x), ms, n)) ,sol[i])
    return sol

def expl_euler(T,num, right_hand_side, initial_conditions, ms,n):
    t, dt = np.linspace(0,T, num, retstep=True)
    sol = np.empty((num,) + np.shape(initial_conditions))
    sol[0] = initial_conditions
    for i in tqdm(range(num-1)):
        sol[i+1] = sol[i] + dt*right_hand_side(sol[i], ms, n)
    return sol

def vvv(T,num, acc, initial_conditions, ms,n):
    """returns: position with shape: (num, )
    """
    t, dt = np.linspace(0,T, num, retstep=True)
    xi = initial_conditions[:N*3]
    pi = initial_conditions[3*N:]
    x = np.empty((num,) + np.shape(xi))
    v = np.empty((num,) + np.shape(pi))
    x[0] = xi
    print(f"num: {num}")
    print(v[0].shape, ms.shape, pi.shape)
    v[0] = [pi[n]/ms[n%3] for n in range(len(pi)) ] #TODO optimize
    for i in tqdm(range(num-1)):
        x[i+1] = x[i] + dt* v[i] + 0.5* dt**2*acc(x[i], ms, n)
        v[i+1] = v[i] + dt* acc(x[i], ms, n)
    return x

def plot_orbits(three_dim, y, animate):
    if animate:
        plot_animation(y)
        return

    N = len(y[0])/(3*2)
    assert N.is_integer(), "is your problem not in 3 dimensions?"
    N = int(N)
    fig = plt.figure()
    if three_dim:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    if method == "vvv":
        N = int(len(y[0])/(3))

    y_t = np.transpose(y)
    slicing = 100
    for i in range(N):
        xi = y_t[3*i][::slicing]
        yi = y_t[3*i+1][::slicing]
        zi = y_t[3*i+2][::slicing]
        if three_dim and animate:
            plot_animation((xi,yi,zi))
        if three_dim:
            ax.plot(xi,yi,zi, "-")
        else:
            plt.plot(xi,yi)
    plt.show()
    return

def plot_animation(data):
    print("plot animation")

    def update_lines(num, dataLines, lines) :
        for line, data in zip(lines, dataLines) :
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data(data[0:2, num])
            line.set_3d_properties(data[2,num])
        return lines

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    # NOTE: Can't pass empty arrays into 3d version of plot()
    # passing first data points
    number_of_planets = 6
        
    data = np.transpose(data) # [time, posistions | momenta] -> [posistions | momenta, time]
    print(f"shape {data.shape}")
    data = np.array([[data[3*n,:], data[3*n+1,:], data[3*n +2, :]] for n in range(number_of_planets)])
    print(f"new shape {data.shape}")
    for planet in data:
        print(planet[0,0], planet[1,0], planet[2,0])
    lines = [ax.plot(planet[0,0], planet[1,0], planet[2,0], "o")[0] for planet in data]

    # Setting the axes properties
    ax.set_xlim3d([-15, 20.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-30.0, 25.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-15.0, 5.0])
    ax.set_zlabel('Z')

    ax.set_title('3D Test')

    # Creating the Animation object
    number_of_time_points = data.shape[-1]
    line_ani = FuncAnimation(fig, update_lines, np.arange(0,number_of_time_points,1), fargs=(data, lines), interval=50, blit=False)

    plt.show()

if __name__== "__main__":
    # task = input("which task should be solved? ")
    task = "f"
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
        N = 6
        G = 2.95912208286e-4
        ms, ic = planet_ic()
        ic = np.ravel(ic)
    else:
        print("not implemented, try again")
        quit()

    # method = input("method (expl_euler, mpr, vvv): ")
    method = "expl_euler"
    translator = {
        "expl_euler": expl_euler,
        "mpr": impl_mpr,
        "vvv": vvv
    }
    T = 20000
    # T = float(input("endtime: "))
    num = 1000
    # num = int(input("steps: "))
    d = {"y" : True,"n": False}
    # output = d[input("3D output? (y/n) ")]
    output = True
    animate = True

    if method == "vvv":
        y = translator[method](T,num,acc_vvv,ic, ms, N)
    else: y = translator[method](T,num,right_hand_side,ic, ms, N)

    print(f"shape of solution: {y.shape}")
    print(y)
    plot_orbits(output, y, animate)