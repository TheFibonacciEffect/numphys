import numpy as np
import matplotlib.pyplot as plt

omega = 1.
m = 1.

def T(p):
    """
    Kinetic energy of the harmonic oscillator,
    where p is the momentum of the particle.
    """
    return p**2/(2*m)

def dTdp(p):
    """
    Derivative of the kinetic energy with respect to p.
    """
    return p / m

def U(q):
    """
    Potential energy of the harmonic oscillator with frequency omega
    where q is the spatial coordinate.
    """
    return 0.5*omega**2*q**2

def dUdq(q):
    """
    Derivative of the potential energy with respect to q.
    """
    return omega**2 * q

def phi_T(h,y0):
    """
    Analytic evolution operator of the kinetic part.
    """
    q0,p0,t0 = y0
    y1 = np.zeros(y0.shape)
    y1[0] = q0 + h*dTdp(p0)
    y1[1] = p0
    y1[2] = t0 + h
    return y1

def phi_U(h,y0):
    """
    Analytic evolution operator of the potential part.
    """
    q0,p0,t0 = y0
    y1 = np.zeros(y0.shape)
    y1[0] = q0
    y1[1] = p0 - h*dUdq(q0)
    y1[2] = t0
    return y1

def integrate_LTS(evA,evB,y0,T,N):
    """
    Integrates an ODE using Lie-Trotter-Splitting for two
    evolution operators evA and evB.
    y0: Initial conditions
    T: final time
    N: number of timesteps
    """
    t, h = np.linspace(0,T,N+1,retstep=True)
    y = np.zeros((N+1,)+y0.shape)
    y[0,...] = y0
    for j in range(N):
        yA = evA(h,y[j,...])
        y[j+1,...] = evB(h,yA)
    
    return t, y

if __name__ == "__main__":
    y0 = np.array([0.,1.,0.])
    T = 10.
    N = 100

    def solution(t):
        """
        Analytical solution to the ODE.
        """
        return np.sin(t)

    t,y = integrate_LTS(phi_T,phi_U,y0,T,N)
    plt.plot(t,y[:,0],'.',label='Lie-Trotter')
    plt.plot(t,solution(t),'--',label='Analytical Solution')
    plt.legend(loc='best')
    plt.show()
