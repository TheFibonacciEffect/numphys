# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

# Wir wollen
# dy/dt = A(t) y + b*y
# mittels Splitting lösen. Dabei ist A eine Rotationsmatrix und b
# eine reelle Zahl.

# Die Lösung ist also eine Komposition von Rotation
# und Streckung von y.

b = -0.1
theta = 0.25 * np.pi


def rhs(t, y):
    return np.dot(dRdt(t), np.dot(invR(t), y)) + b * y


def R(t):
    angle = theta * t
    A = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return A


def invR(t):
    return R(-t)


def dRdt(t):
    angle = theta * t
    A = theta * np.array(
        [[-np.sin(angle), -np.cos(angle)], [np.cos(angle), -np.sin(angle)]]
    )

    return A



# TODO vervollständigen Sie das Template.

def integrate(method, y0, t_end, n_steps):
    """Führt wiedeholt Zeitschritte des Verfahrens `method` aus.

    Input:
        method : Einschrittverfahren mit Signatur `(y[i,:], t, dt)`.
    """

    t, dt = np.linspace(0, t_end, n_steps + 1, retstep=True)
    y = np.empty((n_steps + 1,) + y0.shape)
    y[0, ...] = y0

    for i in range(n_steps):
        y[i + 1, ...] = method(y[i, ...], dt)

    return t, y



# TODO vervollständigen Sie das Template.



def reference_solution(rhs, t_end, y0, n_steps):
    # Wir wollen die Lösung an folgenden Punkten
    t_eval = np.linspace(0.0, t_end, n_steps + 1)

    soln = scipy.integrate.solve_ivp(rhs, (0, t_end), y0, t_eval=t_eval)
    return soln.t, soln.y.transpose()



# TODO vervollständigen Sie das Template.

