# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from splitting_parameters import splitting_parameters


def integrate(method, y0, t_end, n_steps):
    t, dt = np.linspace(0, t_end, n_steps + 1, retstep=True)
    y = np.empty((n_steps + 1,) + y0.shape)
    y[0, ...] = y0

    for i in range(n_steps):
        y[i + 1, ...] = method(y[i, ...], dt)

    return t, y

def


def plot_angle(t, all_y, all_labels):
    """Plottet den Winkel als Funktion der Zeit.

    t:          Zeitpunkte an denen die Lösung approximiert wurde.
    all_y:      Eine Liste aller Lösungen die geplottet werden sollen.
    all_labels: Eine Liste der Labels der entsprechenden Approximationen.
    """
    fig = plt.figure(figsize=(20, 10))
    plt.title("Auslenkung")

    for y, label in zip(all_y, all_labels):
        plt.plot(t, y[:, 0], label=label)

    plt.grid()
    plt.legend()
    return fig


def plot_phasespace(all_y, all_labels):
    """Plottet

    all_y:      Eine Liste aller Lösungen die geplottet werden sollen.
    all_labels: Eine Liste der Labels der entsprechenden Approximationen.
    """
    fig = plt.figure(figsize=(20, 10))
    plt.title("Phasenraum")

    for y, label in zip(all_y, all_labels):
        plt.plot(y[:, 0], y[:, 1], label=label)

    plt.grid()
    plt.legend()

    return fig



# TODO vervollständigen Sie das Template.

