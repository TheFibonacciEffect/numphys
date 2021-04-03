# coding: utf-8

import numpy as np
import numpy.linalg

import matplotlib.pyplot as plt

from splitting_parameters import splitting_parameters


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

