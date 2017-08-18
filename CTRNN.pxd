import numpy as np
cimport numpy as np

cdef class CTRNN:
    cdef int N
    cdef double step_size, sigma
    cdef np.ndarray Tau, W, Theta, I, G, Y, dy_dt
    cdef object tau_range, g_range, theta_range, w_range

cdef center_cross(np.ndarray weights)
cdef sigmoid(np.ndarray x)
