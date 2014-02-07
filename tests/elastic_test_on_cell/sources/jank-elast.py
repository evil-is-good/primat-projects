import scipy
import math
from math import cos
from math import sin

def D(phi):
    tmp = scipy.zeros((6, 6))
    tmp[0, 0] = tmp[1, 1] = cos(phi)**2.0
    tmp[0, 1] = tmp[1, 0] = sin(phi)**2.0
    tmp[1, 3] = -sin(2.0 * phi)
    tmp[0, 3] = -tmp[1, 3]
    tmp[2, 2] = 1.0
    tmp[3, 1] = 0.5 * sin(2.0 * phi)
    tmp[3, 0] = -tmp[3, 1]
    tmp[3, 3] = cos(2.0 * phi)
    tmp[4, 4] = tmp[5, 5] = cos(phi)
    tmp[5, 4] = sin(phi)
    tmp[4, 5] = -tmp[5, 4]

    return tmp

def make_A(E, nu):
    A = scipy.zeros((6, 6))

    A[0, 0] = 1.0 / E
    A[1, 1] = 1.0 / E
    A[2, 2] = 1.0 / E

    A[0, 1] = - nu / E
    A[0, 2] = - nu / E
    A[1, 0] = - nu / E
    A[1, 2] = - nu / E
    A[2, 0] = - nu / E
    A[2, 1] = - nu / E

    A[3, 3] = (1 + nu) / E
    A[4, 4] = (1 + nu) / E
    A[5, 5] = (1 + nu) / E

    return A

def make_G(E, nu):
    G = scipy.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            G[i, j] = (1 + nu) / E
    return G

def make_B_static(A0, Ak, G0, Gk, phi):
    x = 0
    y = 1
    z = 2

    tmp = scipy.zeros((6,6))
    tmp[0, 0] = cos(phi)**2.0
    tmp[0, 1] = sin(phi)**2.0
    tmp[0, 3] = sin(2.0 * phi)
    tmp[1, 0] = (A0[0, 0] * sin(phi)**2.0 + (A0[1, 0] - Ak[1, 0]) * cos(phi)**2.0) / Ak[1, 1]
    tmp[1, 1] = (A0[1, 1] * cos(phi)**2.0 + (A0[1, 0] - Ak[1, 0]) * sin(phi)**2.0) / Ak[1, 1]
    tmp[1, 2] = (A0[0, 2] * sin(phi)**2.0 + A0[1,2] * cos(phi)**2.0 - Ak[1, 2]) / Ak[1, 1]
    tmp[1, 3] = -sin(2.0 * phi) * (2.0 * G0[0, 1] + Ak[1, 0]) / Ak[1, 1]
    tmp[2, 2] = 1.0
    tmp[3, 0] = sin(2.0 * phi) * (A0[1, 0] - A0[0, 0]) / (4.0 * Gk[x, y])
    tmp[3, 1] = sin(2.0 * phi) * (A0[1, 1] - A0[0, 1]) / (4.0 * Gk[x, y])
    tmp[3, 2] = sin(2.0 * phi) * (A0[1, 2] - A0[0, 2]) / (4.0 * Gk[x, y])
    tmp[3, 3] = cos(2.0 * phi) * G0[0, 1] / Gk[x, y]
    tmp[4, 4] = cos(phi) * G0[1, 2] / Gk[y, z]
    tmp[4, 5] = -sin(phi) * G0[0, 2] / Gk[y, z]
    tmp[5, 4] = sin(phi)
    tmp[5, 5] = cos(phi)

    return tmp

def static(OMEGA, omega, YOUNG, POISSON, young, poisson, phi):

    n = len(omega)

    A0 = make_A(YOUNG, POISSON)
    a = [make_A(young, poisson) for i in range(n)]

    G0 = make_G(YOUNG, POISSON)
    G = [make_G(young, poisson) for i in range(n)]

    I = scipy.eye(6, 6)

    B = [make_B_static(A0, a[k], G0, G[k], phi[k]) for k in range(n)]

    E = scipy.matrix(OMEGA * I + sum([omega[k] * D(phi[k]) * B[k] for k in range(n)])).I

    A = E.T * (OMEGA * A0 + sum([omega[k] * B[k].T * a[k] * B[k] for k in range(n)])) * E

    return A

# delta = 0.005
# 
# V = 1.0
# V0 = (1.0 - delta)**2.0
# Va = V - V0
# 
# OMEGA = V0 / V
# 
# omega = [(1.0 * delta) / V, ((1.0 - delta) * delta) / V]
# 
# phi = [0.0, math.pi / 2.0]
# 
# A = static(OMEGA, omega, 40.0, 0.35, 393.0, 0.4, phi)

V = 1.0
V0 = 1.0 * 0.5
Va = V - V0

OMEGA = V0 / V

omega = [Va / V]

phi = [0.0]

E0 = 40.0
E1 = 393.0

A = static(OMEGA, omega, E0, 0.35, E1, 0.4, phi)


print A
print 1.0 / A[0,0], 1.0 / A[1, 1]
print (E0 + E1) / 2.0, (2.0*E0*E1) / (E1 + E0)
