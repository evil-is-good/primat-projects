import scipy
import math


# d = 0.5
d = 0.05
# d = 1.0
# d = 0.05 * 8.0 / 3.0
# d = 0.033516
l = [16.0, 8.0, 8.0]
# l = [16.0, 16.0]
# l = [32.0]
# l = [8.0, 8.0]
# l = [1.0]
# l = [4.0, 4.0]
S = 83.138438
# S = 100.0
# S = 1.0
# S = 8.0 * math.sqrt(3.0) / 2.0
S0 = 81.538438
# S0 = S - d * l[0]
# S0 = 0.5

# print S, S0 + d * 32.0

I = scipy.eye(3, 3)
l0 = 0.030238
# l0 = 0.01
# l0 = 1.0
lk = 146.538
# lk = 393.1
# lk = 200.0
# lk = 2.0
L0 = scipy.matrix([[l0, 0.0, 0.0],
                   [0.0, l0, 0.0],
                   [0.0, 0.0, l0]])
Lk = scipy.matrix([[lk, 0.0, 0.0],
                   [0.0, lk, 0.0],
                   [0.0, 0.0, lk]])
# fi = [0.0, math.pi / 3.0 * 2.0, math.pi / 3.0]
fi = [math.pi / 2.0, math.pi / 6.0, math.pi / 6.0 * 5.0]
# fi = [math.pi / 2.0, math.pi / 6.0, - math.pi / 6.0]
# fi = [math.pi / 2.0, 0.0]
# fi = [math.pi / 6.0, math.pi / 6.0 * 5.0]
# fi = [math.pi / 6.0 * 5.0]
# fi = [0.0, 0.0, 0.0]
# fi = [0.0]
# fi = [math.pi / 2.0, math.pi / 2.0, math.pi / 2.0]
# fi = [math.pi / 2.0]
# fi = [0.0, math.pi / 2.0]


def static():
    tmp = scipy.zeros((3, 3), float)

    for i in range(0, len(l)):
        L1 = L0[0, 0] / Lk[1, 1]
        L2 = L0[1, 1] / Lk[1, 1]
        D = scipy.matrix([[math.cos(fi[i]), - math.sin(fi[i]), 0.0],
                          [math.sin(fi[i]),   math.cos(fi[i]), 0.0],
                          [0.0,               0.0,             1.0]])
        B = scipy.matrix([[math.cos(fi[i]),        math.sin(fi[i]),      0.0],
                          [- L1 * math.sin(fi[i]), L2 * math.cos(fi[i]), 0.0],
                          [0.0,                    0.0,                  1.0]])
        tmp += l[i] * D * B

    E = S0 / S * I + tmp * d / S
    E = scipy.matrix(E).I

    tmp = scipy.zeros((3, 3), float)

    for i in range(0, len(l)):
        L1 = L0[0, 0] / Lk[1, 1]
        L2 = L0[1, 1] / Lk[1, 1]
        B = scipy.matrix([[math.cos(fi[i]),        math.sin(fi[i]),      0.0],
                          [- L1 * math.sin(fi[i]), L2 * math.cos(fi[i]), 0.0],
                          [0.0,                    0.0,                  1.0]])
        tmp += l[i] * B.T * Lk * B

    L = E.T * (S0 / S * L0 + d / S * tmp) * E

    print L[0, 0], L[1, 1], L[2, 2]

    # print L

    # print (L0[0, 0] + Lk[0, 0]) / 2.0, 2.0 / (1.0 / L0[0, 0] + 1.0 / Lk[0, 0])
    # print (L0[0, 0] * S0 + Lk[0, 0] * (S - S0)) / S

    # print scipy.matrix([[1.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]]).I


def dinamic():
    X0 = L0.I
    Xk = Lk.I
    A = Xk
    A[1, 1] = 1.0

    tmp = scipy.zeros((3, 3), float)

    for i in range(0, len(l)):
        c = math.cos(fi[i])
        s = math.sin(fi[i])
        D = scipy.matrix([[c, - s,   0.0],
                          [s,   c,   0.0],
                          [0.0, 0.0, 1.0]])
        F = scipy.matrix([[X0[0, 0] * c + X0[1, 0] * s, X0[0, 1] * c + X0[1, 1] * s, X0[0, 2] * c + X0[1, 2] * s],
                          [- s,                         c,                           0.0                        ],
                          [X0[2, 0],                    X0[2, 1],                    X0[2, 2]                   ]])
        C = A.I * F
        tmp += l[i] * D * C

    P = S0 / S * I + tmp * d / S
    P = scipy.matrix(P).I

    tmp = scipy.zeros((3, 3), float)

    for i in range(0, len(l)):
        c = math.cos(fi[i])
        s = math.sin(fi[i])
        F = scipy.matrix([[X0[0, 0] * c + X0[1, 0] * s, X0[0, 1] * c + X0[1, 1] * s, X0[0, 2] * c + X0[1, 2] * s],
                          [-s,                          c,                           0.0                        ],
                          [X0[2, 0],                    X0[2, 1],                    X0[2, 2]                   ]])
        C = A.I * F
        tmp += l[i] * C.T * Xk * C

    X = P.T * (S0 / S * X0 + d / S * tmp) * P

    L = X.I

    print L[0, 0], L[1, 1], L[2, 2]

    # print " "
    # print L

    # print (L0[0, 0] + Lk[0, 0]) / 2.0, 2.0 / (1.0 / L0[0, 0] + 1.0 / Lk[0, 0])
    # print (L0[0, 0] * S0 + Lk[0, 0] * (S - S0)) / S

    # print scipy.matrix([[1.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]]).I

# static()
# dinamic()


def jstatic(total_area, share_include):
    S = total_area
    S0 = S * share_include
    d = (S - S0) / 32.0

    tmp = scipy.zeros((3, 3), float)

    for i in range(0, len(l)):
        L1 = L0[0, 0] / Lk[1, 1]
        L2 = L0[1, 1] / Lk[1, 1]
        c = math.cos(fi[i])
        s = math.sin(fi[i])
        D = scipy.matrix([[c, - s,   0.0],
                          [s,   c,   0.0],
                          [0.0, 0.0, 1.0]])
        B = scipy.matrix([[c,        s,      0.0],
                          [- L1 * s, L2 * c, 0.0],
                          [0.0,      0.0,    1.0]])
        # print (D * B)[0, 0], (c * c + (-s) * (-s) * L1)
        tmp += l[i] * D * B
    # print tmp[0, 0]

    E = S0 / S * I + tmp * d / S
    # print E
    E = scipy.matrix(E).I

    tmp = scipy.zeros((3, 3), float)

    for i in range(0, len(l)):
        L1 = L0[0, 0] / Lk[1, 1]
        L2 = L0[1, 1] / Lk[1, 1]
        B = scipy.matrix([[math.cos(fi[i]),        math.sin(fi[i]),      0.0],
                          [- L1 * math.sin(fi[i]), L2 * math.cos(fi[i]), 0.0],
                          [0.0,                    0.0,                  1.0]])
        tmp += l[i] * B.T * Lk * B

    L = E.T * (S0 / S * L0 + d / S * tmp) * E

    return L[0, 0], L[1, 1], L[2, 2], (S0*l0 + (S-S0)*lk) / S, S, S0


def jkinematic(total_area, share_include):
    S = total_area
    S0 = S * share_include
    d = (S - S0) / 32.0

    X0 = L0.I
    Xk = Lk.I
    A = Xk
    A[1, 1] = 1.0

    tmp = scipy.zeros((3, 3), float)

    for i in range(0, len(l)):
        c = math.cos(fi[i])
        s = math.sin(fi[i])
        D = scipy.matrix([[c, - s,   0.0],
                          [s,   c,   0.0],
                          [0.0, 0.0, 1.0]])
        F = scipy.matrix([[X0[0, 0] * c + X0[1, 0] * s, X0[0, 1] * c + X0[1, 1] * s, X0[0, 2] * c + X0[1, 2] * s],
                          [- s,                         c,                           0.0                        ],
                          [X0[2, 0],                    X0[2, 1],                    X0[2, 2]                   ]])
        C = A.I * F
        tmp += l[i] * D * C

    P = S0 / S * I + tmp * d / S
    P = scipy.matrix(P).I

    tmp = scipy.zeros((3, 3), float)

    for i in range(0, len(l)):
        c = math.cos(fi[i])
        s = math.sin(fi[i])
        F = scipy.matrix([[X0[0, 0] * c + X0[1, 0] * s, X0[0, 1] * c + X0[1, 1] * s, X0[0, 2] * c + X0[1, 2] * s],
                          [-s,                          c,                           0.0                        ],
                          [X0[2, 0],                    X0[2, 1],                    X0[2, 2]                   ]])
        C = A.I * F
        tmp += l[i] * C.T * Xk * C

    X = P.T * (S0 / S * X0 + d / S * tmp) * P

    L = X.I

    return L[0, 0], L[1, 1], L[2, 2]

# print jstatic(83.138438, 81.538438 / 83.138438)
# print jkinematic(83.138438, 81.538438 / 83.138438)
# print jstatic(10.0, 0.0)

# print jstatic(100.0, 92.0 / 100.0)
# print jkinematic(100.0, 92.0 / 100.0)

# f = open('jancovsky_cuprum.gpd', 'w')
f = open('jancovsky.gpd', 'w')

share = 0.01
for i in range(99):
    stat = jstatic(83.138438, share)
    kine = jkinematic(83.138438, share)
    f.write("{0} {1} {2} {3} {4} {5} {6} {7}\n".format(
         1.0 - share,
         stat[0],
         stat[1],
         kine[0],
         kine[1],
         (83.138438 * (1.0 - share)) / 32.0,
         stat[2],
         (l0 * share + lk * (1.0 - share))))
    share += 0.01

f.close()

