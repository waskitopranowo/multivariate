import numpy as np
from scipy.interpolate import interp1d

def x_ecdf(x):
    # data transformation by using empirical CDF. The result are ranging between 0 and 1
    eps = 7./3 - 4./3 - 1 # machine epsilon

    xsort = np.sort(x)
    freq = np.linspace(1, len(xsort), len(xsort))/len(xsort)

    f = interp1d(xsort, freq)
    xnorm = f(x) - eps
    return xnorm

def cop_syn_m(typec, u, v, m):
    # Archimedian Copula Function
    # typec is type of copula (clayton, gumbel, frank)
    # m is copula parameter

    u = np.asarray([u]).T @ np.ones((1, m.shape[1]))
    v = np.asarray([v]).T @ np.ones((1, m.shape[1]))


    m1 = np.ones((u.shape[0], 1)) @ np.asarray([m[0,:]])
    m2 = np.ones((u.shape[0], 1)) @ np.asarray([m[1,:]])
    m3 = np.ones((u.shape[0], 1)) @ np.asarray([m[2,:]])

    if typec == 'gumbel':
        C = np.exp(-((-m2*np.log(u))**m1 + (-m3*np.log(v))**m1)**(1/m1))
    elif typec == 'clayton':
        C = (u**(-m1*m2) + v**(-m1*m3) - 1)**(-1/m1)
    elif typec == 'frank':
        p1 = (1 - np.exp(m1))**(-1)
        p2 = 1 - np.exp(m1*(u**m2))
        p3 = 1 - np.exp(m1*(v**m3))
        C = 1/ m1*np.log(1 - p1*p2*p3)
    return C

type1 = 'gumbel'
type2 = 'clayton'

xy = np.loadtxt('data copula man.txt', skiprows=0)
x1 = xy[:, 0]
x2 = xy[:, 2]

mingam1 = 1
maxgam1 = 10
mingam2 = 0.001
maxgam2 = 10
nm = 50  # number of particles
Nit = 30  # number of iteration
w = 2
c1 = 2.05
c2 = 2.05
phi = c1+c2
k = 2/(2 - phi - np.sqrt(phi**2-4*phi))

u = x_ecdf(x1)
v = x_ecdf(x2)

Gam1 = np.random.rand(nm)*(maxgam1 - mingam1) + mingam1
Gam2 = np.random.rand(nm)*(maxgam2 - mingam2) + mingam2
Th11 = np.random.rand(nm)
Th12 = np.random.rand(nm)
type = [type1, type2]

mm = np.vstack((Gam1, Th11, Th12, Gam2))

eps = 7./3 - 4./3 -1
du = 0.001
dv = 0.001

up = u + du
up[up > 1] = 1
vp = v + dv
vp[vp > 1] = 1
ud = u - du
ud[ud <= 0] = eps
vd = v - dv
vd[vd <= 0] = eps

C1 = cop_syn_m(type1, up, vp,
              np.asarray([mm[0, :], mm[1, :], mm[2, :]]))*cop_syn_m(type2, up, vp, np.asarray([mm[3, :], 1-mm[1, :], 1-mm[2, :]]))
C2 = cop_syn_m(type1, up, vd,
              np.asarray([mm[0, :], mm[1, :], mm[2, :]]))*cop_syn_m(type2, up, vd, np.asarray([mm[3, :], 1-mm[1, :], 1-mm[2, :]]))
C3 = cop_syn_m(type1, ud, vp,
              np.asarray([mm[0, :], mm[1, :], mm[2, :]]))*cop_syn_m(type2, ud, vp, np.asarray([mm[3, :], 1-mm[1, :], 1-mm[2, :]]))
C4 = cop_syn_m(type1, ud, vd,
              np.asarray([mm[0, :], mm[1, :], mm[2, :]]))*cop_syn_m(type2, ud, vd, np.asarray([mm[3, :], 1-mm[1, :], 1-mm[2, :]]))
c = (C1.real - C2.real - C3.real + C4.real)/(4*du*dv)
logC = np.log(c + 10**-7)
L = np.sum(logC, axis=0)
Lbest = L

zonemax = np.vstack((maxgam1, 1, 1, maxgam2)) @ np.ones((1, nm))
zonemin = np.vstack((mingam1, 0, 0, mingam2)) @ np.ones((1, nm))

pBest = mm
gBest = mm[:, Lbest == max(Lbest)]
vi = (np.random.rand(mm.shape[0], mm.shape[1])-0.5)*2

for i in range(Nit):
    r1 = np.random.rand(1)
    r2 = np.random.rand(1)

    vi = k * (vi + r1 * c1 * (pBest - mm) + r2 * c2 * (gBest @ np.ones((1, mm.shape[1])) - mm))

    mmn = mm + vi
    mmn[mmn > zonemax] = mm[mmn > zonemax]
    mmn[mmn < zonemin] = mm[mmn < zonemin]
    C1 = cop_syn_m(type1, up, vp,
                   np.asarray([mmn[0, :], mmn[1, :], mmn[2, :]])) * cop_syn_m(type2, up, vp, np.asarray([mmn[3, :], 1 - mmn[1, :], 1 - mmn[2, :]]))
    C2 = cop_syn_m(type1, up, vd,
                   np.asarray([mmn[0, :], mmn[1, :], mmn[2, :]])) * cop_syn_m(type2, up, vd, np.asarray([mmn[3, :], 1 - mmn[1, :], 1 - mmn[2, :]]))
    C3 = cop_syn_m(type1, ud, vp,
                   np.asarray([mmn[0, :], mmn[1, :], mmn[2, :]])) * cop_syn_m(type2, ud, vp, np.asarray([mmn[3, :], 1 - mmn[1, :], 1 - mmn[2, :]]))
    C4 = cop_syn_m(type1, ud, vd,
                   np.asarray([mmn[0, :], mmn[1, :], mmn[2, :]])) * cop_syn_m(type2, ud, vd, np.asarray([mmn[3, :], 1 - mmn[1, :], 1 - mmn[2, :]]))
    c = (C1.real - C2.real - C3.real + C4.real) / (4 * du * dv)
    logC = np.log(c + 10**-7)
    L = np.sum(logC, axis=0)

    mm = pBest

    pBest[:, L > Lbest] = mmn[:, L > Lbest]
    Lbest[L > Lbest] = L[L > Lbest]
    n = np.where(Lbest == np.max(Lbest))[0]
    ni = int(np.floor(np.random.rand(1)*(len(n))))
    n = n[ni]
    gBest = np.asarray([pBest[:, int(n)]]).T

parameter = np.vstack((gBest, 1-gBest[1], 1-gBest[2]))
print(parameter)


