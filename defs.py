# imports
import numpy as np
import sys
import pandas as pd

def mco_x2el(mu,x,y,z,u,v,w):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

          MCO_X2EL.FOR    (ErikSoft  20 February 2001)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     Author: John E. Chambers

     Calculates Keplerian orbital elements given relative coordinates and
     velocities, and GM = G times the sum of the masses.

     The elements are: q = perihelion distance
                       e = eccentricity
                       i = inclination
                       p = longitude of perihelion (NOT argument of perihelion!!)
                       n = longitude of ascending node
                       l = mean anomaly (or mean longitude if e < 1.e-8)

    ------------------------------------------------------------------------------    
    """

    NMAX = 2000
    CMAX = 50
    NMESS = 200
    HUGE = 9.9e29
    PI = np.pi
    TWOPI = 2*PI
    PIBY2 = PI*0.5
    DR = PI/180.
    K2 = 2.959122082855911e-4
    AU = 1.4959787e13
    MSUN = 1.9891e33

    hx = y * w  -  z * v
    hy = z * u  -  x * w
    hz = x * v  -  y * u
    h2 = hx*hx + hy*hy + hz*hz
    v2 = u * u  +  v * v  +  w * w
    rv = x * u  +  y * v  +  z * w
    r = np.sqrt(x*x + y*y + z*z)
    h = np.sqrt(h2)
    s = h2 / mu

    # inclination and node
    ci = hz/h
    if (np.abs(ci) < 1):
        i = np.arccos(ci)
        n = np.arctan2(hx,-hy)
        if (n < 0):
            n = n + TWOPI
    else:
        if (ci > 0):
            i = 0.
        if (ci < 0):
            i = PI
        n = 0.


    # eccentricity and perihelion distance
    temp = 1. + s*(v2/mu - 2./r)
    if (temp <= 0):
        e = 0.
    else:
        e = np.sqrt(temp)
    q = s/(1.+e)

    # true longitude
    if (hy != 0):
        to = -hx/hy
        temp = (1. - ci)*to
        tmp2 = to*to
        true = np.arctan2((y*(1.+tmp2*ci)-x*temp),(x*(tmp2+ci)-y*temp))
    else:
        true = np.arctan2(y * ci, x)
    if (ci < 0):
        true = true + PI
    if (e < 1e-8):
        p = 0.
        l = true
    else:
        ce = (v2*r - mu) / (e*mu)
        # mean anomaly for ellipse
        if (e < 1):
            if (np.abs(ce) > 1):
                ce = np.sign(ce)
            bige = np.arccos(ce)
            if (rv < 0):
                bige = TWOPI - bige
            l = bige - e*np.sin(bige)
        # mean anomaly for hyperbola
        else:
            if (ce < 1):
                ce = 1.
            bige = np.log(ce + np.sqrt(ce*ce-1.))
            if (rv < 0):
                bige = TWOPI - bige
            l = e*np.sinh(bige) - bige
        # longitude of perihelion
        cf = (s-r)/(e*r)
        if (np.abs(cf) > 1):
            cf = np.sign(cf)
        f = np.arccos(cf)
        if (rv < 0):
            f = TWOPI - f
        p = true - f
        p = (p + TWOPI + TWOPI)%(TWOPI)

    if (l < 0):
        l = l + TWOPI
    if (l > TWOPI):
        l = l%TWOPI

    return q,e,i,p,n,l

def mco_kep(e,oldl):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	  MCO_KEP.FOR    (ErikSoft  7 July 1999)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     Author: John E. Chambers

     Solves Kepler's equation for eccentricities less than one.
     Algorithm from A. Nijenhuis (1991) Cel. Mech. Dyn. Astron. 51, 319-330.

      e = eccentricity
      l = mean anomaly      (radians)
      u = eccentric anomaly (   "   )

    ------------------------------------------------------------------------------
    """
    pi = np.pi
    twopi = 2*pi
    piby2 = pi/2.
    # Reduce mean anomaly to lie in the range 0 < l < pi
    if (oldl >= 0):
        l = oldl%twopi
    else:
        l = oldl%twopi + twopi
    sign = 1.
    if (l > pi):
        l = twopi - l
        sign = -1.
    
    ome = 1. - e
    if ((l >= 0.45) or (e < 0.55)):
        # Regions A,B or C in Nijenhuis
        # -----------------------------
        # Rough starting value for eccentric anomaly
        if (l < ome):
            u1 = ome
        else:
            if (l > (pi-1.-e)):
                u1 = (l+e*pi)/(1.+e)
            else:
                u1 = l + e
        # Improved value using Halley's method
        flag = (u1 > piby2)
        if flag:
            x = pi - u1
        else:
            x = u1
        x2 = x**2
        sn = x*(1. + x2*(-0.16605 + x2*0.00761))
        dsn = 1. + x2*(-.49815 + x2*.03805)
        if flag: dsn = -dsn
        f2 = e*sn
        f0 = u1 - f2 - l
        f1 = 1. - e*dsn
        u2 = u1 - f0/(f1 - .5*f0*f2/f1)
    else:
        # Region D in Nijenhuis
        # ---------------------
        # Rough starting value for eccentric anomaly
        z1 = 4.*e + 0.5
        p = ome/z1
        q = 0.5*l/z1
        p2 = p**2
        z2 = np.exp(np.log(np.sqrt(p2*p + q**2) + q)/1.5)
        u1 = 2.*q/(z2 + p + p2/z2)
        # Improved value using Newton's method
        z2 = u1**2
        z3 = z2**2
        u2 = u1 - .075*u1*z3 / (ome + z1*z2 + .375*z3)
        u2 = l + e*u2*(3. - 4.*u2*u2)
    # Accurate value using 3rd-order version of Newton's method
    # N.B. Keep cos(u2) rather than sqrt( 1-sin^2(u2) ) to maintain accuracy!
    # First get accurate values for u2 - sin(u2) and 1 - cos(u2)
    bigg = (u2 > piby2)
    if bigg:
        z3 = pi - u2
    else:
        z3 = u2

    big = (z3 > (0.5*piby2))
    if big:
        x = piby2 - z3
    else:
        x = z3
    
    x2 = x**2
    ss = 1.
    cc = 1.
    
    ss = x*x2/6.*(1. - x2/20.*(1. - x2/42.*(1. - x2/72.*(1. - \
       x2/110.*(1. - x2/156.*(1. - x2/210.*(1. - x2/272.)))))))
    cc = x2/2.*(1. - x2/12.*(1. - x2/30.*(1. - x2/56.*(1. - \
       x2/ 90.*(1. - x2/132.*(1. - x2/182.*(1. - x2/240.*(1. - x2/306.))))))))

    if big:
        z1 = cc + z3 - 1.
        z2 = ss + z3 + 1. - piby2
    else:
        z1 = ss
        z2 = cc

    if bigg:
        z1 = 2.*u2 + z1 - pi
        z2 = 2. - z2
    
    f0 = l - u2*ome - e*z1
    f1 = ome + e*z2
    f2 = .5*e*(u2-z1)
    f3 = e/6.*(1.-z2)
    z1 = f0/f1
    z2 = f0/(f2*z1+f1)
    mco_kep_ans = sign*( u2 + f0/((f3*z1+f2)*z2+f1) )

    return mco_kep_ans

def mco_el2x(mu,q,e,i,p,n,l):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

          MCO_EL2X.FOR    (ErikSoft  7 July 1999)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     Author: John E. Chambers

     Calculates Cartesian coordinates and velocities given Keplerian orbital
     elements (for elliptical, parabolic or hyperbolic orbits).

     Based on a routine from Levison and Duncan's SWIFT integrator.

      mu = grav const * (central + secondary mass)
      q = perihelion distance
      e = eccentricity
      i = inclination                 )
      p = longitude of perihelion !!! )   in
      n = longitude of ascending node ) radians
      l = mean anomaly                )

      x,y,z = Cartesian positions  ( units the same as a )
      u,v,w =     "     velocities ( units the same as sqrt(mu/a) )

    ------------------------------------------------------------------------------
    """
    # Change from longitude of perihelion to argument of perihelion
    g = p - n
    # Rotation factors
    si,ci = np.sin(i),np.cos(i)
    sg,cg = np.sin(g),np.cos(g)
    sn,cn = np.sin(n),np.cos(n)
    z1 = cg*cn
    z2 = cg*sn
    z3 = sg*cn
    z4 = sg*sn
    d11 = z1 - z4*ci
    d12 = z2 + z3*ci
    d13 = sg*si
    d21 = -z3 - z2*ci
    d22 = -z4 + z1*ci
    d23 = cg*si
    # Semi-major axis
    a = q/(1.-e)
    # Ellipse
    # TODO: add calculation for parabola and hyperbola
    if (e < 1.):
        romes = np.sqrt(1. - e**2)
        temp = mco_kep(e,l)
        se,ce = np.sin(temp),np.cos(temp)
        z1 = a * (ce - e)
        z2 = a * romes * se
        temp = np.sqrt(mu/a)/(1.-e*ce)
        z3 = -se*temp
        z4 = romes * ce * temp
    else:
        print 'eccentricity >= 1.'
        sys.exit()

    x = d11*z1 + d21*z2
    y = d12*z1 + d22*z2
    z = d13*z1 + d23*z2
    u = d11*z3 + d21*z4
    v = d12*z3 + d22*z4
    w = d13*z3 + d23*z4

    return x,y,z,u,v,w

#######################################################
# dont change these values
cf = 0.5 # courant safety factor
istep = 1
ie = 0 #-1 # energy equation index: 1 = gamma law, 0 = polytrope, -1 local isothermal (only 0 implemented)
pK = 1. #0.25 # polytropic constant
vturb = 0 # 0 for off, 1 for on
npert = 0 # number of active turbulet modes
amplitude = 5e-4 # amplitude of turbulent potential
bts = 0.00 # bulk to shear viscosity coefficients rati for disk
vnu = 1e-5 # kinematic viscosity coefficient
htr = 0.1 # height to radius ratio for initial soundspeed distribution
#######################################################



from parameters import *
n1 = 1
n2 = nr
n3 = nphi
# grid
x2a,x2b = np.zeros(n2+4),np.zeros(n2+4)
x3a,x3b = np.zeros(n3+4),np.zeros(n3+4)
dx2a,dx2b = np.zeros(n2+4),np.zeros(n2+4)
dx3a,dx3b = np.zeros(n3+4),np.zeros(n3+4)
dv = np.zeros((n2+3,n3+3)) # volume element array
#
v3orig = np.zeros(n3+4) #
fext = np.zeros(n3+4) #
# variables
d = np.zeros((n2+4,n3+4))
v2 = np.zeros((n2+4,n3+4))
v3 = np.zeros((n2+4,n3+4))
phi = np.zeros((n2+4,n3+4))
phiT = np.zeros((n2+4,n3+4))
e = np.zeros((n2+4,n3+4))
cs = np.zeros((n2+4,n3+4))
xp = np.zeros((2,2))
vp = np.zeros((2,2))
# potential
Glm = np.zeros((n3*2,n2*2))
Mlm = np.zeros((n3*2,n2*2))
du = np.nan
dphi = np.nan
# ds
ds = np.zeros((n2+4, n3+4))
Fm = np.zeros((n2+4, n3+4))
vs = np.zeros((n2+4, n3+4))
fluxr = np.zeros(n3+4)
# turbulenceR
tstart = np.zeros(npert+1)
tfinish = np.zeros(npert+1)
phaseT = np.zeros(npert+1)
Amp = np.zeros(npert+1)
omegaT = np.zeros(npert+1)
# turbulenceI
jcent = np.zeros(npert+1).astype(int)
iactive = np.zeros(npert+1).astype(int)
na = np.zeros(n2+1).astype(int)

# bodies
rm = np.zeros(2)
rm[0] = mstar
rm[1] = mplanet
aplanet = dperi/(1-ecc)
bperiod = 2*np.pi*np.sqrt((aplanet**3)/np.sum(rm)) # period of inner planet

#---------------------------------------------------
# initial conditions
va1 = np.sum(rm) #+ 0.01 # mu: G*(M_star + M_planet)
va2 = dperi # q: perihelion distance
va3 = ecc # e: eccentricity
va4 = 0. # i: inclination (radians)
va5 = lperi # p: longitude of perihelion (radians)
va6 = 0. # n: longitude of ascending node (radians)
va7 = meananom #358./180. * np.pi # l: mean anomaly (radians)
xi,yi,zi,ui,vi,wi = mco_el2x(va1,va2,va3,va4,va5,va6,va7)
# setup xp
xp[0,0] = 0. # x of the star
xp[0,1] = 0. # y of the star
xp[1,0] = xi # x of planet
xp[1,1] = yi # y of planet
# setup vp
vp[0,0] = 0. # v_x of star
vp[0,1] = 0. # v_y of star
vp[1,0] = ui # v_x of planet
vp[1,1] = vi # v_y of planet
#---------------------------------------------------

# rk integrator vectors
tsmv = 1e20
xpf = np.zeros((2,2))
y = np.zeros(8)
yval = np.zeros((8,2))
xx = np.zeros(2)
sd = np.zeros(n2+4)

# local artificial viscosity arrays (store in q2,q3)
p = np.zeros((n2+4, n3+4))
q2 = np.zeros((n2+4, n3+4))
q3 = np.zeros((n2+4, n3+4))

# define grid radii
# set up z zones (evenly spaced)
dz = 1.
# set up r zones (spaced as ln(r))
lnr = np.logspace(np.log(xdmin),np.log(xdmax),nr+1,base=np.exp(1))
dlnr = np.diff(np.log(lnr))[0]
lnr = np.insert(lnr,0,np.exp((np.log(lnr)-dlnr)[0]))
lnr = np.insert(lnr,len(lnr),np.exp((np.log(lnr)+dlnr)[-1]))
lnr = np.insert(lnr,len(lnr),np.exp((np.log(lnr)-(2*dlnr))[1]))
x2a = lnr
"""
# set up r zones (spaced as r (linear))
J = np.ogrid[2:n2+1]
dr = (xdmax-xdmin)/n2
x2a[1] = xdmin
x2a[J] = xdmin + (J-1)*dr
x2a[0] = xdmin-dr
x2a[-1] = xdmin-(2*dr)
x2a[n2+1] = xdmax
x2a[n2+2] = xdmax+dr
"""
J = np.ogrid[-1:n2+2]
x2b[J] = (x2a[J+1] + x2a[J])/2.
dx2a[J] = x2a[J+1] - x2a[J]
J = np.ogrid[0:n2+3]
dx2b = np.zeros(n2+5)
dx2b[J] = x2b[J] - x2b[J-1]
# set up the phi zones
dphi = (2*np.pi)/n3
x3a[-1] = -2*dphi
x3a[0] = -dphi
x3a[1] = 0.
K = np.ogrid[2:n3+3]
x3a[K] = (K-1)*dphi
K = np.ogrid[-1:n3+2]
x3b[K] = (x3a[K+1] + x3a[K])/2.
dx3a[K] = x3a[K+1]-x3a[K]
K = np.ogrid[0:n3+2]
dx3b[K] = x3b[K]-x3b[K-1]

# compute volume element array
J,K = np.ogrid[1:n2+1,1:n3+1]
junk = (x2a[J+1]**2 - x2a[J]**2)
dv[J,K] = dz*(junk/2.)*dx3a[K]
#dv[1,1:n2+1,1:n3+1] = dz*(junk/2.)*dx3a[1:n3+1]

# sin and cos
sx3a = np.sin(x3a)
sx3b = np.sin(x3b)
cx3a = np.cos(x3a)
cx3b = np.cos(x3b)

# set ic
nstep = 1
nsnap = 1
time = 0.
dt = 1e-5
iout = 0

# compute equilibrium surface density (density) distribution
ghi = 1e3
glo = 0.
sz = (ghi+glo)/2.
for jcount in range(1,n2+1):
    for j in range(1,n2+1):
        sd[j] = sz/x2b[j]
    rmdt = 0.
    for j in range(1,n2+1):
        rmdt = rmdt + np.pi*(x2a[j+1]**2 - x2a[j]**2)*sd[j]
    if (rmdt > rmdisk):
        ghi = sz
        sz = (sz+glo)/2.
    if (rmdt < rmdisk):
        glo = sz
        sz = (sz+ghi)/2.
    if (np.abs(rmdt - rmdisk)/rmdt < 1e-6):
        break


J,K = np.ogrid[1:n2+1,1:n3+1]
d[J,K] = sd[J]
cond = (d[J,K] < 1e-12)
J,K = np.mgrid[1:n2+1,1:n3+1]
J,K = J[cond],K[cond]
d[J,K] = 1e-12

J,K = np.ogrid[1:n2+1,1:n3+1]
cs = np.sqrt(gamma*pK*d**(gamma-1))
ndj = np.where(x2b > rdj)[0][0]
ondj = np.where(x2b > ordj)[0][0]

#########################################################3
# disk surface density setup
#########################################################3
# testing # TODO: check this. should r be x2a[1:-2]?
r = x2a[1:-3]
deltar = x2a[2:-2]**2 - x2a[1:-3]**2
# make gaussian surface density profile
sig_gauss = 0.5
mu_gauss = 1.
J,K = np.ogrid[1:n2+1,1:n3+1]
d[J,K] = np.exp(-(x2a[J] - mu_gauss)**2 / (2.*sig_gauss**2)) # gaussian
sigma = d[1:-3,n3/2]
Mdisk = np.sum(sigma*deltar)*np.pi
# normalize surface density to get disk mass correct
d = d*rmdisk/Mdisk
Mdisk = np.sum(sigma*deltar)*np.pi
#########################################################3
# disk toomre Q setup
#########################################################3
# grab current toomre Q
sigma = np.mean(d[1:-3],axis=1)
Omega_kep = (rm[0]/r**3)**0.5 * r
soundspeed = np.mean(cs[1:-3],axis=1)
toomreQ = soundspeed*Omega_kep/(np.pi*sigma)
# recalibrate soundspeed and toomreQ
pK = pK*(Qmin/np.min(toomreQ))**2.
cs = np.sqrt(gamma*pK*d**(gamma-1))
soundspeed = np.mean(cs[1:-3],axis=1)
sigma = np.mean(d[1:-3],axis=1)
toomreQ = soundspeed*Omega_kep/(np.pi*sigma)
pK = pK*(Qmin/np.min(toomreQ))**2
cs = np.sqrt(gamma*pK*d**(gamma-1))
soundspeed = np.mean(cs[1:-3],axis=1)
toomreQ = soundspeed*Omega_kep/(np.pi*sigma)
# finally reset toomreQ to zeroval
Omega = np.mean(v3[1:-3],axis=1)
toomreQ = soundspeed*Omega/(np.pi*sigma)

#######################################################
# output file setup
#######################################################

# coordinates are fixed so write those to file
r_arr,phi_arr = x2a[1:n2+2],x3a[1:n3+2]
with open('output/coordinates_%s.out' %run_name,'w+') as foo:
    line = '# r phi\n'
    foo.write(line)
    foo.write(' '.join(map(str,r_arr))+'\n')
    foo.write(' '.join(map(str,phi_arr))+'\n')

# file for bodies
bodfname = 'output/bodies_%s.out' %run_name
bodf = open(bodfname,'w+')
header = '# x1 y1 x2 y2 x3 y3 '+\
         'vx1 vy1 vx2 vy2 vx3 vy3 \n'
bodf.write(header)
bodf.close()

# file for snapshots
h5file = 'output/snapshots_%s.h5' %run_name
store = pd.HDFStore(h5file)
store.close()

# file for timestep information
timefname = 'output/timesteps_%s.out' %run_name
timef = open(timefname,'w+')
header = '# istep time/bperiod \n'
timef.write(header)
timef.close()

# file for toomreQ profile
toomreQfname = 'output/toomreQ_%s.out' %run_name
toomref = open(toomreQfname,'w+')
header = '# radial profile of toomreQ \n'
toomref.write(header)
toomref.close()

