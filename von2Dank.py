# imports
import sys
import pandas as pd
import numpy as np
np.random.seed(69)
from defs import *

#########################################################################
# 2D hydrodynamics code: cylindrical geometry
#
# includes: accretion or reflection radial boundary conditions
#           artificial viscosity
#           kinematic viscosity
#           internal energy equation
#           FFT 2D potential solver
#           runge-kutta few-body integrator
#########################################################################

def boundary():
    global d,e,v2,v3,phi
    # Full boundary condition routine (do all boundaries in one call)
    # Current conditions
    #----------------------------------------------------------------------
    # r direction: reflecting condition (inner), reflecting condition (outer)
    # phi direction: periodic conditions.
    #----------------------------------------------------------------------
    # Reflecting inner boundary condition:
    # For a reflective inner boundary condition, the zone centered
    # variables and the tangential components of velocity in the
    # ghost zones are set equal to the corresponding values of
    # their images among the active zones. The normal component
    # of the velocity is set to zero on the boundary and reflected
    # for the second ghost zone.
    jc = -1
    k = np.ogrid[-1:n3+3]
    for j in [-1,0]:
        jc = jc+1
        v2[j,k] = -v2[2-jc+1,k]
        v3[j,k] = v3[2-jc,k]
        d[j,k] = d[2-jc,k]
        e[j,k] = e[2-jc,k]
        phi[j,k] = phi[2-jc,k]
        if (j == 0): v2[j+1,k] = 0.
    # reflecting outer boundary condition
    jc = -1
    k = np.ogrid[-1:n3+3]
    for j in [n2+1,n2+2]:
        jc = jc+1
        v2[j,k] = -v2[n2-jc+1,k]
        v3[j,k] = v3[n2-jc,k]
        d[j,k] = d[n2-jc,k]
        e[j,k] = e[n2-jc,k]
        phi[j,k] = phi[n2-jc,k]
        if (j == n2+1): v2[j,k] = 0.
    # Phi condition
    j = np.ogrid[-1:n2+2]
    for k in [-1,0]:
        v2[j,k] = v2[j,n3+k]
        v3[j,k] = v3[j,n3+k]
        d[j,k] = d[j,n3+k]
        e[j,k] = e[j,n3+k]
        phi[j,k] = phi[j,n3+k]
    for k in [1,2]:
        v2[j,n3+k] = v2[j,k]
        v3[j,n3+k] = v3[j,k]
        d[j,n3+k] = d[j,k]
        e[j,n3+k] = e[j,k]
        phi[j,n3+k] = phi[j,k]

def timestep():
    global dt,time
    # Establish timestep criteria
    if ((istep%nprint==0)&verbose):
        print istep, '\t', '%.3f \t' %time
    dt = 0 
    dt1 = 1.e+20
    if (cv < 0):
        # no artificial viscosity in timestep limiter
        j,k = np.ogrid[1:n2+1,1:n3+1]
        if (ie < 0): csound = cs[j,k]
        elif (ie == 0): csound = np.sqrt(gamma*pK*d[j,k]**(gamma-1))
        else: csound = np.sqrt(gamma*(gamma-1)*e[j,k]/d[j,k])
        dt1 = dx2a[j]/csound
        sample = dx3a[k]*x2a[j]/csound
        if np.any(sample < dt1): dt1 = sample
        dt3 = dx2a[j]/np.max([np.max(np.abs(v2[j,k])),1e-16])
        dt4 = dx3a[k]*x2a[j]/np.max([np.max(np.abs(v3[j,k])),1e-16])
        sample = (1./dt1**2 + 1./dt3**2 + 1./dt4**2)
        # Find which zone limits timestep:
        sample = sample.max()
        if (sample > dt): dt = sample
        dt = cf/np.sqrt(dt)
    else:
        # include artificial viscosity in the timestep limiter
        j,k = np.ogrid[1:n2+1,1:n3+1]
        if np.any(d[j,k] < 0):
            print 'negative density zone:',np.where(d<0)
            sys.exit()
        if (ie < 0):
            csound = cs[j,k]
        elif (ie == 0):
            csound = np.sqrt(gamma*pK*d[j,k]**(gamma-1))
        else:
            csound = np.sqrt(gamma*(gamma-1)*e[j,k]/d[j,k])
        dt1 = np.min(dx2a[j]/csound)
        sample = np.min(dx3a[k]*x2a[j]/csound)
        if (sample < dt1):
            dt1 = sample
        dt3 = np.min(dx2a[j]/np.max([np.max(np.abs(v2[j,k])),1e-16]))
        dt4 = np.min(dx3a[k]*x2a[j]/np.max([np.max(np.abs(v3[j,k])),1e-16]))
        dt5 = np.min(dx2a[j]/(4*cv*np.max([np.max(np.abs(v2[j+1,k] - v2[j,k])),1e-16])))
        sample = np.min(dx3a[k]*x2a[j]/(4*cv*np.max([np.max(np.abs(v3[j,k+1] - v3[j,k])),1e-16])))
        if (sample < dt5):
            dt5 = sample
        sample = (1./dt1**2 + 1./dt3**2 + 1./dt4**2 + 1./dt5**2)
        # determine which zone limits timestep:
        if (sample > dt):
            dt = sample
        dt = cf/np.sqrt(dt)

    # viscous limitation on timestep
    if (dt > tsmv): dt = tsmv
    
    time += dt
    return time,dt

def central():
    global xp,xpf,vp,xcom,ycom,rmtot
    # Evaluate the center of mass of the system
    xp,xpf,vp = rkbody(xp,xpf,vp,dt)
    xcom = 0.
    ycom = 0.
    rmtot = 0.
    J,K = np.ogrid[1:n2+1,1:n3+1]
    rm_zone = d[J,K]*dv[J,K]
    rmtot = np.sum(rm_zone)
    junk = x2a[J]*np.cos(x3b[K])*rm_zone
    xcom = np.sum(junk)
    junk = x2a[J]*np.sin(x3b[K])*rm_zone
    ycom = np.sum(junk)
    xcom += np.sum(xp[:,0]*rm[:])
    ycom += np.sum(xp[:,1]*rm[:])

def x2iz(var):
    var[1,-1:n3+2+1] = 0.
    return var

def x2bi(var,i_add,i_sign):
    # routine implements reflecting inner boundary condition on standard array variable `var'
    var[-1,-1:n3+2+1] = i_sign*var[2-0+i_add,-1:n3+2+1]
    var[0,-1:n3+2+1] = i_sign*var[2-1+i_add,-1:n3+2+1]
    return var

def x2bo(var,i_add,i_sign):
    # routine implements reflecting inner boundary condition on standard array variable `var'
    var[n2+1,-1:n3+2+1] = i_sign*var[n2-0+i_add,-1:n3+2+1]
    var[n2+2,-1:n3+2+1] = i_sign*var[n2-1+i_add,-1:n3+2+1]
    return var

def x2oz(var):
    # routine zeros face-centered variables along inner boundary
    var[n2+1,-1:n3+2+1] = 0.
    return var

def x3bio(var):
    # routine implements periodic boundary condition on standard array variable `var'
    J = np.ogrid[-1:n2+2]
    var[J,-1] = var[J,-1+n3]
    var[J,0] = var[J,n3]
    var[J,n3+1] = var[J,1]
    var[J,n3+2] = var[J,2]
    return var

def viscosity():
    global v2,v3,tsmv
    # This routine computes viscous forces, and applies them to update the velocities
    
    #local viscous tensor components
    q22 = np.zeros((n2+4, n3+4))
    q33 = np.zeros((n2+4, n3+4))
    q23 = np.zeros((n2+4, n3+4))
    q32 = np.zeros((n2+4, n3+4))
    eta = np.zeros((n2+4, n3+4))

    #temporarily define q32 as (bts-1/3)*trace of q22 and q33
    J,K = np.ogrid[1:n2+1,1:n3+1]
    q22[J,K] = (v2[J+1,K]-v2[J,K])/(dx2a[J])  
    q33[J,K] = (1./x2b[J]) * (v3[J,K+1]-v3[J,K])/dx3a[K] \
            + 0.5*(v2[J+1,K]+v2[J,K])/x2b[J]
    q32[J,K] = (bts - 1/3.)*(q22[J,K] + q33[J,K])
    q23[J,K] = 0.5*((1./x2a[J])*((v2[J,K] - v2[J,K-1])/dx3b[K]) \
            + (v3[J,K] - v3[J-1,K])/dx2b[J] - 0.5*(v3[J,K] + v3[J-1,K])/x2a[J])
    
    eta[J,K] = vnu*d[J,K]
    J,K = np.ogrid[1:5+1,1:n3+1]
    eta[J,K] = eta[5,K]*(0.8-(5.-J)/5.)
    J,K = np.ogrid[n2-4:n2+1,1:n3+1]
    eta[J,K] = eta[n2-4,K]*((n2-J)/5.)

    eta = x2bi(eta,0,1)
    eta = x2bo(eta,0,1)
    eta = x3bio(eta)

    tsmv = 1.e+20
    J,K = np.ogrid[1:n2+1,1:n3+1]
    condition = (eta[J,K] > 0)
    J,K = np.mgrid[1:n2+1,1:n3+1]
    j,k = J[condition],K[condition]
    J,K = j,k
    sample = np.min([(dx3a[K]*x2b[J]),(dx2a[J])])**2/(vnu*2.)
    tsmv = np.min([sample,tsmv])

    # compute tensor components
    J,K = np.ogrid[1:n2+1,1:n3+1]
    q22[J,K]=eta[J,K]*(q22[J,K]+q32[J,K])
    q33[J,K]=eta[J,K]*(q33[J,K]+q32[J,K])
    eta_av = 0.25*(eta[J,K] + eta[J-1,K] + eta[J-1,K-1] + eta[J,K-1])
    q32[J,K]=eta_av*(1./x2a[J])*q23[J,K]
    q23[J,K]=eta_av*x2a[J]*q23[J,K]

    # call boundary conditions on the tensor components
    q22 = x2bi(q22,0,1)
    q22 = x2bo(q22,0,1)
    q22 = x3bio(q22)
    q33 = x2bi(q33,0,1)
    q33 = x2bo(q33,0,1)
    q33 = x3bio(q33)
    q23 = x2bi(q23,1,-1)
    q23 = x2iz(q23)
    q23 = x2bo(q23,1,-1)
    q23 = x2oz(q23)
    q23 = x3bio(q23)
    q32 = x2bi(q32,1,-1)
    q32 = x2iz(q32)
    q32 = x2bo(q32,1,-1)
    q32 = x2oz(q32)
    q32 = x3bio(q32)

    # update radial velocity as a result of viscous force/unit volume
    v2[J,K] = v2[J,K] + dt/(0.5*(d[J,K] + d[J-1,K]))* \
            ((x2b[J]*q22[J,K]-x2b[J-1]*q22[J-1,K])/(dx2b[J]*x2a[J])\
            -.5*(q33[J,K]+q33[J-1,K])/x2a[J] \
            +(q32[J,K+1]-q32[J,K])/dx3a[K])
    v2 = x2bi(v2,1,-1)
    v2 = x2iz(v2)
    v2 = x2bo(v2,1,-1)
    v2 = x2oz(v2)
    v2 = x3bio(v2)

    # update angular velocity as a result of viscous force/unit volume
    v3[J,K]=v3[J,K]+dt/(.5*(d[J,K]+d[J,K-1]))* \
            ((q23[J+1,K]*x2a[J+1]-q23[J,K]*x2a[J])/ \
            (dx2a[J]*x2b[J]**2) \
            +(q33[J,K]-q33[J,K-1])/(dx3b[K]*x2b[J]))
    v3 = x2bi(v3,0,1)
    v3 = x2bo(v3,0,1)
    v3 = x3bio(v3)


def source():
    global p,cs,v2,v3,q2,q3
    # source terms in the momentum equation.
    
    if (ie < 0):
        # Pressure from locally isothermal equation of state
        p = cs**2 * d
        p = x2bi(p,0,1)
        p = x2bo(p,0,1)
        p = x3bio(p)
    elif (ie == 0):
        # Pressure from a polytropic equation of state
        p = pK*d**gamma
        cs = np.sqrt(gamma*pK*d**(gamma-1))
        p = x2bi(p,0,1)
        p = x2bo(p,0,1)
        p = x3bio(p)
    else:
        # Pressure from energy equation
        p = e*(gamma-1)
        p = x2bi(p,0,1)
        p = x2bo(p,0,1)
        p = x3bio(p)
    
    # Replace equilibrium rotation curve:
    # This step necessary to produce equilibrium condition matched to the FFT potential. 
    if (istep == 1):
        ip = np.ogrid[1:npert]
        iactive[ip] = 0
        J,K = np.ogrid[1:n2+1,1:n3+1]
        dro = -htr*rm[0]*sz/(x2b[J]**3)
        dstar = rm[0]/(x2b[J]**2 + dmin**2)
        v3orig[J] = np.sqrt(x2b[J]*(dro+dstar))
        v3[J,K] = v3orig[J]
    
    v3 = x2bi(v3,0,1)
    v3 = x2bo(v3,0,1)
    v3 = x3bio(v3)

    if (vturb == 1):
        # evaluate the turbulent velocity potential
        j,k = np.ogrid[1:n2+1,1:n3+1]
        phiT[j,k] = 0.
        if (istep > 2):
            j,k = np.ogrid[2:n2,1:n3+1]
            for ip in xrange(1,npert+1):
                if (iactive[ip] == 1):
                    # then the mode is active
                    if (na[ip] == n2/2):
                        phiT[j,k] = phiT[j,k] + 2.*Amp[ip]*(1./np.sqrt(x2b[jcent[ip]]))\
                                *np.exp(-((j-jcent[ip])/(na[ip]/8.))**2)\
                                *np.sin(np.pi*(time-tstart[ip])\
                                /(tfinish[ip]-tstart[ip]))
                    else:
                        phiT[j,k] = phiT[j,k] + Amp[ip]*(1./np.sqrt(x2b[jcent[ip]]))\
                                *np.exp(-((j-jcent[ip])/(na[ip]/8.))**2)\
                                *np.cos((n3/na[ip])*x3b[k]\
                                -phaseT[ip]-omegaT[ip]*(time-tstart[ip]))\
                                *np.sin(np.pi*(time-tstart[ip])\
                                /(tfinish[ip]-tstart[ip]))
                    if (time > tfinish[ip]):
                        iactive[ip] = 0

                else:
                    iactive[ip] = 1
                    jrange = np.ogrid[1:n2+1]
                    ntot = 2**jrange
                    jwant = np.where(ntot > n2)[0][0]
                    nexpo = jrange[jwant]-1
                    power = int(np.log2(npert))
                    rchoice = np.random.random()*power
                    na[ip] = int(2**rchoice)
                    jcent[ip] = int(np.random.random()*(n2-1))+1
                    omegaT[ip] = v3[jcent[ip],1]/x2b[jcent[ip]] 
                    tstart[ip] = time
                    tfinish[ip] = x3b[(n3/na[ip])]*x2b[jcent[ip]]/cs[jcent[ip],1]
                    tfinish[ip] += time
                    phaseT[ip] = np.random.random()*2*np.pi
                    Amp[ip] = np.random.random()*amplitude
    p = x3bio(p)
    
    #----------------------------------------------
    # r direction geometry term + force terms
    # First determine (r,phi) location of central particle.
    j,k,ib = np.ogrid[1:n2+1,1:n3+1,0:2]
    xd = np.zeros((n2+1,n3+1,2))
    yd = np.zeros((n2+1,n3+1,2))
    xa = np.zeros((n2+1,n3+1,2))
    ya = np.zeros((n2+1,n3+1,2))

    xd[j,k,ib] = x2a[j]*cx3b[k]-xp[ib,0]
    yd[j,k,ib] = x2a[j]*sx3b[k]-xp[ib,1]
    tvari = xd**2 + yd**2 + dmin**2
    denom = tvari*np.sqrt(tvari)
    
    xa[j,k,ib] = rm[ib]*xd[j,k,ib]/denom[j,k,ib]
    ya[j,k,ib] = rm[ib]*yd[j,k,ib]/denom[j,k,ib]
    
    fgrv_r1 = xa[j,k,0]*cx3b[k]+ya[j,k,0]*sx3b[k]
    fgrv_r2 = xa[j,k,1]*cx3b[k]+ya[j,k,1]*sx3b[k]

    if (istep == 1):
        k = 1
        fextj = ((v3[j-1,k]+v3[j-1,k+1]\
                +v3[j,k+1]+v3[j,k])/4.)**2/x2a[j]\
                -((p[j,k] - p[j-1,k])/dx2b[j])\
                /(0.5*(d[j,k] + d[j-1,k]))\
                -fgrv_r1\
                -(phiT[j,k]-phiT[j-1,k])/dx2b[j]\
                -(phi[j,k]-phi[j-1,k])/dx2b[j]
        fext.flat[j] = fextj[:,0,:]
    
    j,k,ib = np.ogrid[1:n2+1,1:n3+1,1:4]
    v2[j,k] = v2[j,k] + dt*(((v3[j-1,k]+v3[j-1,k]\
                +v3[j,k+1]+v3[j,k])/4.)**2/x2a[j]\
                -((p[j,k] - p[j-1,k])/dx2b[j])\
                /(0.5*(d[j,k] + d[j-1,k]))\
                -fgrv_r1-fgrv_r2\
                -(phiT[j,k]-phiT[j-1,k])/dx2b[j]\
                -(phi[j,k]-phi[j-1,k])/dx2b[j]-fext[j])
    
    v2 = x2bo(v2,1,-1)
    v2 = x2oz(v2)
    v2 = x3bio(v2)

    #----------------------------------------------
    # phi direction coriolis term + force terms
    j,k,ib = np.ogrid[1:n2+1,1:n3+1,0:2]
    xd = np.zeros((n2+1,n3+1,2))
    yd = np.zeros((n2+1,n3+1,2))
    xa = np.zeros((n2+1,n3+1,2))
    ya = np.zeros((n2+1,n3+1,2))

    xd[j,k,ib] = x2b[j]*cx3a[k]-xp[ib,0]
    yd[j,k,ib] = x2b[j]*sx3a[k]-xp[ib,1]
    tvari = xd**2 + yd**2 + dmin**2
    denom = tvari*np.sqrt(tvari)
    
    xa[j,k,ib] = rm[ib]*xd[j,k,ib]/denom[j,k,ib]
    ya[j,k,ib] = rm[ib]*yd[j,k,ib]/denom[j,k,ib]

    fgrv_t1 = ya[j,k,0]*cx3a[k]-xa[j,k,0]*sx3a[k]
    fgrv_t2 = ya[j,k,1]*cx3a[k]-xa[j,k,1]*sx3a[k]

    v3[j,k] = v3[j,k] + dt*(-(1./(((d[j,k]+d[j,k-1])/2.)*x2b[j]))\
                *(p[j,k]-p[j,k-1])/dx3b[k]\
                -fgrv_t1-fgrv_t2\
                -(1./x2b[j])*(phiT[j,k]-phiT[j,k-1])/dx3b[k]\
                -(1./x2b[j])*(phi[j,k]-phi[j,k-1])/dx3b[k])
    v3 = x2bi(v3,0,1)
    v3 = x2bo(v3,0,1)
    v3 = x3bio(v3)
    
    #-------------------------------------------------------------------
    # determine the artificial viscosity
    # note that artificial viscosity corrections to the velocity
    # are not computed along the inner+outer radial grid boundaries
    if (cv > 0):
        
        j,k = np.ogrid[1:n2+1,1:n3+1]
        q2[j,k] = 0.
        condition = (((v2[j+1,k]-v2[j,k]) < 0)&(j > 1)&(j < n2))
        j,k = np.mgrid[1:n2+1,1:n3+1]
        jc,kc = j[condition],k[condition]
        j,k = jc,kc
        q2[j,k] = cv*d[j,k]*(v2[j+1,k]-v2[j,k])**2
        
        j,k = np.ogrid[1:n2+1,1:n3+1]
        q3[j,k] = 0.
        condition = (((v3[j,k+1]-v3[j,k]) < 0)&(j > 1)&(j < n2))
        j,k = np.mgrid[1:n2+1,1:n3+1]
        jc,kc = j[condition],k[condition]
        j,k = jc,kc
        q3[j,k] = cv*d[j,k]*(v3[j,k+1]-v3[j,k])**2

        q2 = x3bio(q2)
        q3 = x3bio(q3)

        # apply viscous pressure to update the velocities
        J,K = np.ogrid[2:n2,1:n3+1]
        v2[J,K] = v2[J,K] - dt*(q2[J,K]-q2[J-1,K])\
                    /(dx2b[J]*((d[J,K]+d[J-1,K])/2.))
        v3[J,K] = v3[J,K] - dt*(q3[J,K]-q3[J,K-1])\
                    /(x2b[J]*dx3b[J]*((d[J,K]+d[J,K-1])/2.))

        q2 = x3bio(q2)
        q3 = x3bio(q3)

    if (ie > 0):
        J,K = np.ogrid[1:n2+1,1:n3+1]
        delve = (x2a[K+1]*v2[K+1,K]-x2a[K]*v2[K,K])\
                /(x2b[K]*dx2a[i])\
                +(v3[K,K+1] - v3[K,K])/(dx3a[K]*x2b[K])
        delve = (dt/2.)*(gamma-1)*delve
        e[J,K] = ((1.-delve)/(1.+delve))*e[J,K]
    
        e = x2bi(e,0,1)
        e = x2bo(e,0,1)
        e = x3bio(e)

def rho_r_advect():
    global F,Fm,d,ds,dvg,vs,v2,v3,q2,q3
    # density advection (2 -- (r) direction)
    
    #----------------------------------------------------
    # set up interpolated densities at r zone interfaces
    # if (v3[j,k] > 0): 
    j,k = np.ogrid[1:n2+1,1:n3+1]
    condition = (v2[j,k] > 0) 
    j,k = np.mgrid[1:n2+1,1:n3+1]
    jc,kc = j[condition],k[condition]
    j,k = jc,kc
    dqpi12 = (d[j,k]-d[j-1,k])/dx2b[j]
    dqmi12 = (d[j-1,k]-d[j-2,k])/dx2b[j-1]
    condition = ((dqmi12*dqpi12) > 0)
    dqi = (dqmi12*dqpi12)/(dqmi12+dqpi12)
    dqi[~condition] = 0.
    # donor cell
    ds[j,k] = d[j-1,k] + (dx2a[j-1]-v2[j,k]*dt)*dqi
    # else:
    j,k = np.ogrid[1:n2+1,1:n3+1]
    condition = ~(v2[j,k] > 0) 
    j,k = np.mgrid[1:n2+1,1:n3+1]
    jc,kc = j[condition],k[condition]
    j,k = jc,kc
    dqpi12 = (d[j+1,k]-d[j,k])/dx2b[j+1]
    dqmi12 = (d[j,k]-d[j-1,k])/dx2b[j]
    condition = ((dqmi12*dqpi12) > 0)
    dqi = (dqmi12*dqpi12)/(dqmi12+dqpi12)
    dqi[~condition] = 0.
    # donor cell
    ds[j,k] = d[j,k] + (dx2a[j]-v2[j,k]*dt)*dqi
    #----------------------------------------------------
    # boundary condition
    ds[-1:2,1:n3+1] = d[1,1:n3+1]
    # Face centered fluxes in the R direction
    ds[1:n2+2,1:n3+1] *= v2[1:n2+2,1:n3+1]
    # inner accretion segment
    ds = x2bi(ds,1,1)
    ds = x2iz(ds)
    ds = x2bo(ds,1,-1)
    ds = x2oz(ds)
    ds = x3bio(ds)

    J,K = np.ogrid[0:n2+2,1:n3+1]
    Fm[J,K] = ds[J,K]*dz*x2a[J]*dx3a[K]
    fluxr[1:n3+1] = Fm[1,1:n3+1]
    Fm = x3bio(Fm)
    # Update the density
    J,K = np.ogrid[1:n2+1,1:n3+1]
    d[J,K] = d[J,K] - (dt/dv[J,K])*(Fm[J+1,K]-Fm[J,K])
    
    d = x2bi(d,0,1)
    d = x2bo(d,0,1)
    d = x3bio(d)

def rho_phi_advect():
    global F,Fm,d,ds,dvg,vs,v2,v3,q2,q3
    # density advection in phi direction
    # 3 direction -- cylindrical mass flux
    #----------------------------------------------------
    # set up interpolated densities on the phi zone faces
    # if (v3[j,k] > 0): 
    j,k = np.ogrid[1:n2+1,1:n3+2]
    condition = (v3[j,k] > 0) 
    j,k = np.mgrid[1:n2+1,1:n3+2]
    jc,kc = j[condition],k[condition]
    j,k = jc,kc
    dqpi12 = (d[j,k] - d[j,k-1])/(dx3b[k]*x2b[j])
    dqmi12 = (d[j,k-1] - d[j,k-2])/(dx3b[k-1]*x2b[j])
    condition = ((dqmi12*dqpi12) > 0)
    dqi = (dqmi12*dqpi12)/(dqmi12+dqpi12)
    dqi[~condition] = 0.
    # donor cell
    ds[j,k] = d[j,k-1]+(dx3a[k-1]*x2b[j]-v3[j,k]*dt)*dqi
    # else:
    j,k = np.ogrid[1:n2+1,1:n3+2]
    condition = ~(v3[j,k] > 0) 
    j,k = np.mgrid[1:n2+1,1:n3+2]
    jc,kc = j[condition],k[condition]
    j,k = jc,kc
    dqpi12 = (d[j,k+1] - d[j,k])/(dx3b[k+1]*x2b[j])
    dqmi12 = (d[j,k] - d[j,k-1])/(dx3b[k]*x2b[j])
    condition = ((dqmi12*dqpi12) > 0)
    dqi = (dqmi12*dqpi12)/(dqmi12+dqpi12)
    dqi[~condition] = 0.
    # donor cell
    ds[j,k] = d[j,k]-(dx3a[k]*x2b[j]+v3[j,k]*dt)*dqi
    #----------------------------------------------------
    # Compute mass flux densities (by re-labeling ds(i,j,k) etc.)
    # Compute face centered fluxes in the phi direction
    J,K = np.ogrid[1:n2+1,1:n3+2]
    ds[J,K] *= v3[J,K]
    Fm[J,K] = ds[J,K]*dz*dx2a[J]

    ds = x2bi(ds,0,1)
    Fm = x2bi(Fm,0,1)
    ds = x2bo(ds,0,1)
    Fm = x2bo(Fm,0,1)
    ds = x3bio(ds)
    Fm = x3bio(Fm)

    # Update the density
    J,K = np.ogrid[1:n2+1,1:n3+1]
    d[J,K] = d[J,K] - (dt/dv[J,K])*(Fm[J,K+1]-Fm[J,K])

    d = x2bi(d,0,1)
    d = x2bo(d,0,1)
    d = x3bio(d)

def rmoment_r_advect():
    global F,Fm,d,ds,dvg,vs,v2,v3,q2,q3
    # r momentum advection in the r direction
    # Extrapolate r velocity to r control volume surface
    #----------------------------------------------------
    # if ((Fm[j,k] + Fm[j+1,k]) > 0):
    j,k = np.ogrid[0:n2+1,1:n3+1]
    condition = ((Fm[j,k] + Fm[j+1,k]) > 0)
    j,k = np.mgrid[0:n2+1,1:n3+1]
    jc,kc = j[condition],k[condition]
    j,k = jc,kc
    dqmi12 = (v2[j,k] - v2[j-1,k])/dx2a[j-1]
    dqpi12 = (v2[j+1,k] - v2[j,k])/dx2a[j]
    condition = ((dqmi12*dqpi12) > 0)
    dqi = (dqmi12*dqpi12)/(dqmi12+dqpi12)
    dqi[~condition] = 0.
    # donor cell
    vs[j,k] = v2[j,k]+(dx2b[j]-((v2[j,k]+v2[j+1,k])/2.)*dt)*dqi
    # else:
    j,k = np.ogrid[0:n2+1,1:n3+1]
    condition = ~((Fm[j,k] + Fm[j+1,k]) > 0)
    j,k = np.mgrid[0:n2+1,1:n3+1]
    jc,kc = j[condition],k[condition]
    j,k = jc,kc
    dqmi12 = (v2[j+1,k] - v2[j,k])/dx2a[j]
    dqpi12 = (v2[j+2,k] - v2[j+1,k])/dx2a[j+1]
    condition = ((dqmi12*dqpi12) > 0)
    dqi = (dqmi12*dqpi12)/(dqmi12+dqpi12)
    dqi[~condition] = 0.
    # donor cell
    vs[j,k] = v2[j+1,k]-(dx2b[j+1]+((v2[j,k]+v2[j+1,k])/2.)*dt)*dqi
    #----------------------------------------------------
    vs = x2bi(vs,1,-1)
    vs = x2iz(vs)
    vs = x2bo(vs,1,-1)
    vs = x2oz(vs)
    vs = x3bio(vs)
    
    # construct flux array for r momentum advection in r direction
    J,K = np.ogrid[0:n2+1,1:n3+1]
    Area = dx3a[K]*x2b[J]*dz
    F[J,K] = vs[J,K]*0.5*(ds[J,K]+ds[J+1,K])*Area
    dvg[J,K] = 0.5*(x2b[J]**2 - x2b[J-1]**2)*dx3a[K]*dz
    # Update specific 2 -- r momentum due to 2 -- r momentum advection
    # in the 2 -- (r) direction
    J,K = np.ogrid[1:n2+1,1:n3+1]
    q2[J,K] = q2[J,K] - (dt/dvg[J,K])*(F[J,K]-F[J-1,K])

def phimoment_r_advect(): 
    global F,Fm,d,ds,dvg,vs,v2,v3,q2,q3
    # phi momentum advection in the r direction
    # phi momentum in the r direction
    # extrapolate phi velocity to control volume interface
    #----------------------------------------------------
    # if ((Fm[j+1,k]+Fm[j+1,k-1]) > 0.):
    j,k = np.ogrid[0:n2+1,1:n3+1]
    condition = ((Fm[j+1,k]+Fm[j+1,k-1]) > 0.)
    j,k = np.mgrid[0:n2+1,1:n3+1]
    jc,kc = j[condition],k[condition]
    j,k = jc,kc
    dqmi12 = (v3[j,k] - v3[j-1,k])/dx2b[j]
    dqpi12 = (v3[j+1,k] - v3[j,k])/dx2b[j+1]
    condition = ((dqmi12*dqpi12) > 0)
    dqi = (dqmi12*dqpi12)/(dqmi12+dqpi12)
    dqi[~condition] = 0.
    # donor cell
    vs[j+1,k] = v3[j,k]+(dx2a[j]-((v2[j+1,k]+v2[j+1,k-1])/2.)*dt)*dqi
    # else:
    j,k = np.ogrid[0:n2+1,1:n3+1]
    condition = ~((Fm[j+1,k]+Fm[j+1,k-1]) > 0.)
    j,k = np.mgrid[0:n2+1,1:n3+1]
    jc,kc = j[condition],k[condition]
    j,k = jc,kc
    dqmi12 = (v3[j+1,k] - v3[j,k])/dx2b[j+1]
    dqpi12 = (v3[j+2,k] - v3[j+1,k])/dx2b[j+2]
    condition = ((dqmi12*dqpi12) > 0)
    dqi = (dqmi12*dqpi12)/(dqmi12+dqpi12)
    dqi[~condition] = 0.
    # donor cell
    vs[j+1,k] = v3[j+1,k]-(dx2a[j+1]+((v2[j+1,k]+v2[j+1,k-1])/2.)*dt)*dqi
    #----------------------------------------------------
    vs = x2bi(vs,1,1)
    vs = x2iz(vs)
    vs = x2bo(vs,1,1)
    vs = x2oz(vs)
    vs = x3bio(vs)

    # construct flux array for phi (angular) momentum advection in the r dir.
    J,K = np.ogrid[1:n2+2,1:n3+1]
    Area = dx3b[K]*x2a[J]*dz
    F[J,K] = vs[J,K]*0.5*(ds[J,K]+ds[J,K-1])*Area*x2a[J]
    dvg[J,K] = dz*0.5*(x2a[J+1]**2 - x2a[J]**2)*dx3b[K]
    J,K = np.ogrid[1:n2+1,1:n3+1]
    q3[J,K] = q3[J,K] - (dt/dvg[J,K])*(F[J+1,K]-F[J,K])

def rmoment_phi_advect(): 
    global F,Fm,d,ds,dvg,vs,v2,v3,q2,q3
    # r momentum advection in the phi direction
    #----------------------------------------------------
    # if ((Fm[j,k+1]+Fm[j-1,k+1]) > 0.):
    j,k = np.ogrid[1:n2+1,0:n3+1]
    condition = ((Fm[j,k+1]+Fm[j-1,k+1]) > 0.)
    j,k = np.mgrid[1:n2+1,0:n3+1]
    jc,kc = j[condition],k[condition]
    j,k = jc,kc
    dqmi12 = (v2[j,k] - v2[j,k-1])/(dx3b[k]*x2a[j])
    dqpi12 = (v2[j,k+1] - v2[j,k])/(dx3b[k+1]*x2a[j])
    condition = ((dqmi12*dqpi12) > 0)
    dqi = (dqmi12*dqpi12)/(dqmi12+dqpi12)
    dqi[~condition] = 0.
    # donor cell
    vs[j,k+1] = v2[j,k]+(x2a[j]*dx3a[k]-((v3[j-1,k+1]+v3[j,k+1])/2.)*dt)*dqi
    # else:
    j,k = np.ogrid[1:n2+1,0:n3+1]
    condition = ~((Fm[j,k+1]+Fm[j-1,k+1]) > 0.)
    j,k = np.mgrid[1:n2+1,0:n3+1]
    jc,kc = j[condition],k[condition]
    j,k = jc,kc
    dqmi12 = (v2[j,k+1] - v2[j,k])/(dx3b[k+1]*x2a[j])
    dqpi12 = (v2[j,k+2] - v2[j,k+1])/(dx3b[k+2]*x2a[j])
    condition = ((dqmi12*dqpi12) > 0)
    dqi = (dqmi12*dqpi12)/(dqmi12+dqpi12)
    dqi[~condition] = 0.
    # donor cell
    vs[j,k+1] = v2[j,k+1]-(dx3a[k+1]*x2a[j]+((v3[j-1,k+1]+v3[j,k+1])/2.)*dt)*dqi
    #----------------------------------------------------
    vs = x2bi(vs,0,1)
    vs = x2bo(vs,0,1)
    vs = x3bio(vs)

    # construct flux array for r momentum advection in the phi direction
    J,K = np.ogrid[1:n2+1,1:n3+2]
    Area = dx2b[J]*dz
    dvg[J,K] = dz*0.5*dx3a[K]*(x2b[J]**2 - x2b[J-1]**2)
    F[J,K] = vs[J,K]*0.5*(ds[J,K]+ds[J-1,K])*Area
    J,K = np.ogrid[1:n2+1,1:n3+1]
    q2[J,K] = q2[J,K] - (dt/dvg[J,K])*(F[J,K+1]-F[J,K])

def phimoment_phi_advect():
    global F,Fm,d,ds,dvg,vs,v2,v3,q2,q3
    # phi momentum advection in the phi direction
    # phi momentum in the phi direction
    #----------------------------------------------------
    # if ((Fm[j,k]+Fm[j,k+1]) > 0.):
    j,k = np.ogrid[1:n2+1,0:n3+1]
    condition = ((Fm[j,k]+Fm[j,k+1]) > 0.)
    j,k = np.mgrid[1:n2+1,0:n3+1]
    jc,kc = j[condition],k[condition]
    j,k = jc,kc
    dqmi12 = (v3[j,k] - v3[j,k-1])/(dx3a[k-1]*x2b[j])
    dqpi12 = (v3[j,k+1] - v3[j,k])/(dx3a[k]*x2b[j])
    condition = ((dqmi12*dqpi12) > 0)
    dqi = (dqmi12*dqpi12)/(dqmi12+dqpi12)
    dqi[~condition] = 0.
    # donor cell
    vs[j,k] = v3[j,k]+(x2b[j]*dx3b[k]-((v3[j,k]+v3[j,k+1])/2.)*dt)*dqi
    # else:
    j,k = np.ogrid[1:n2+1,0:n3+1]
    condition = ~((Fm[j,k]+Fm[j,k+1]) > 0.)
    j,k = np.mgrid[1:n2+1,0:n3+1]
    jc,kc = j[condition],k[condition]
    j,k = jc,kc
    dqmi12 = (v3[j,k+1] - v3[j,k])/(dx3a[k]*x2b[j])
    dqpi12 = (v3[j,k+2] - v3[j,k+1])/(dx3a[k+1]*x2b[j])
    condition = ((dqmi12*dqpi12) > 0)
    dqi = (dqmi12*dqpi12)/(dqmi12+dqpi12)
    dqi[~condition] = 0.
    # donor cell
    vs[j,k] = v3[j,k+1]-(dx3b[k+1]*x2b[j]+((v3[j,k]+v3[j,k+1])/2.)*dt)*dqi
    #----------------------------------------------------
    vs = x2bi(vs,0,1)
    vs = x2bo(vs,0,1)
    vs = x3bio(vs)
    # construct flux array for phi angular momentum in the phi direction
    J,K = np.ogrid[1:n2+1,0:n3+1]
    Area = dx2a[J]*dz
    dvg[J,K] = dz*0.5*dx3b[K]*(x2a[J+1]**2 - x2a[J]**2)
    F[J,K] = vs[J,K]*0.5*x2b[J]*Area*(ds[J,K]+ds[J,K+1])
    J,K = np.ogrid[1:n2+1,1:n3+1]
    q3[J,K] = q3[J,K] - (dt/dvg[J,K])*(F[J,K]-F[J,K-1])

def transport():
    global F,Fm,d,ds,dvg,vs,v2,v3,q2,q3
    # Transform velocities to momenta, and in the case of the phi
    # velocity, transform to angular momentum.
    F = np.zeros((n2+4,n3+4))
    dvg = np.zeros((n2+4,n3+4))
    J,K = np.ogrid[1:n2+1,1:n3+1]
    q2[J,K] = ((d[J-1,K]+d[J,K]))/2. * v2[J,K]
    q3[J,K] = ((d[J,K-1]+d[J,K]))/2. * v3[J,K]*x2b[J]
    # advect everything
    rho_r_advect()
    rmoment_r_advect()
    phimoment_r_advect()
    rho_phi_advect()
    rmoment_phi_advect()
    phimoment_phi_advect()
    # transform momenta back to velocities
    v2[J,K] = q2[J,K]/((d[J-1,K]+d[J,K])/2.)
    v3[J,K] = q3[J,K]/((d[J,K-1]+d[J,K])/2.)/x2b[J]
    # boundary condition associated with accretion
    v2 = x2bi(v2,1,-1)
    v2 = x2iz(v2)
    v2 = x2bo(v2,1,-1)
    v2 = x2oz(v2)
    v2 = x3bio(v2)
    v3 = x2bi(v3,0,1)
    v3 = x2bo(v3,0,1)
    v3 = x3bio(v3)

def x2bidamp():
    global d,v2,v3,v3orig
    J,K = np.ogrid[1:ndj+1,1:n3+1]
    factor = np.sqrt((J-1)/float(ndj))
    v2[J,K] *= factor
    v3[J,K] = (1.-factor)*v3orig[J] + factor*v3[J,K]
    d[J,K] = factor*d[J,K] + (1.-factor)*(sz/x2b[J])
    v2 = x2bi(v2,1,-1)
    v2 = x3bio(v2)
    v3 = x2bi(v3,0,1)
    v3 = x3bio(v3)
    d = x2bi(d,0,1)
    d = x3bio(d)

def x2bodamp():
    global d,v2,v3,v3orig
    J,K = np.ogrid[ondj:n2+1,1:n3+1]
    factor = np.sqrt((n2-J)/float(n2-ondj))
    v2[J,K] *= factor
    v3[J,K] = (1.-factor)*v3orig[J] + factor*v3[J,K]
    d[J,K] = factor*d[J,K] + (1.-factor)*(sz/x2b[J])
    v2 = x2bo(v2,1,-1)
    v2 = x3bio(v2)
    v3 = x2bo(v3,0,1)
    v3 = x3bio(v3)
    d = x2bo(d,0,1)
    d = x3bio(d)

def advance():
    """
    This routine updates the positions of the central star and planets
    from their values at the middle of the Runge-Kutta interval to
    their values at the end of the Runge-Kutta interval. The values
    at the midpoint of the interval have been used throughout the
    current timestep to retain time centering.
    """
    global xpf,xp,vp
    global qa,rna,ria,ea,pa,rla
    xp[1,0] = xpf[1,0]
    xp[1,1] = xpf[1,1]
    #va1 = np.sum(rm)
    va2 = xp[1,0]
    va3 = xp[1,1]
    va4,va7 = 0.,0.
    va5 = vp[1,0]
    va6 = vp[1,1]
    qa,ea,ria,pa,rna,rla = mco_x2el(va1,va2,va3,va4,va5,va6,va7)

def derivs(x,y,dydx):
    # derivatives for the Runge Kutta Planet-Star Integrator
    xd12 = y[1] - y[0]
    yd12 = y[5] - y[4]
    denom12 = (xd12**2 + yd12**2)**(3/2.)

    # x velocity for particle #1
    dydx[0] = 0. #y[2]
    # x velocity for particle #2
    dydx[1] = y[3]
    # x accleration for particle #1
    dydx[2] = 0.
    # x acceleration for particle #2
    dydx[3] = -rm[0]*xd12/denom12
    # y velocity for particle #1
    dydx[4] = 0.
    # y velocity for particle #1
    dydx[5] = y[7]
    # y accleration for particle #1
    dydx[6] = 0.
    # y acceleration for particle #2
    dydx[7] = -rm[0]*yd12/denom12
    # acceleration on particle #1 due to disk
    dydx[2] = 0.
    dydx[6] = 0.
    # acceleration on particle #2 due to disk
    cont3 = 0.
    cont7 = 0.
    dist = np.sqrt(y[1]**2 + y[5]**2)
    rocherad = dist*(rm[1]/rm[0])**(1/3.)
    J,K = np.ogrid[1:n2+1,1:n3+1]
    condition = (np.abs(x2b[J] - dist) > rocherad)
    J = np.vstack(J[condition])
    distx = x2b[J]*cx3b[K]-y[1]
    disty = x2b[J]*sx3b[K]-y[5]
    factor1 = distx**2 + disty**2
    factor = factor1 + dmin**2
    factor = factor**(3/2.)
    denom = d[J,K] * dv[J,K]/factor
    cont3 += np.sum(denom*distx)
    cont7 += np.sum(denom*disty)
    if planetfeeldisk==True:
        dydx[3] += cont3
        dydx[7] += cont7

    return dydx

def rk4(Y,DYDX,N,X,H,YOUT):
    # from numerical recipies
    nmax = len(Y)
    YT = np.zeros(nmax)
    DYT = np.zeros(nmax)
    DYM = np.zeros(nmax)
    HH = H*0.5
    H6 = H/6.
    XH = X+HH
    
    i = np.arange(N)
    YT[i] = Y[i] + HH*DYDX[i]
    DYT = derivs(XH,YT,DYT)
    YT[i] = Y[i] + HH*DYT[i]
    DYM = derivs(XH,YT,DYM)
    YT[i] = Y[i] + H*DYM[i]
    DYM[i] = DYT[i]+DYM[i]
    DYT = derivs(X+H,YT,DYT)
    YOUT[i] = Y[i]+H6*(DYDX[i] + DYT[i] + 2*DYM[i])
    
    return YOUT

def rkdumb(ystart,nvar,X1,X2,NSTEP):
    global y,xx,yval
    # from Numerical Recipies
    nmax = len(yval)
    V = np.zeros(nmax)
    DV = np.zeros(nmax)
    i = np.arange(nvar)
    V[i] = ystart[i]
    yval[i,1] = V[i]
    xx[1] = X1
    X = X1
    H = (X2-X1)/NSTEP
    for k in range(NSTEP):
        DV = derivs(X,V,DV)
        V = rk4(V,DV,nvar,X,H,V)
        if (X+H == X): print 'Stepsize not significant in RKDUMB.'
        X = X+H
        xx[k] = X
        i = np.arange(nvar)
        yval[i,k] = V[i]
    return yval 

def rkbody(xp,xpf,vp,dt):
    global y,xx,yval
    
    # rkbody variables
    y[0] = xp[0,0]
    y[1] = xp[1,0]
    y[2] = vp[0,0]
    y[3] = vp[1,0]
    y[4] = xp[0,1]
    y[5] = xp[1,1]
    y[6] = vp[0,1]
    y[7] = vp[1,1]
    
    nvar = len(y)
    nstep = 2
    nstep1 = 1
    nstep2 = 0
    x1 = 0.
    x2 = dt
    yval = rkdumb(y,nvar,x1,x2,nstep)
  
    x1dum = np.array([0,0,1,1])
    x2dum = np.array([0,1,0,1])
    ydum = np.array([0,4,1,5])
    xp[x1dum,x2dum] = yval[ydum,nstep2]
    xpf[x1dum,x2dum] = yval[ydum,nstep1]
    ydum = np.array([2,6,3,7])
    vp[x1dum,x2dum] = yval[ydum,nstep1]

    return xp,xpf,vp

def interrupt(istep):
    global iout
    if (time >= iout*outint):
        iout += 1
        print 'writing at istep = '+str(istep)+', time = '+str(time)+', dt = '+str(dt)
        #-----------------------------------------------------------
        # write bodies information
        with open(bodfname,'a') as bodf:
            row = '%s %s %s %s %s %s %s %s\n' %(xp[0,0],xp[0,1],xp[1,0],xp[1,1],
                                                vp[0,0],vp[0,1],vp[1,0],vp[1,1])
            bodf.write(row)
        # write timestep info
        with open(timefname,'a') as timef:
            row = '%s %s \n' %(istep,time/bperiod)
            timef.write(row)
        # write density mesh
        #den = d[1:n2+2,1:n3+2]
        den = d[1:n2+1,1:n3+1]
        # option 1: as pandas dataframe
        keyname = 'den_%06d' %istep
        df = pd.DataFrame(den)
        df.to_hdf(h5file,keyname)
        # option 2: as text/numpy array
        # TODO

        # also TODO: write velocity meshs
        # write toomreQ profile
        global soundspeed,Omega,sigma,toomreQ
        #update toomre Q
        soundspeed = np.mean(np.sqrt(gamma*pK*d**(gamma-1)),axis=1)[1:-3]
        Omega = np.mean(v3,axis=1)[1:-3]
        sigma = np.mean(d,axis=1)[1:-3]
        toomreQ = soundspeed*Omega/(np.pi*sigma)
        with open(toomreQfname,'a') as toomref:
            #np.savetxt(toomref,toomreQ,fmt='%.5e')
            np.savetxt(toomref,toomreQ.reshape(1, toomreQ.shape[0]),fmt='%.5e')
        #-----------------------------------------------------------
    #print 'printed to %s' %fname
    if (time > trun):
        print 'system time ran over expected time'
        sys.exit()

def set_gravity2d():
    global d,Glm,x2b
    """
    Prepare fixed terms for calls to FFT 2--D potential solver
    This routine only called once during run.
    Note, for historical reasons, 2D gravity routines implement
    two dimensional variables with (i=theta,j=radius) ordering. 
    """
    # logarithmic interval
    du = (np.log(xdmax) - np.log(xdmin))/nr
    # equally spaced theta intervals
    dphi = dx3a[1]
    # compute fixed constants for the Poisson solver
    i,j = np.ogrid[-n3:n3,-n2:n2]
    denom = np.sqrt((np.exp(j*du)+np.exp(-j*du))/2.-np.cos(i*dphi))
    good = np.where(denom!=0.)
    Glm[good] = -np.sqrt(2.)/2. / denom[good]
    bad = np.where(denom==0.)
    junk = -2*((1./dphi)*np.log((dphi/du)+np.sqrt((dphi/du)**2+1)))\
            + -2*((1./du)*np.log((du/dphi)+np.sqrt((du/dphi)**2+1)))
    Glm[bad] = junk

def getphilm(l,m):
    global Glm
    global Mlm
    global philm
    sumlm = 0.
    lp,mp = np.ogrid[0:philm.shape[0],0:philm.shape[1]]
    junk = Glm[l-lp,m-mp]*Mlm[lp,mp]
    sumlm += np.sum(junk)
    return sumlm

def gravity2d():
    global Glm
    global Mlm
    global d
    global phi
    global x2b
    global phiFFT
    """
    Use 2D Fourier Convolution theorem to solve for gravitational
    potential phi(1,j,i). See Binney & Tremaine pp. 96-97, details
    Note 2d indexing on subroutine internal variables is (theta,r)
    """
    k = nr
    n = nr*2
    # logarithmic interval in r
    du = dlnr
    # equally spaced phi intervals
    dphi = dx3a[1]
    
    ioff,joff = 1,1
    data1 = np.zeros((2*n3+ioff,2*n2+ioff))
    data2 = np.zeros((2*n3+ioff,2*n2+ioff))
   
    i,j = np.mgrid[0:n3,0:n2]
    junk = d[j+1,i+1]*dz*(x2b[j+1]**(3/2.))*du*dphi
    Mlm[i,j] = junk

    i,j = np.ogrid[1:2*n3+1,1:2*n2+1]
    data1[i,j] = Glm[i-1-2*n3,j-1-2*n2]
    data2[i,j] = Mlm[i-1-2*n3,j-1-2*n2]
    ii,jj = np.mgrid[1:2*n3+1,1:2*n2+1]
    cond1 = ((ii<=n3)&(jj<=n2))
    i,j = ii[cond1],jj[cond1]
    data1[i,j] = Glm[i-1,j-1]
    data2[i,j] = Mlm[i-1,j-1]
    cond2 = ((ii<=n3)&(jj>n2))
    i,j = ii[cond2],jj[cond2]
    data1[i,j] = Glm[i-1,j-1-2*n2]
    data2[i,j] = Mlm[i-1,j-1-2*n2]
    cond3 = ((ii>n3)&(jj<=n2))
    i,j = ii[cond3],jj[cond3]
    data1[i,j] = Glm[i-1-2*n3,j-1]
    data2[i,j] = Mlm[i-1-2*n3,j-1]
    i,j = np.mgrid[1:2*n3+1,1:2*n2+1]
    
    # call the FFT routines and perform convolution
    data1 = np.fft.rfft2(data1,[data1.shape[0]-1,data1.shape[1]])
    data2 = np.fft.rfft2(data2,[data2.shape[0]-1,data2.shape[1]])
    fac = 1./(2*n3*n2)
    dum1 = fac*data1*data2
    i,j = np.ogrid[1:n3+1,1:n2+1]
    phi[j,i] = np.fft.irfft2(data1*data2)[i,j]
    phiFFT = np.copy(phi)

    phi = x2bi(phi,0,1)
    phi = x2bo(phi,0,1)
    phi = x3bio(phi)


if __name__ == "__main__":
    ###############################
    # main evolution cycle
    ###############################
    # set up initial potential variables
    if selfgrav: set_gravity2d()
    # Here retain full call to the boundary conditions
    boundary()
    # call timestep
    interrupt(istep)
    time,dt = timestep()
    global toomreQ
    for istep in xrange(1,nsteps):
        # call central
        central()
        # call gravity2d
        if selfgrav:
            gravity2d()
        # call source
        source()
        # call viscosity
        #viscosity()
        # call transport
        transport()
        # cadencing
        itmp = istep%int((dx2a[1]/cs[1,1])/dt)
        if (itmp == 0):
            # call x2bidamp
            x2bidamp()
        itmpout = istep%int((dx2a[n2]/cs[n2,1])/dt)
        if (itmpout == 0):
            # call x2bodamp
            x2bodamp()
        # call advance
        advance()
        # call interrupt
        interrupt(istep)
        # call timestep
        time,dt = timestep()
    # DONE
