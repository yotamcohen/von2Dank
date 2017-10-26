###############################
# output and timing options
###############################
run_name = 'quickstart' # will appear at the end of the output filenames
verbose = 0 # True to print the timestep 
nprint = 1 # print out the time to the terminal after every nprint timesteps if verbose = 1
outint = 0.5 # time interval for output (in units of the system time)
trun = 20. # expected elapsed system time at finish
nsteps = int(1e9) # maximum number of timesteps allowed

###############################
# resolution
###############################
nr = 128 # number of radial grid points
nphi = nr*2 # number of azimuthal grid points

###############################
# star and planet
###############################
mstar = 1.0 # mass of the central star in solar masses
mplanet = 1.e-3 # mass of planet, in units of the star mass
ecc = 0. # eccentricity of orbit
dperi = 1.5 # periastron distance in AU
lperi = 0. # longitude of periastron in radians
meananom = 0. # initial mean anomaly in radians
dmin = 0.05 # softening length for planets
planetfeeldisk = 1 # whether the planet will feel accelerations from the disk

###############################
# fluid disk
###############################
selfgrav = 1 # 0 for off, 1 for on
Qmin = 1.5 # normalization (minimum value) of toomre Q
rmdisk = 0.05 # mass fraction of the disk (as a fraction of the stars mass)
xdmin = 0.4  # disk inner radius in AU
xdmax = 3.0 # disk outer radius im AU
rdj = 0.5 # inner damping radius in AU
ordj = 2.5 # outer damping radius in AU
cv = 4. # coefficient for artificial viscosity (negative if inactive)
gamma = 1.40 # polytropic index
