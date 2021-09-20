These codes are for rununing MCMC analyses of TESS data. Most of these
codes constitute a middle layer bewteen other light-curve model codes
and (commonly) an specific general MCMC code located at
https://github.com/JohnGBaker/ptmcmc.git

All these codes are developmental and not recommended for other outside
direct collaboration.

SLB_MCMC.py: interfaces to 'LightCurveCode' for self-lensing binaries
located at https://gitlab.fit.nasa.gov/jgbaker/tessslb.git

HB_MCMC.py: A version of the above which attempts too apply the SLB
code above to analyze "Heartbeat" ecccentric binary stars.

HB_MCMC2.py: A version of the above linked to a an adaptation of the light
curve code to properly support heartbeat systems.  Currently that code
is available at https://github.com/JohnGBaker/HB_MCMC.git, a fork of
https://github.com/sidruns30/HB_MCMC.git

