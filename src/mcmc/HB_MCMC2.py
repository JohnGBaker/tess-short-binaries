#Written by John G Baker @ NASA 2020-2021
#The original version of this code is at
#https://gitlab.fit.nasa.gov/jgbaker/tessslb.git
#
# To use this code you will need the HB lightcurve code
# at github/sidruns30/HB_MCMC and the MCMC code at
# github/johngbaker/ptmcmc
#
from astropy.timeseries import LombScargle
import pandas as pd
import numpy as np
import json

#import matplotlib as mpl
#import matplotlib.pyplot as plt
import astroquery
from astroquery.mast import Catalogs,Observations
#import re
import sys
import os

#dirp='../../../TessSLB/src/LightCurveCode'
#if dirp not in sys.path: sys.path.append(dirp)
#dirp='../../../TessSLB/src/LightCurveCode/ptmcmc/cython'
#if dirp not in sys.path: sys.path.append(dirp)
dirp='../ptmcmc/python'
if dirp not in sys.path: sys.path.append(dirp)
dirp='ptmcmc/python'
if dirp not in sys.path: sys.path.append(dirp)
import ptmcmc
import ptmcmc_analysis
import pyHB
#import BrowseSLBs
import copy
#import warnings
import scipy
import glob
import pickle
import re


useM=False

def fold_lc(times,fluxes,errs,Pfold,downfac=1.0,decimate_level=None,rep=False):
    #if decimate level is set, that overrides the native downsampling/binning
    #method using the decimate.py approach with the specified level.
    phases=(np.array(times)/Pfold)%1
    isort=np.argsort(phases)
    phases=phases[isort]
    fluxes=np.array(fluxes)[isort]
    errs=np.array(errs)[isort]
    nold=len(times)

    if decimate_level is not None and decimate_level>=0:
        import decimate
        if rep: print('nold,Pfold,decimate_lev:',nold,Pfold,decimate_level,times[0],"< t <",times[-1])
        data=np.array([[phases[i],fluxes[i],errs[i]] for i in range(len(phases))])
        newdata=decimate.decimate(data,lev=decimate_level,npretemper=0,verbose=True)
        fphases=newdata[:,0]
        ffluxes=newdata[:,1]
        ferrs=newdata[:,2]
    else:      
        if rep: print('nold,Pfold,downfac:',nold,Pfold,downfac,times[0],"< t <",times[-1])
    
        groupwidth=(times[-1]-times[0])*(1+0.1/nold)/nold/Pfold #frac period bin size
        groupwidth*=downfac
        #print('mean errs=',errs.mean())
        if rep: print('groupwidth=',groupwidth, 'mean group size=',groupwidth*nold)
        fphases=[]
        ffluxes=[]
        ferrs=[]
        i=0
        j=0
        while(i<nold):
            #print(i,j)
            xs=[]
            ys=[]
            es=[]
            tr=phases[0]+groupwidth*j
            while(i<nold and phases[i]<tr):
                #print(i,times[i],tr)
                xs.append(phases[i])
                ys.append(fluxes[i])
                es.append(errs[i])
                i+=1
            #print(tr,xs,ys,es)
            if(len(xs)>0):
                xs=np.array(xs)
                ys=np.array(ys)
                es=np.array(es)
                ws=1/es**2
                w=np.sum(ws)
                x=np.sum(xs*ws)/w
                y=np.sum(ys*ws)/w
                v=np.sum((ys-y)**2*ws)/w
                #print(ys)
                #print(es)
                #print(np.sqrt(1/w),np.sqrt(v/len(xs)),np.sqrt(np.sum((ys-y)**2)/len(xs)**2))
                e=np.sqrt(1/w+v/len(xs))#Not 100% sure this is right
                #if rep:print(xs,ys,es,'-->',x,y,e,1/w,v)
                fphases.append(x)
                ffluxes.append(y)
                ferrs.append(e)
            j+=1
        fphases=np.array(fphases)
        ffluxes=np.array(ffluxes)
        ferrs=np.array(ferrs)
    #print('mean err=',ferrs.mean())
    return fphases,ffluxes,ferrs

def weighted_likelihood(ftimes,ffluxes,ferrs,x,sp,constraint_weight=10000,lctype=3,marginalized_noise_pars=None):
    pars=sp.get_pars(x);
    if sp.out_of_bounds(pars): #Hopefully doesn't happen?
        lr=sp.live_ranges()
        lmeans=np.mean(lr,axis=1)
        lwidths=lr[:,1]-lr[:,0]
        print('A par is our of range: dpar/hwidth:\n',(x-lmeans)/lwidths*2)
        parwt=np.sum((pars-sp.get_pars(lmeans))**2)        
        #print(lmeans,parwt)
        return -2e18*(1+parwt*0)
    else:
        mlike=pyHB.likelihood(ftimes,ffluxes,ferrs,pars,lctype=lctype)

        if marginalized_noise_pars is not None:
            alpha,beta0=marginalized_noise_pars
            mlike=-alpha*np.log(1-mlike/beta0)

        if constraint_weight > 0:
            roche_frac=pyHB.test_roche_lobe(pars)
            mlike-=constraint_weight*max([0,roche_frac-1.0])
        #print(x,mlike)
        #print(roche_frac,pars)
        return mlike

def adjust_sectors(data, verbose=False):
    sector_tag='sector'
    sectors=data[sector_tag].unique()
    if verbose: print('sectors',sectors)
    if(len(sectors)>1):
        medians=np.array([np.median(data.flux[data[sector_tag]==sec]) for sec in sectors])
        offsets=medians-medians.mean()
        #print('offsets',offsets)
        for i in range(len(sectors)):
            data.loc[data[sector_tag]==sectors[i],'flux']/=1+offsets[i]/medians.mean()
        if verbose: 
            print('Adjusted sector levels:',offsets)
            print('Adjusted sector factors:',1+offsets/medians.mean())
    return data

#*******************
# Approx symmetries
#*******************

def invert_binary_symmetry_transf(s, randoms): 
    sp=s.getSpace()
    iinc=sp.requireIndex("inc")
    parvals=s.get_params()
    parvals[iinc]=np.pi-parvals[iinc]
    return ptmcmc.state(s,parvals);

def back_view_symmetry_transf(s, randoms):
    #This is based on an observed approximate symmetry if we switch to
    #back side view (omega0->omega0+pi) and also swap the combination
    #logM+2*log_rad_resc between starts 1 and 2.
    #We realize the latter by:
    #logrr1 -> logrr2 + 0.5*(logM2-logM1)
    #logrr2 -> sim
    #i.e. preserving M1,M2
    #
    #The jacobian is trivial
    sp=s.getSpace()
    iom0=sp.requireIndex("omega0")
    im1=sp.requireIndex("logM1")
    im2=sp.requireIndex("logM2")
    irr1=sp.requireIndex("log_rad1_resc")
    irr2=sp.requireIndex("log_rad2_resc")
    parvals=s.get_params()
    parvals[iom0]+=np.pi
    dm=(parvals[im1]-parvals[im2])/2
    if parvals[iom0]>np.pi:parvals[iom0]-=2*np.pi
    newrr1=parvals[irr2]-dm
    newrr2=parvals[irr1]+dm
    parvals[irr1]=newrr1
    parvals[irr2]=newrr2
    return ptmcmc.state(s,parvals);

#############################################################################

class HB_likelihood(ptmcmc.likelihood):
    
    def __init__(self,id,data,period=None,Mstar=None,massTol=0,lensMax=0,eMax=None,maxperiod=14,fixperiod=None,downfac=1.0,constraint_weight=10000,outname="",rep=False,forceM1gtM2=False,rescalesdict={},rescalefac=1.0,viz=False,lctype=3,pins={},prior_dict={},min_per_bin=0,savePfig="",marginalize_noise=False,decimate_level=None,use_syms=False):
        self.bestL=None
        self.forceM1gtM2=forceM1gtM2
        self.lctype=lctype
        
        ## Prepare data ##
        if True:
            data=adjust_sectors(data)
            data[['time','flux','err']].to_csv(outname+'_adjusted.dat')
        self.data=data
        self.constraint_weight=constraint_weight
        self.rep=rep
        #dofold=(period is not None)
        dofold=True
        if dofold:    
            ## Compute period and fold data ##
            if period is not None and period>0: fperiod=period
            else:
                print("Computing folding period")
                #For TESS data we set some reasonable limits on the Period
                minimum_period=0.25
                maximum_period=14
                #Because our lightcurves are not nearly sinusoidal, it is
                #essential to use more terms in the Fourier model underlying
                #the Lomb-Scargle analysis. Otherwise a harmonic is likely to
                #dominate. We also find the standard 5 samples per peak to be
                #insufficient.
                frequency, power = LombScargle(data['time'].values,data['flux'].values,nterms=15).autopower(minimum_frequency=1/maximum_period,maximum_frequency=1/minimum_period,samples_per_peak=50)
                #print('LombScargle samples:',len(power))
                #ilfcut=int(len(power)/20)+1
                ilfcut=int(len(power))
                if0=0
                for i,f in enumerate(frequency):
                    if 1/f < maxperiod:
                        if0=i
                        break
                imax=if0+np.argmax(power[if0:ilfcut])
                pm,p0,pp=power[imax-1:imax+2]
                eps=(pm-pp)/(pm+pp-2*p0)/2
                f0=frequency[imax]
                df=frequency[imax+1]-f0
                fmax=f0+df*eps
                if rep:
                    print('Lomb-Scargle raw f,P=',f0,1/f0)
                    print('             fit f,P=',fmax,1/fmax)
                fperiod=1.0/fmax
                if rep and viz:
                    import matplotlib.pyplot as plt
                    #print('Lomb-Scargle period',fperiod)
                    fig, ax1 = plt.subplots()
                    ax1.plot(frequency,power)
                    ax1.plot(frequency[if0:ilfcut],power[if0:ilfcut])
                    if True: #add inset
                        from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,mark_inset)
                        #ax2=plt.axes([0,0,1,1])
                        #inspos=InsetPosition(ax1,[0.4,0.4,0.5,0.5])
                        #ax2.set_axes_locator(inspos)
                        #mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none", ec='0.5')
                        ax2=ax1.inset_axes([0.45,0.45,0.5,0.5])
                        ax2.plot(frequency,power)
                        ax2.plot(frequency[if0:ilfcut],power[if0:ilfcut])
                        ax2.set_xlim(fmax*0.9,fmax*1.1)
                        ax1.indicate_inset_zoom(ax2)
                    ax1.set_title(str(id))
                    ax1.set_xlabel("frequency (1/day)")
                    
                    if len(savePfig)>0 and savePfig!="None":
                        plt.savefig(savePfig)
                        plt.close()
                    else:
                        plt.show()
                #sys.exit
                    
            doubler=1#set to 2 to fold on  double period
            if(fixperiod is not None):
                ffold=fixperiod*doubler
                fperiod=fixperiod
            else:ffold=fperiod*doubler
            self.fperiod=fperiod
            
            times=data['time'].values
            ndataraw=len(times)
            dts=np.diff(times)
            dt=np.percentile(dts,90)
            obs_time=sum(dts[dts<dt*1.5])
            if rep: print('typical dt=',dt,'observation time',obs_time)
            obs_cycles=obs_time/self.fperiod
            cycles=(times[-1]-times[0])/self.fperiod
            n_per=cycles*downfac
            while n_per<min_per_bin and n_per<ndataraw/2:
                downfac*=2
                n_per=cycles*downfac
            if rep: 
                print('Folding period',ffold)
                print('Data has',cycles,'cycles')
                print('Estimated n per downsampled bin:',n_per,'>',min_per_bin)
            self.fphases,self.ffluxes,self.ferrs=fold_lc(data['time'].values,data['flux'].values,data['err'].values,ffold,downfac=downfac,rep=rep,decimate_level=decimate_level)
            self.ftimes=self.fphases*ffold+int(data['time'].values[0]/ffold)*ffold
            if self.rep:
                array=np.vstack((self.ftimes,self.ffluxes,self.ferrs)).T
                pd.DataFrame(data=array,columns=['ftime','fflux','ferr']).to_csv(outname+'_folded.dat')
                print('Folded data length is',len(self.ftimes))
        else:  #no fold
            self.ftimes=data['time'].values
            self.ffluxes=data['flux'].values
            self.fperiod=period
        #wts=1/self.ferrs**2
        wts=1+0*self.ffluxes
        ffmean=np.sum(self.ffluxes*wts)/np.sum(wts)
        #logFmean=np.log10(ffmean+50)
        if False and self.rep:
            print('ffmean',ffmean)
            print('ftimes',self.ftimes)
            print('ffluxes',self.ffluxes)

        ## Set up parameter space
        ## This piggybacks on the parameter space tool from  pyHB        
        if lctype==2:
            sp=copy.deepcopy(pyHB.sp2)
        elif lctype==3:
            sp=copy.deepcopy(pyHB.sp3)

        #Set pinned params
        for name in pins:
            names=sp.live_names()
            if name in names:
                val=pins[name]
                if val is None:
                    val=np.mean(sp.live_ranges()[names.index(name)])
                if rep: print('Pinning param: '+name+'='+str(val))
                sp.pin(name,val)
            
        #Allow periods within a factor of just over 2% of specified
        sp.reset_range('logP',[np.log10(self.fperiod/1.02),np.log10(self.fperiod*1.02)])
        sp.pin('logP',np.log10(self.fperiod))
        if 'logTanom' in sp.live_names(): sp.pin('logTanom',0)

        #sp.reset_range('log(F+50)',[logFmean-dlogF,logFmean+dlogF])
        #if not allow_blend: sp.pin('blend_frac',-3)
            
        if(Mstar is not None):
            if massTol==0:
                sp.pin('logM1',np.log10(M1))
            else:
                sp.reset_range('logM1',[np.log10(Mstar/(1+massTol)),np.log10(Mstar*(1+massTol))])
        if eMax is not None:
            if eMax>0:sp.reset_range('e',[0,eMax])
            else:sp.pin('e',0)
        self.sp=sp

        #T0 is not meaningful beyond modP
        sp.reset_range('T0',[self.ftimes[0],self.ftimes[0]+self.fperiod])

        #Expand the mass range for test
        sp.reset_range('logM1',[-1.5,2.5])

        #Prep noise_marginialization
        self.marginalized_noise_pars=None
        if marginalize_noise:
            par='ln_noise_resc'
            if par in sp.live_names():
                #note: we assume zero mean on the log noise scaling
                #otherwise we copy the steps from below to set the scale
                #FIXME: It would be much better to move this after, but
                #       then we also need to movev the specification of 
                #       the ptmcmc space to after that.  Not hard...
                scale = sp.live_ranges()[sp.live_names().index(par)][1]
                if prior_dict is not None:
                    if 'ln_noise_resc' in prior_dict:
                        pardict=prior_dict[par]
                        if not isinstance(pardict,dict):raise ValueError('While processing user prior data for parameter "'+par+'". Expected value associated with this par to be a dict, but got '+str(pardict))
                        if 'scale' in pardict:
                            scale=pardict['scale']
                sp.pin('ln_noise_resc',0)
                sigma0 = scale / (len(data['time'].values)/len(self.ftimes)) #matches how prior is set below
                alpha0 = 2 + 1/(np.exp(4*sigma0**2)-1)
                beta0 = np.exp(6*sigma0**2) / ( np.exp(4*sigma0**2) - 1 )
                alpha = alpha0 + len(self.ftimes)/2
                self.marginalized_noise_pars=(alpha,beta0)
                if rep: print('Noise level marginalization activated with sigma0=',sigma0,'-->alpha0,alpha,beta0=',alpha0,alpha,beta0)
            else: 
                raise ValueError('No free noise parameter to marginialize.')
        
        ###Compute SNR
        #pars0=[-10,1,10000,0,0,0,0,logFmean,0]
        #logMlens, Mstar, Pdays, e, sini, omgf, T0overP,logFp50,Fblend=pars0

        #SNR=np.sqrt(-llike0*2)

        print('sp:live',sp.live_names())
        print('mins/maxs',sp.live_ranges().T)
        print('pinvals',sp.pinvals)

        #Set up  stateSpace
        names=sp.live_names()
        ranges=sp.live_ranges()
        npar=len(names)
        space=ptmcmc.stateSpace(dim=npar);
        space.set_names(names);
        wraps=['Omega','Omega0','T0']#T0 not meaningfor beyond T0%period
        centers=[0]*npar
        scales=[1]*npar
        for i  in range(npar):
            name=names[i]
            xmin=ranges[i,0]
            xmax=ranges[i,1]
            if name in wraps:
                space.set_bound(name,ptmcmc.boundary('wrap','wrap',xmin,xmax))
            #else:
            #    space.set_bound(name,ptmcmc.boundary('limit','limit',xmin,xmax)) #May not be needed
            #set prior info
            centers[i]=(xmax+xmin)/2.0
            scales[i]=(xmax-xmin)/2.0
        types=['uni']*npar
        types[names.index('inc')]='polar'
        #These shhould be gaussian, if present
        for pname in ['logTanom', 'mu_1', 'tau_1', 'mu_2', 'tau_2', 'alpha_ref_1', 'alpha_ref_2', 'ln_beam_resc_1', 'ln_beam_resc_2', 'ln_alp_Teff_1', 'ln_alp_Teff_2', 'flux_tune', 'ln_noise_resc']:
            if pname in names:
                types[names.index(pname)]='gaussian'
                sp.reset_range(pname,[float('-inf'),float('inf')])
        if prior_dict is not None:
            for par in prior_dict:
                if par in names:
                    ipar=names.index(par)
                    pardict=prior_dict[par]
                    if not isinstance(pardict,dict):raise ValueError('While processing user prior data for parameter "'+par+'". Expected value associated with this par to be a dict, but got '+str(pardict))
                    if 'center' in pardict:
                        centers[ipar]=pardict['center']
                    if 'scale' in pardict:
                        scales[ipar]=pardict['scale']
                    if 'type' in pardict:
                        types[ipar]=pardict['type']
                        
        #If ln_noise_scale fitting is included, we reduce the prior width if we have already downsampled the data
        if 'ln_noise_resc' in names:
            pname='ln_noise_resc'
            print('Rescaling noise fitting prior scale[ln_noise_resc] =',scales[names.index(pname)],'by the folding factor.')
            scales[names.index(pname)] /= len(data['time'].values)/len(self.ftimes)
            
        
        #some rescaling for better Gaussian proposals
        rescales=[1]*npar
        for name in rescalesdict:
            if name in names:
                rescales[names.index(name)]=rescalesdict[name]
            
        
        rescales=[val*rescalefac for val in rescales]
        
        if use_syms: 
            #Add information about potential symmetries
            if rep: print("Applying symmetry transform.")
            space.addSymmetry(ptmcmc.involution(space,"invert_binary",0,invert_binary_symmetry_transf))
            space.addSymmetry(ptmcmc.involution(space,"back_view",0,back_view_symmetry_transf))

        print("HB_likelihood::setup: space="+space.show())
        self.basic_setup(space, types, centers, scales, rescales);


    def evaluate_log(self,s):
        params=s.get_params()
        done=False
        if self.forceM1gtM2:
            #Here we hard-code M1,M2 indices, could do better...
            im1=0
            im2=1
            if params[im2]>params[im1]:
                result = -1e100
                done=True
        if not done:
            #print(params)
            result=weighted_likelihood(self.ftimes,self.ffluxes,self.ferrs,params,self.sp,self.constraint_weight,self.lctype,marginalized_noise_pars=self.marginalized_noise_pars)

        if False:
            global count
            print(count)
            count+=1
            print("state:",s.get_string())
            print("  logL={0:.13g}".format(result))
        if self.bestL is None or result>self.bestL:
            self.bestX=params
            self.bestL=result
        return result

    def report(self):
        print('Best fit results:')
        print('  pars =',bestXp)
        print('  SNR =',self.SNR)
        print('  chi2 =',-bestL)
        print('  fit percent = %5.2f'%((1-bestL/llike0)*100.0))

    def getModels(self,parslist):
        if self.lctype==2:
            models=[pyHB.lightcurve2(self.ftimes,self.sp.get_pars(pars)) for pars in parslist]
        elif self.lctype==3:
            models=[pyHB.lightcurve3(self.ftimes,self.sp.get_pars(pars)) for pars in parslist]
        return models
        
count=0

def read_data_from_sector_files(id,basepath,edgeskip=0.5,allowsecs=None,trueerr=1.0,tmin=None,tmax=None):
    if tmin is not None or tmax is not None: print('read_data_from_sector_files: Time limits are not yet implemented and will be ignored!')
    if allowsecs is None:allowsecs=range(1,20)
    #print('id=',id)
    datafiles=glob.glob(basepath+'/*/*/tesslc_'+str(id)+'.pkl')
    found_in_sectors=[]
    df=pd.DataFrame(columns=['sector','time','flux','err'])
    for path in datafiles:
        data=pickle.load(open(path,'rb'))
        sector=int(re.findall(r'sector_(\d*)',path)[0])
        found_in_sectors+=[sector]
        if not sector in allowsecs:continue
        flux = data[6] 
        time = data[4]
        fluxerr = data[8]*trueerr
        dt=time[1]-time[0]
        iedgeskip=int(edgeskip/dt)
        #print('iedgeskip',iedgeskip)
        if(iedgeskip>0):#process edge skips
            keeps=np.array([True]*len(time))
            keeps[0:iedgeskip]=False
            keeps[-iedgeskip:]=False
            for i in range(1,len(time)):
                if keeps[i] and time[i]-time[i-1]>0.5: #gap >1/2 day
                    #print('cut detected at t =',time[i])
                    #print(time[i-1],time[i],time[i]-time[i-1])
                    keeps[i-iedgeskip:i]=False
                    #print('skipping from',time[i-iedgeskip],'to',time[i+iedgeskip])
                    keeps[i:i+iedgeskip]=False
            flux=flux[keeps]
            time=time[keeps]
            fluxerr=fluxerr[keeps]
        #print('time',time)
        ddf=pd.DataFrame([[sector,time[i],flux[i],fluxerr[i]] for i in range(len(time))],columns=['sector','time','flux','err'])
        #print(ddf)
        df=df.append(ddf,ignore_index=True)
        df=df.sort_values('time')
        #print(df)
    print("Found in sectors",found_in_sectors)
    return df

def read_data_from_file(id,path,edgeskip=0.5,trueerr=1.0,tmin=None,tmax=None,weight_err_dt=True,verbose=False):
    #trueerr is our estimate of the 1-sigma error on the data at flux=1, 
    #  otherwise err ~ sqrt(flux)
    #This routine intended for data that are normalized to near-unity flux
    #If weight_err_dt, then trueerr applies if the local cadence is 
    #  30min (1/48 day) otherwise the unit flux err is trueerr/sqrt(dt). 
    #  At longer gaps dt is assumed 1/48. 
    if verbose: print('Reading data from file:',path)
    if path.endswith(".fits"):
        if verbose: print('This seems to be a FITS file.')
        from astropy.io import fits
        f=fits.open(path)
        time=f[1].data["TIME"]
        flux=f[1].data["CORR_FLUX"]
        flux=flux/np.median(flux)
        if verbose: print('time.shape',time.shape)
        data=np.column_stack((time,flux,0*flux))  #Maybe there is useful error info??
        if verbose: print('data.shape',data.shape)
    else:
        if verbose: print('Will read as text data.')
        data=np.genfromtxt(path,skip_header=1)
        #assumed format t flux other_flux flag
        data=data[data[:,3]==0] #clean out the flagged rows
    if tmin is not None: data=data[data[:,0]>=tmin]
    if tmax is not None: data=data[data[:,0]<=tmax]
    flux = data[:,1] 
    time = data[:,0]
    cadfac=np.diff(time,prepend=2*time[0]-time[1])*48
    fluxerr = np.sqrt(flux/np.minimum(1,cadfac))*trueerr
    #dt=time[1]-time[0]    
    #iedgeskip=int(edgeskip/dt)
    #print('iedgeskip',iedgeskip)
    if(edgeskip>0):#process edge skips
        #keeps=np.array([True]*len(time))
        #keeps[0:iedgeskip]=False
        #keeps[-iedgeskip:]=False
        keeps = np.logical_and( time-time[0]>edgeskip, time[-1]-time>edgeskip )
        for i in range(1,len(time)):
            if keeps[i] and time[i]-time[i-1]>0.5: #gap >1/2 day
                if verbose: 
                    print('cut detected at t =',time[i])
                    print(time[i-1],time[i],time[i]-time[i-1])
                #keeps[i-iedgeskip:i]=False
                #keeps[i:i+iedgeskip]=False
                #print('skipping from',time[i-iedgeskip],'to',time[i+iedgeskip])
                keeps=np.logical_and(keeps,
                                     np.logical_or(
                                         time<time[i-1]-edgeskip,
                                         time>time[i]+edgeskip    ) )
                if verbose: print('skipping from',time[i-1]-edgeskip,'to',time[i]+edgeskip)
        flux=flux[keeps]
        time=time[keeps]
        fluxerr=fluxerr[keeps]
    sector=0
    df=pd.DataFrame([[sector,time[i],flux[i],fluxerr[i]] for i in range(len(time))],columns=['sector','time','flux','err'])
    return df



#//***************************************************************************************8
#//main test program
def main(argv):

    ptmcmc.Init()

    #//prep command-line options
    #Options opt(true);
    opt=ptmcmc.Options()
    
    #//Add some command more line options

    #data specific flags
    opt.add("seed","Pseudo random number grenerator seed in [0,1). (Default=-1, use clock to seed.)","-1")
    ##opt.add("precision","Set output precision digits. (Default 13).","13")
    opt.add("outname","Base name for output files (Default 'mcmc_output').","mcmc_output")
    
    #data_model flags
    opt.add("data_style","Provide data model flags as a json string","")
    opt.add("id","TIC_ID","")
    opt.add("period","Set fixed period for folding and model. (Default None)","None")
    opt.add("datafile","Explicitly indicate the data file.","")
    opt.add("Mstar","Override TIC (primary) star mass. (Default None)","None")
    opt.add("sectors","Only use these sectors (comma-separated)","")
    opt.add("tlimits","Set tmin,tmax time limits, outside which to ignore data. (def=none)","")    
    opt.add("noTIC","Set to 1 to skip any online query about TIC id.","0")
    opt.add("trueerr","scalefactor of the error following JS's code. (def=1)","1")
    opt.add('min_per_bin','Minimum mean number of samples per bin after folding and downsampling.(Default 0)','0')
    opt.add('decimate_level','Level (0-15) to apply in decimat.py data decimation algorithm. Larger numbr is more aggressive. Overrides native downsampling. (Default none.)','-1')
    opt.add('edgeskip','Size of region to exclude from data near data gaps in days. (Default=0.5)','0.5')

    #//Create the sampler
    #ptmcmc_sampler mcmc;
    #hb model style flags
    opt.add("hb_style","Provide heartbeat model style flags as a json string","")
    opt.add("datadir","directory where processed sector data files are located",".")
    opt.add("eMax","Set max value for eccentricity. (Default 0.95)","0.95")
    opt.add("massTol","Uniform confidence width factor for TIC mass. (Default 0.2)","0.2")
    opt.add("plotSamples","File with samples to plot, (eg chain output)","")
    opt.add("nPlot","If plotting samples, how many to sample curves to include","20")
    opt.add("downfac","Extra downsampling factor in lightcurve folding.","1")
    opt.add("Roche_wt","Weight factor for Roche-limit constraint (def 10000).","10000")
    opt.add("M1gtM2","Set to 1 to force M1>M2. (def=0)","0")
    #opt.add('blend','Set to 1 to vary the blending flux','0')
    opt.add('lctype','Light curve model version. Options are 2 or 3. (Default 3)','3')
    opt.add('pins','json formatted string with parname:pinvalue pairs','{}')
    opt.add('marginalize_noise','Set to 1 to analytically marginalize noise scaling.','0')
    opt.add('rescales','Rescaling factors to base proposals, etc., as json formatted string with parname:value pairs','{}')

    #for MCMC
    opt.add("mcmc_style","Provide mcmc flags as a json string","{}")
    opt.add('rescalefac','Rescale factor for gaussian proposals. Default=1','1')

    #Other
    opt.add("savePfig","Location to save period fig file in plotting mode (Default: interactive display).","")
    opt.add("saveLfig","Location to save lightcurve fig file in plotting mode (Default: interactive display).","")

    s0=ptmcmc.sampler(opt)
    rep=s0.reporting()
    opt.parse(argv)
    
    #Process flags:
    intf=lambda x: int(x)
    pval=lambda name:opt.value(name)
    getpar=lambda name,typ:style.get(name,typ(opt.value(name)) if len(opt.value(name))>0 or typ==str else None)
    getboolpar=lambda name:style.get(name,(opt.value(name)!='0'))
    getNpar=lambda name,typ:style.get(name,typ(opt.value(name)) if opt.value(name)!='None' else None)


    #basic
    outname=opt.value('outname')
    seed=float(opt.value('seed'))

    #viz only option
    do_plot = opt.value('plotSamples')!="" or int(opt.value('nPlot'))==0    
    ncurves=int(opt.value('nPlot'))
    sampfiles=opt.value('plotSamples')
    saveLfig=opt.value('saveLfig')
    savePfig=opt.value('savePfig')
    
    #data
    style={}
    if opt.value('data_style')!='': 
        style=opt.value('data_style')
        if style.startswith('{'):
            style=json.loads(style)
        else:
            with open(style,'r') as sfile:
                style=json.load(sfile)
    id=getpar('id',int)
    datadir=getpar('datadir',str)
    massTol=getpar('massTol',float)
    noTIC=getboolpar('noTIC')
    Mstar=getNpar('Mstar',float)
    sectors=getpar('sectors',str)
    tlimits=getpar('tlimits',str)
    trueerr=getpar('trueerr',float)
    decimate_level=getpar('decimate_level',int)
    datafile=getpar('datafile',str)
    period=getNpar('period',float)
    if period is None and opt.value('period') != 'None':period=float(opt.value('period'))
    downfac=getpar('downfac',float)
    min_per_bin=getpar('min_per_bin',float)
    if min_per_bin <=0 and opt.value('min_per_bin')!='0':min_per_bin=float(opt.value('min_per_bin'))
    if rep:print('decimate-level',decimate_level)
    edgeskip=getpar('edgeskip',float)
    if edgeskip ==0.5 and opt.value('edgeskip')!='0.5':edgeskip=float(opt.value('edgeskip'))
    datastyle=style

    # HB model style
    style={}
    if rep: print('hb_style',opt.value('hb_style'))
    if opt.value('hb_style')!='':
        style=opt.value('hb_style')
        if style.startswith('{'):
            style=json.loads(style)
        else:
            with open(style,'r') as sfile:
                style=json.load(sfile)
    if rep: print('processed:',"'"+json.dumps(style)+"'")
    Roche_wt=getpar('Roche_wt',float)
    pindict=getpar('pins',json.loads)
    eMax=getpar('eMax',float)
    forceM1gtM2=getboolpar('M1gtM2')
    marginalize_noise=getboolpar('marginalize_noise')
    use_syms=False #May change based on mcmc_options
    rescalesdict=getpar('rescales',json.loads)
    #blend=getboolpar('blend')
    lctype=getpar('lctype',int)
    prior_dict=style.get('prior',{})
    if rep: print('Roche_wt,emax,lctype:',Roche_wt,eMax,lctype)

    #Process mcmc options
    style=opt.value('mcmc_style')
    if style.startswith('{'):
        style=json.loads(style)
    else:
        with open(style,'r') as sfile:
            style=json.load(sfile)
    mcmc_options=style
    hb_mcmc_flags=['rescalefac']
    style={}
    optlist=[]
    no_arg_flags=['de_mixing','gauss_temp_scaled','prop_adapt_more','pt_reboot_grad']
    keys=list(mcmc_options.keys())
    for key in keys:
        if key in hb_mcmc_flags:
            style[key]= mcmc_options[key]
            del mcmc_options[key]
    for key in mcmc_options:            
        arg=mcmc_options[key]
        if key in no_arg_flags:
            if arg: optlist.append('--'+key)
        else:
            optlist.append('--'+key+'='+str(arg))
    rescalefac=getpar('rescalefac',float)
    if rep: print('rescalefac=',rescalefac)
    if 'sym_prop_frac' in mcmc_options:
        if rep: print('sym_prop_frac=',mcmc_options['sym_prop_frac'])
        if mcmc_options['sym_prop_frac']>0:
            use_syms=True
    
    #Pass to ptmcmc
    opt.parse(optlist)
    
    #Get TIC catalog info:    
    if noTIC:
        TICData = None
    else:
        try:
            TICData = Catalogs.query_object('TIC '+str(id),radius=0.0011,catalog='TIC')#0.011 deg is 2 px
            if rep: print(TICData['ID','Tmag','Vmag','ra','dec','d','objType','lumclass','Teff','mass','rad'][0])
            #print(TICData.columns)
        except: 
            if rep:print("**TIC Query Failed**")
            if rep:print("id=",id)
            TICData=None

    
    if TICData is not None:
        if rep:print('Vmag',TICData['Vmag'][0], 'Teff',TICData['Teff'][0])

    Rstar=None
    Mstar=None
    global useM
    if useM:
        if TICdata is None:
            useM=False
            if rep:print('Cannot "useM" since I have no TIC data! Overriding')
        if massTol==0 and str(TICData['rad'][0]).isnumeric:  #Don't fix radius if we are varying the mass
            Rstar=TICData['rad'][0]
            if rep:print('Rstar=',Rstar)        
        if Mstar is None and not np.isnan(float(TICData['mass'][0])):
            Mstar=TICData['mass'][0]
            #print('float(Mstar)',float(Mstar))
            if rep:print('Mstar(TIC)=',Mstar)
        if rep:print('Mstar=',Mstar)

    allowsecs=None
    if sectors!='':
        allowsecs=sectors.split(',')
        allowsecs=[int(sec) for sec in allowsecs]
    tmin=None
    tmax=None
    if tlimits!='':
        tlims=tlimits.split(',')
        if len(tlims)<2:tlims.append('')
        if tlims[0].isnumeric():tmin=float(tlims[0])
        if tlims[1].isnumeric():tmax=float(tlims[1])
        if rep: print('Constraining',tmin,'< t <',tmax)
        
    #Prepare the data:

    if datafile=='':
        dfg=read_data_from_sector_files(id,datadir,edgeskip=0.5,allowsecs=allowsecs,trueerr=trueerr,tmin=tmin,tmax=tmax)
    else:
        if datafile.startswith('/'):
            filepath=datafile
        else:
            filepath=datadir+'/'+datafile
        dfg=read_data_from_file(id,filepath,edgeskip=edgeskip,trueerr=trueerr,tmin=tmin,tmax=tmax,verbose=rep)
    if rep:
        print('Trimmed data length is',len(dfg))
        dfg[['time','flux','err']].to_csv(outname+'_trimmed.dat')

    #//Create the likelihood
    fixperiod=None
    if period is not None and period<0:
        period=-period
        fixperiod=period
        
    
    like=HB_likelihood(id,dfg,period,Mstar,massTol=massTol,eMax=eMax,maxperiod=20,fixperiod=fixperiod,downfac=downfac,constraint_weight=Roche_wt,outname=outname,rep=rep,forceM1gtM2=forceM1gtM2,rescalesdict=rescalesdict,rescalefac=rescalefac,viz=do_plot,lctype=lctype,pins=pindict,prior_dict=prior_dict,min_per_bin=min_per_bin,savePfig=savePfig,marginalize_noise=marginalize_noise,decimate_level=decimate_level,use_syms=use_syms)

    if fixperiod is None:
        dataPfile="data_style_Pfit.json"
        if len(savePfig)>0 or not os.path.exists(dataPfile):
            #Only overwrite when savePfig flag is set
            datastyle['period']=like.fperiod
            datastyle['period-note']='period determined automatically by Lomb-Scargle'
            with open(dataPfile,'w') as dpf:
                json.dump(datastyle,dpf,indent=4)
    do_residual=True
    resid_rescaled=False
    if(do_plot):
        #Plot samples instead of running chains
        t=like.ftimes
        ts=np.linspace(t[0],t[-1],300)
        data=like.ffluxes
        if ncurves>0:
            if sampfiles.startswith('[') and sampfiles.endswith(']'):
                if ',' in sampfiles:
                    sampfiles=sampfiles[1:-1].split(',')
                else:
                    sampfiles=sampfiles[1:-1].split()
            else:
                sampfiles=[sampfiles]
            nmaxs=[None for x in sampfiles]
            for i in range(len(sampfiles)):
                if ':' in sampfiles[i]:
                    sampfiles[i],nmaxs[i]=sampfiles[i].split(':')
                    if i>0 and len(sampfiles[i])==0:sampfiles[i]=sampfiles[i-1]
                    if len(nmaxs[i])==0:nmaxs[i]=None
            print('samples files:',sampfiles)
            print('sample nmaxs:',nmaxs)
            modelsets=[]
            residsets=[]
            for i in range(len(sampfiles)):
                sfile=sampfiles[i]
                n=nmaxs[i]
                print('Processing',sfile)
                chain=ptmcmc_analysis.chainData(sfile,useLike=True)
                if n is None or not '%' in n and int(n)>chain.getSteps(): 
                    n=chain.getSteps()
                elif '%' in n:
                    n=int(min(100,float(n[:-1]))/100*chain.getSteps())
                else: n=int(n)
                nmaxs[i]=str(n)
                rows,samples=chain.get_samples(ncurves,nmax=n,good_length=n//10,return_rows=True)
                print('sample_rows:',rows)
                colnames=chain.names
                for att in ['samp','post','like']:
                    if att in colnames:
                        print('mean',att,np.mean(chain.data[rows][:,colnames.index(att)]))
                print('mean pars:',np.mean(samples,axis=0))
                print('std pars:',np.std(samples,axis=0))
                #print("samples:")
                #for sample in samples:print(sample)
                #cnames=chain.names[chain.names.index('post')+1:]
                cnames=chain.names[chain.ipar0:]
                idx=[cnames.index(name) for name in like.sp.live_names()]
                print(idx,cnames,like.sp.live_names())
                psamples=[like.sp.get_pars([pars[idx[i]] for i in range(len(idx))]) for pars in samples]
                if lctype==2:
                    lightcurve=pyHB.lightcurve2
                elif lctype==3:
                    lightcurve=pyHB.lightcurve3
                models=[lightcurve(ts,p[:-1]) for p in psamples]
                modelsets.append(models)
                roches=[pyHB.test_roche_lobe(p,verbose=True) for p in psamples[-1:] ]
                print('roche fracs:',roches)
                if do_residual:
                    resc=[1]*len(psamples)
                    if 'ln_noise_resc' in cnames:
                        resid_rescaled=True
                        iresc=cnames.index('ln_noise_resc')
                        resc=np.exp([p[iresc] for p in psamples])
                    #resids=[(data-lightcurve(t,p[:-1])) for p in psamples]
                    resids=[(data-lightcurve(t,psamples[j][:-1]))/resc[j] for j in range(len(psamples))]
                    residsets.append(resids)
        else: modelsets =[]
        import matplotlib.pyplot as plt
        if do_residual:
            fig, axs = plt.subplots(2, 1, figsize=[6.4, 6.4],sharex=True)
            fig.subplots_adjust(hspace=0)
            ax=axs[0]
            rax=axs[1]
        else:
            fig, axs = plt.subplots(1, 1)
            ax=axs
        plt.subplots_adjust(bottom=0.25)
        lims0=None
        lims1=None
        ax.errorbar(t,data,yerr=like.ferrs,ls='None',label='data')
        if do_residual:rax.errorbar(t,data*0,yerr=like.ferrs,ls='None')
        colors=['r','b','g','y','m','c','k']
        for i in range(len(modelsets)):
            label=sampfiles[i]+':'+nmaxs[i]
            col=colors[i]
            for model in modelsets[i]:
                ax.plot(ts,model,col,alpha=0.2,label=label)
                lims0=autoscale(model,lims0)                
                label=None
            if do_residual:
                for resid in residsets[i]:
                    rax.plot(t,resid,col,ls='None',marker='.',alpha=0.2,label=label)
                    lims1=autoscale(resid,lims1)                

        rawftimes=like.data['time']%(like.fperiod)+int(like.data['time'][0]/like.fperiod)*like.fperiod
        #-0*like.data['time'][0]%(like.fperiod)+like.ftimes[0]
        
        ax.plot(rawftimes,like.data['flux'],'k.',ls='None',markersize=0.5,label='raw data')
        lims0=autoscale(like.data['flux'],lims0)

        ax.set_ylim(lims0)
        rax.set_ylim(lims1)

        leg=plt.figlegend(loc='upper center',fontsize='small',bbox_to_anchor=(0.5, 0.20))
        #leg=ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True)
        for lh in leg.legendHandles: 
            lh.set_alpha(1)
        #Title
        title=str(id)
        if do_residual:
            title+=' with residual'
            if resid_rescaled:
                title+=' (noise model scaled)'
        ax.set_title(title)
        #plt.tight_layout()
        if len(saveLfig)>0:
            plt.savefig(saveLfig)
            plt.close()
        else:
            plt.show()
        return
        
    if seed<0:seed=np.random.random();
    
    #//report
    if rep:print("\noutname = '"+outname+"'")
    
    #//Should probably move this to ptmcmc/bayesian
    ptmcmc.resetRNGseed(seed);
    space=like.getObjectStateSpace();
    if rep:print("like.nativeSpace=\n"+space.show())
    Npar=space.size();
    if rep:print("Npar=",Npar)
    s0.setup(like)


    s=s0.clone();
    s.initialize();
    print('initialization done')
    s.run(outname,0);

def autoscale(y,lims=None,tol=0.0025,expand=0.10):
    #Cut up to tol fraction of the extreme data before autoscaling
    ysort=np.sort(y)
    icut=int(tol*len(y))
    ymin,ymax=ysort[icut],ysort[-(1+icut)]
    dy=(ymax-ymin)*expand
    ymin=ymin-dy
    ymax=ymax+dy
    if lims is not None:
        ymin=min(lims[0],ymin)
        ymax=max(lims[1],ymax)
    return [ymin,ymax]

if __name__ == "__main__":
    import sys
    argv=sys.argv[:]
    del argv[0]
    main(argv)
