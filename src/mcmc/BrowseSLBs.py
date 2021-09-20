from astropy.stats import LombScargle
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astroquery.mast import Catalogs,Observations
import re

import sys
dir='../../TessSLB/src/LightCurveCode'
if dir not in sys.path: sys.path.append(dir)
import copy
import pyAvst
import importlib
import warnings
import scipy
import scipy.linalg
import scipy.optimize as opt
warnings.simplefilter("ignore")
importlib.reload(pyAvst)

from timeout import timeout

def loop_search(f,sp,Rstar=None):
    maxit=9600
    #maxit=4800
    tol=0.001
    bestL=float('-inf')
    bestX=[]
    ncycle=6
    #ncycle=5
    for cycle in range(ncycle):
        icount=int(2**(ncycle-1-cycle))
        #icount=1
        imaxit=int(maxit/icount)
        if cycle==0:
            x0s=[sp.draw_live() for i in range(icount)]
        else:
            order=np.argsort([results[i].fun for i in range(len(results))])
            print('best=',[results[i].fun for i in order[:icount]])
            x0s=[results[i].x for i in order[:icount]]
        results=[]
        #print('\nicount',icount,'imaxit',imaxit)

        for i in range(icount):
            try:
                with timeout(seconds=2**(cycle+1)):
                    #print(x0s[i])
                    opt_res=opt.minimize(f,x0s[i],method='Nelder-Mead',options={'maxiter':imaxit},bounds=sp.live_ranges())
                    #opt_res=opt.minimize(f,x0s[i],method='Nelder-Mead',options={'maxiter':imaxit},tol=tol,bounds=sp.live_ranges())
                    #opt_res=opt.minimize(f,x0s[i],method='Newton-CG',options={'maxiter':imaxit},tol=tol,bounds=sp.live_ranges())
                #print(opt_res.success,opt_res.nit,opt_res.nfev)
                X=opt_res.x.copy()
                L=-f(opt_res.x)
                #print(X,L)
                #hilltest=pyAvst.test_roche_lobe(sp.get_pars(X),Rstar)
                #print('Hill test:',hilltest)
                #if hilltest<0.8:
                if True:
                    results.append(opt_res)
                    if L>bestL:
                        bestL=L
                        bestX=X
            except(TimeoutError):
                #print('Timed out.')
                pass
    return bestL,bestX

def global_search(f,sp,Rstar=None,Temp=50):
    maxit=150
    tol=0.001
    count=16
    bestL=float('-inf')
    bestX=[]
    ncycle=1
    for cycle in range(ncycle):
        icount=1
        imaxit=maxit
        temp=Temp*4**(ncycle-1-cycle)
        if cycle==0:
            x0s=[sp.draw_live() for i in range(icount)]
        else:
            order=np.argsort([results[i].fun for i in range(len(results))])
            print('best=',[results[i].fun for i in order[:icount]])
            x0s=[results[i].x for i in order[:icount]]
        results=[]
        print('\nicount',icount,'imaxit',imaxit)

        for i in range(icount):
            try:
                with timeout(seconds=2**(cycle+1)):
                    #print(x0s[i])
                    opt_res=opt.basinhopping(f,x0s[i],T=temp,niter=imaxit,stepsize=0.1,disp=True,minimizer_kwargs={'method':'Nelder-Mead'})
                    #opt_res=opt.minimize(f,x0s[i],method='Newton-CG',options={'maxiter':imaxit},tol=tol,bounds=sp.live_ranges())
                #print(opt_res.success,opt_res.nit,opt_res.nfev)
                X=opt_res.x.copy()
                L=-f(opt_res.x)
                #print(X,L)
                #hilltest=pyAvst.test_roche_lobe(sp.get_pars(X),Rstar)
                #print('Hill test:',hilltest)
                #if hilltest<0.8:
                if True:
                    results.append(opt_res)
                    if L>bestL:
                        bestL=L
                        bestX=X
            except(TimeoutError):
                #print('Timed out.')
                pass
    return bestL,bestX




def fold_lc(times,fluxes,errs,Pfold):
    phases=(np.array(times)/Pfold)%1
    isort=np.argsort(phases)
    phases=phases[isort]
    fluxes=np.array(fluxes)[isort]
    errs=np.array(errs)[isort]
    nold=len(times)
    groupwidth=(times[-1]-times[0])*(1+0.1/nold)/nold/Pfold
    groupwidth*=1.0
    #print('mean errs=',errs.mean())
    #print('groupwidth=',groupwidth, 'mean group size=',groupwidth*nold)
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
            #print(xs,ys,es,'-->',x,y,e)
            fphases.append(x)
            ffluxes.append(y)
            ferrs.append(e)
        j+=1
    fphases=np.array(fphases)
    ffluxes=np.array(ffluxes)
    ferrs=np.array(ferrs)
    #print('mean err=',ferrs.mean())
    return fphases,ffluxes,ferrs

def f(id,lcs):
    if 'id' in lcs:
        data=lcs[lcs['id']==id].copy()
    else: #assume just one lightcurve
        data=lcs
    try: 
        TICData = Catalogs.query_object(str(id),radius=0.011,catalog='TIC')#0.011 deg is 2 px
        print(TICData['ID','Tmag','ra','dec','d','objType','lumclass','Teff','mass','rad'][0])
    except: 
        print("**TIC Query Failed**")
        TICData=None  
    plt.rcParams.update({'font.size': 22})
    if 'sector' in data:
        sector_tag='sector'
    else:
        print('Confidence =',data['summed_confidence'].unique())
        print('    Scores =',data['confidence_score'].unique())
        sector_tag='confidence_score'
    sectors=data[sector_tag].unique()
    #print('sectors',sectors)
    if(len(sectors)>1):
        medians=np.array([np.median(data.flux[data[sector_tag]==sec]) for sec in sectors])
        offsets=medians-medians.mean()
        #print('offsets',offsets)
        for i in range(len(sectors)):
            #print(offsets[i])
            #block=(data[sector_tag]==sectors[i])
            #flux=data.flux[block]
            #print('bef',flux)
            #flux=flux-offsets[i]
            #print('aft',flux)
            #data.loc[block,'flux']-=offsets[i]
            data.loc[data[sector_tag]==sectors[i],'flux']-=offsets[i]
            #pd.Series(flux,index=data[block].index)
        print('Adjusted sector levels:',offsets)
    fig, ax = plt.subplots(1,3,figsize=(48,16)) 
    ax[0].plot(data['time'].values,data['flux'].values)
    #plt.show()
    frequency, power = LombScargle(data['time'].values,data['flux'].values).autopower()
    #plt.plot(frequency, power)
    #plt.show()
    ilfcut=int(len(power)/20)+1
    #print('ilfcut',ilfcut,1/ilfcut)
    if0=0
    maxperiod=14 #by eye, seems hard to trust anything longer
    for i,f in enumerate(frequency):
        if 1/f < maxperiod:
            if0=i
            break
    ax[1].plot(1/frequency[if0:ilfcut], power[if0:ilfcut])
    #plt.show()
    fmax=frequency[if0:ilfcut][np.argmax(power[if0:ilfcut])]/2.0
    fperiod=1/fmax
    print('Folding period',1/fmax)
    fphases,ffluxes,ferrs=fold_lc(data['time'].values,data['flux'].values,data['err'].values,1/fmax)

    #f0=ffluxes.mean()
    #ffluxes=ffluxes/f0
    #ferrs=ferrs/f0
    ax[2].errorbar(fphases.tolist()+(fphases+1).tolist(),ffluxes.tolist()*2,yerr=ferrs.tolist()*2,marker='.',linestyle=' ',ms=10,alpha=0.6)
    plt.show()
    #plt.errorbar(fphases,ffluxes,ferrs)
    #print('clean=',(id in lfids))
    #print(TICData['GAIA'])
    #print(len(TICData))
    if TICData is not None:
        #print('GAIA obs within 2px:')
        radec=' '.join([str(x) for x in TICData['ra','dec'][0]])
        try:
            GAIAData=Catalogs.query_object(radec,unit='deg',radius=0.011,catalog='Gaia')
            #print(GAIAData.columns)
            print('GAIA obs within 2px:')
            GAIAData['parallax_over_error'].name='plax_/_err'
            GAIAData['radial_velocity'].name='rad_vel'
            GAIAData['phot_g_mean_mag'].name='g_mag'
            GAIAData['delta_ra']=GAIAData['ra']-TICData[0]['ra']
            GAIAData['delta_dec']=GAIAData['dec']-TICData[0]['dec']
            GAIAData['delta_ra'].format=GAIAData['delta_dec'].format=GAIAData['g_mag'].format=GAIAData['parallax'].format=GAIAData['plax_/_err'].format=GAIAData['pmra'].format=GAIAData['pmdec'].format=GAIAData['rad_vel'].format='9.5f'
            GAIAData['designation','delta_ra','delta_dec','g_mag','parallax','plax_/_err','pmra','pmdec','rad_vel'].pprint(max_width=200)
        except:
            print("**GAIA Query Failed**")
    if(False):
        radec=coord.SkyCoord(TICData['ra'],TICData['dec'],unit='deg')
        #print(radec)
        SimData=Simbad.query_region(radec)       
        print('Simbad\n',SimData)

def constrain(ids,lcs,VmagLimit=11,TeffMin=3500,TeffMax=5200):
    constrained_ids=[]
    for id in ids:
        data=lcs[lcs['id']==id].copy()
        try: 
            TICData = Catalogs.query_object(str(id),radius=0.011,catalog='TIC')#0.011 deg is 2 px
            print(TICData['ID','Tmag','Vmag','ra','dec','d','objType','lumclass','Teff','mass','rad'][0])
            #print(TICData.columns)
        except:
            print("**TIC Query Failed**")
            TICData=None  

        if TICData is not None and TICData['Vmag'][0]<VmagLimit and TICData['Teff'][0]>TeffMin and TICData['Teff'][0]<TeffMax:
            constrained_ids.append(id)
            print('**** Good ****')
        else:
            print('** Rejected **')
    return constrained_ids

def weighted_likelihood(ftimes,ffluxes,ferrs,x,sp,Rstar):
    pars=sp.get_pars(x);
    if sp.out_of_bounds(pars):
        lmeans=np.mean(sp.live_ranges(),axis=1)
        parwt=np.sum((pars-sp.get_pars(lmeans))**2)
        #print(lmeans,parwt)
        return 2e18*(1+parwt*0)
    else:
        roche_frac=pyAvst.test_roche_lobe(pars,Rstar=Rstar)
        mlike=-pyAvst.likelihood(ftimes,ffluxes,ferrs,pars,ulambda=0)+10000*max([0,roche_frac-0.8])
        #print(x,mlike)
        #print(roche_frac,pars)
        return mlike

import glob
def g(id,lcs,fit=False,method='loop',useM=True,massTol=0,lensMax=0,eMax=0,doGAIA=False,maxperiod=14,skipGiants=False,minCycles=0,SNRonly=False):
    if 'id' in lcs:
        data=lcs[lcs['id']==id].copy()
    else: #assume just one lightcurve
        data=lcs
    try: 
        TICData = Catalogs.query_object(str(id),radius=0.011,catalog='TIC')#0.011 deg is 2 px
        print(TICData['ID','Tmag','Vmag','ra','dec','d','objType','lumclass','Teff','mass','rad'][0])
        #print(TICData.columns)
    except: 
        print("**TIC Query Failed**")
        print("id=",id)
        TICData=None  
    SNR=0

    if(len(data)==0):# not found
        print('No data found')
        return 0
    
    if TICData['lumclass'] is not None and TICData['lumclass'][0]=='GIANT':
        print('Skipping giant')
        return 0
    
    if TICData is not None:
        print('Vmag',TICData['Vmag'][0], 'Teff',TICData['Teff'][0])

    #if TICData is not None and TICData['Vmag'][0]<10.5 and TICData['Teff'][0]>3500 and TICData['Teff'][0]<5200:
    if True:
        Rstar=None
        Mstar=None
        if useM:
            if massTol==0 and str(TICData['rad'][0]).isnumeric:  #Don't fix radius if we are varying the mass
                Rstar=TICData['rad'][0]
                print('Rstar=',Rstar)
            Mstar=None
            if not np.isnan(float(TICData['mass'][0])):
                Mstar=TICData['mass'][0]
                #print('float(Mstar)',float(Mstar))
                print('Mstar=',Mstar)
            plt.rcParams.update({'font.size': 22})
        if 'sector' in data:
            sector_tag='sector'
        else:
            print('Confidence =',data['summed_confidence'].unique())
            print('    Scores =',data['confidence_score'].unique())
            sector_tag='confidence_score'
        sectors=data[sector_tag].unique()
        #print('sectors',sectors)
        if(len(sectors)>1):
            medians=np.array([np.median(data.flux[data[sector_tag]==sec]) for sec in sectors])
            offsets=medians-medians.mean()
            #print('offsets',offsets)
            for i in range(len(sectors)):
                data.loc[data[sector_tag]==sectors[i],'flux']-=offsets[i]
            print('Adjusted sector levels:',offsets)
        fig, ax = plt.subplots(1,3,figsize=(48,16)) 
        ax[0].plot(data['time'].values,data['flux'].values)
        #plt.show()
        frequency, power = LombScargle(data['time'].values,data['flux'].values).autopower()
        #plt.plot(frequency, power)
        #plt.show()
        ilfcut=int(len(power)/20)+1
        #print('ilfcut',ilfcut,1/ilfcut)
        if0=0
        #maxperiod=14 #by eye, seems hard to trust anything longer
        for i,f in enumerate(frequency):
            if 1/f < maxperiod:
                if0=i
                break
        ax[1].plot(1/frequency[if0:ilfcut], power[if0:ilfcut])
        #plt.show()
        fmax=frequency[if0:ilfcut][np.argmax(power[if0:ilfcut])]/2.0
        fperiod=1.0/fmax
        
        print('Folding period',fperiod)
        cycles=len(data['time'].values)/48.0/(fperiod/2.0)
        print('Data has',cycles,'cycles')
        if cycles<minCycles:
            print('Skipping case with too few cycles')
            plt.clf()
            return 0
        fphases,ffluxes,ferrs=fold_lc(data['time'].values,data['flux'].values,data['err'].values,1/fmax)
        #f0=ffluxes.mean()
        ftimes=fphases*fperiod
        wts=1/ferrs**2
        ffmean=np.sum(ffluxes*wts)/np.sum(wts)
        print('ffmean',ffmean)
        logFmean=np.log10(ffmean+50)
        #logFmean=np.log10(np.mean(ffluxes)+50)
        #ffluxes=ffluxes/f0
        #sferrs=ferrs/f0
        ax[2].errorbar(fphases.tolist()+(fphases+1).tolist(),ffluxes.tolist()*2,yerr=ferrs.tolist()*2,marker='.',linestyle=' ',ms=10,alpha=0.6)
        if fperiod>50:#hack
            print('Folding period>50 days, skipping fit.')
        else:
 
            #fitting
            tgrid=np.arange(0,2,.01)*fperiod

            sp=copy.deepcopy(pyAvst.sp) 
            if not sp.pin('Pdays',fperiod):
                print('period out of range, expanding to',fperiod+0.1)
                sp.reset_range('Pdays',[0.2,fperiod+0.1])
                sp.pin('Pdays',fperiod)
            if(Mstar is not None):
                if massTol==0:
                    sp.pin('Mstar',Mstar)
                else:
                    sp.reset_range('Mstar',[Mstar*(1-massTol),Mstar*(1+massTol)])
            if lensMax>0:sp.reset_range('logMlens',[-1.0,np.log10(lensMax)])  
            if eMax>0:sp.reset_range('e',[0,eMax])
            
            sp2=copy.deepcopy(pyAvst.sp)
            if not sp2.pin('Pdays',fperiod/2):
                print('period out of range')
                sp2.reset_range('Pdays',[0.2,fperiod/2+0.1])
                sp2.pin('Pdays',fperiod/2)
            if(Mstar is not None):
                if massTol==0:
                    sp2.pin('Mstar',Mstar)
                else:
                    sp2.reset_range('Mstar',[Mstar*(1-massTol),Mstar*(1+massTol)])
            if lensMax>0:sp2.reset_range('logMlens',[-1.0,np.log10(lensMax)])  
            if eMax>0:sp2.reset_range('e',[0,eMax])

            if fit:
                print('sp:live',sp.live_names())
                print('mins/maxs',sp.live_ranges().T)
                print('pinvals',sp.pinvals)
                print('sp2:live',sp2.live_names())
                print('mins/maxs',sp2.live_ranges().T)
                print('pinvals',sp2.pinvals)

            f=lambda x:weighted_likelihood(ftimes,ffluxes,ferrs,x,sp,Rstar)
            f2=lambda x:weighted_likelihood(ftimes,ffluxes,ferrs,x,sp2,Rstar)

            lpars=sp.draw_live()
            pars=sp.get_pars(lpars)
            pars0b=pars[:]
            pars0b[2]=10000
            pars0b[7]=logFmean
            pars0b[8]=0
            llike0=pyAvst.likelihood(ftimes,ffluxes,ferrs,pars0b)
            model=pyAvst.lightcurve(tgrid,pars0b)
            SNR=np.sqrt(-llike0*2)
            print('Like0:',llike0)
            print('  SNR =',SNR)
            if(SNRonly):
                plt.clf()
                plt.close()
                return SNR
            ax[2].plot(tgrid/fperiod,model,label='mean')
            snrcut=5
            if(llike0>-snrcut**2/2):
                print('SNR<%.0f,skipping fit'%snrcut)
            elif fit==True:
                print('\nFitting with P=',fperiod)
                if(method=='loop'):
                    bestL,bestX=loop_search(f,sp,Rstar=Rstar)
                elif(method=='basin'):
                    bestL,bestX=global_search(f,sp,Temp=-llike0/2,Rstar=Rstar)
                    #bestL,bestX=loop_search(f,sp,Rstar)
                else:
                    print('method not recognized.')
                print('\nbest',bestX,bestL)
                print(len(bestX))
                print(sp.get_pars(bestX))
                pyAvst.test_roche_lobe(sp.get_pars(bestX),Rstar,True)
                if(bestL>llike0):#don't show horrible fits
                    model=pyAvst.lightcurve(tgrid,sp.get_pars(bestX),t0=ftimes[0])
                    plt.plot(tgrid/fperiod,model,label='fit')
                bestP=fperiod
                bestXp=sp.get_pars(bestX)

                if(True):
                    print('\nFitting with P=',fperiod/2)

                    if(method=='loop'):
                        bestL2,bestX2=loop_search(f2,sp2,Rstar=Rstar)
                    elif(method=='basin'):
                        bestL2,bestX2=global_search(f2,sp2,Rstar=Rstar)
                        #bestL2,bestX2=loop_search(f2,sp2,Rstar)
                    pyAvst.test_roche_lobe(sp2.get_pars(bestX2),Rstar,True)
                    print('\nbest2',bestX2,bestL2)
                    #if not sp2.pin('Fblend',0.3):print('logF out of range')
                    #if not sp2.pin('T0overP',0.53-.15):print('T0 out of range') 
                    #bestX2=[]
                    #bestX2p=sp2.pinvals
                    #bestL2=-f2([])
                    #print('bestL2hack=',bestL2)
                    #model=pyAvst.lightcurve(tgrid,bestX2p)
                    #plt.plot(tgrid/fperiod,model,label='bestL2hack')
                    if(bestL2>llike0):
                        model=pyAvst.lightcurve(tgrid,sp2.get_pars(bestX2),t0=ftimes[0])
                        plt.plot(tgrid/fperiod,model,label='fit, P/2')
                    if bestL2>bestL:
                        bestL=bestL2
                        bestX=bestX2
                        bestP=fperiod/2
                        bestXp=sp2.get_pars(bestX2)
                        
                print('Best fit results:')
                print('  period =',bestP)
                print('  pars =',bestXp)
                print('  SNR =',np.sqrt(-llike0*2))
                print('  chi2 =',-bestL)
                print('  fit percent = %5.2f'%((1-bestL/llike0)*100.0))
                      
        plt.legend()
        
        plt.show()

        #plt.errorbar(fphases,ffluxes,ferrs)
        #print('clean=',(id in lfids))
        #print(TICData['GAIA'])
        #print(len(TICData))
        if TICData is not None and doGAIA:
            #print('GAIA obs within 2px:')
            radec=' '.join([str(x) for x in TICData['ra','dec'][0]])
            try:
                GAIAData=Catalogs.query_object(radec,unit='deg',radius=0.011,catalog='Gaia')
                #print(GAIAData.columns)
                print('GAIA obs within 2px:')
                GAIAData['parallax_over_error'].name='plax_/_err'
                GAIAData['radial_velocity'].name='rad_vel'
                GAIAData['phot_g_mean_mag'].name='g_mag'
                GAIAData['delta_ra']=GAIAData['ra']-TICData[0]['ra']
                GAIAData['delta_dec']=GAIAData['dec']-TICData[0]['dec']
                GAIAData['delta_ra'].format=GAIAData['delta_dec'].format=GAIAData['g_mag'].format=GAIAData['parallax'].format=GAIAData['plax_/_err'].format=GAIAData['pmra'].format=GAIAData['pmdec'].format=GAIAData['rad_vel'].format='9.5f'
                GAIAData['designation','delta_ra','delta_dec','g_mag','parallax','plax_/_err','pmra','pmdec','rad_vel'].pprint(max_width=200)
            except:
                print("**GAIA Query Failed**")
    return SNR
                
def read_data_from_sector_files(id,basepath,tag='*n15*',gapsize=0):
    #print('id=',id)
    found_in_sectors=[]
    df=pd.DataFrame(columns=['sector','time','flux','err'])
    for isec in range(2,10): #Note: we skip sector 1 since it seems dubious in many cases
        #print('sector',isec)
        wildstring=basepath+'/sector'+str(isec)+'/lc_'+str(id)+tag+'.???'
        print('searching for ',wildstring)
        files=glob.glob(wildstring)
        if(len(files)==1):
            #print('processing '+files[0])
            #arr=np.loadtxt(files[0])
            secdf=pd.read_csv(files[0],header=None, comment='#', delim_whitespace=True,names=['time','flux','err'])
            secdf['sector']=isec
            if gapsize>0: #Note in days
                times=secdf['time'].values
                tlast=0
                drops=[]
                for t in times:
                    if t-tlast>gapsize:
                        #if the gap is bigger than gapsize then we expand the hole by
                        #gapsize on either side, also for the beginning and end
                        drops.append((tlast-gapsize,t+gapsize,))
                    tlast=t
                drops.append((tlast-gapsize,1e20,))
                #print('drops',drops)
                for tmin,tmax in drops:
                    times=times[np.where(np.logical_or(times < tmin , times > tmax ) )]
                #print('times',times)
                secdf=secdf[secdf['time'].isin(times)]
            #print('secdf=\n',secdf)
            df=df.append(secdf,ignore_index=True)
            found_in_sectors.append(isec)
        elif len(files)>1:
            print('More than one file matches! files=',files)
    print("Found in sectors",found_in_sectors)
    return df
