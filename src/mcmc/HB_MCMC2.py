#Written by John G Baker @ NASA 2020-2021
#The original version of this code is at
#https://gitlab.fit.nasa.gov/jgbaker/tessslb.git
#
# To use this code you will need the HB lightcurve code
# at github/sidruns30/HB_MCMC and the MCMC code at
# github/johngbaker/ptmcmc
#
from astropy.stats import LombScargle
import pandas as pd
import numpy as np

#import matplotlib as mpl
#import matplotlib.pyplot as plt
import astroquery
from astroquery.mast import Catalogs,Observations
#import re

import sys

#dirp='../../../TessSLB/src/LightCurveCode'
#if dirp not in sys.path: sys.path.append(dirp)
#dirp='../../../TessSLB/src/LightCurveCode/ptmcmc/cython'
#if dirp not in sys.path: sys.path.append(dirp)
dirp='../../MCMC/ptmcmc/python'
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

def fold_lc(times,fluxes,errs,Pfold,downfac=1.0,rep=False):
    phases=(np.array(times)/Pfold)%1
    isort=np.argsort(phases)
    phases=phases[isort]
    fluxes=np.array(fluxes)[isort]
    errs=np.array(errs)[isort]
    nold=len(times)
    print('nold,Pfold,downfac:',nold,Pfold,downfac,times[0],"< t <",times[-1])
    
    groupwidth=(times[-1]-times[0])*(1+0.1/nold)/nold/Pfold #frac period bin size
    groupwidth*=downfac
    #print('mean errs=',errs.mean())
    print('groupwidth=',groupwidth, 'mean group size=',groupwidth*nold)
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

def weighted_likelihood(ftimes,ffluxes,ferrs,x,sp,constraint_weight=10000):
    pars=sp.get_pars(x);
    if sp.out_of_bounds(pars): #Hopefully doesn't happen?
        lmeans=np.mean(sp.live_ranges(),axis=1)
        parwt=np.sum((pars-sp.get_pars(lmeans))**2)
        #print(lmeans,parwt)
        return 2e18*(1+parwt*0)
    else:
        mlike=pyHB.likelihood(ftimes,ffluxes,ferrs,pars)
        if constraint_weight > 0:
            roche_frac=pyHB.test_roche_lobe(pars)
            mlike-=constraint_weight*max([0,roche_frac-0.8])
        #print(x,mlike)
        #print(roche_frac,pars)
        return mlike

def weighted_likelihood_lferr0(ftimes,ffluxes,lferr0,x,sp,constraint_weight=10000):
    
    pars=sp.get_pars(x);
    if sp.out_of_bounds(pars): #Hopefully doesn't happen?
        print('pars out of bounds:',pars)
        lmeans=np.mean(sp.live_ranges(),axis=1)
        parwt=np.sum((pars-sp.get_pars(lmeans))**2)
        #print(lmeans,parwt)
        return 2e18*(1+parwt*0)
    else:
        mlike=pyHB.likelihood_log10ferr0(ftimes,ffluxes,lferr0,pars)
        if constraint_weight > 0:
            roche_frac=pyHB.test_roche_lobe(pars)
            mlike-=constraint_weight*max([0,roche_frac-0.8])
        #print(x,mlike)
        #print(roche_frac,pars)
        return mlike

def adjust_sectors(data):
    sector_tag='sector'
    sectors=data[sector_tag].unique()
    print('sectors',sectors)
    if(len(sectors)>1):
        medians=np.array([np.median(data.flux[data[sector_tag]==sec]) for sec in sectors])
        offsets=medians-medians.mean()
        #print('offsets',offsets)
        for i in range(len(sectors)):
            data.loc[data[sector_tag]==sectors[i],'flux']/=1+offsets[i]/medians.mean()
        print('Adjusted sector levels:',offsets)
        print('Adjusted sector factors:',1+offsets/medians.mean())
    return data


class HB_likelihood(ptmcmc.likelihood):
    
    def __init__(self,id,data,period=None,lferr0=None,Mstar=None,massTol=0,lensMax=0,eMax=None,maxperiod=14,fixperiod=None,downfac=1.0,constraint_weight=10000,outname="",rep=False,forceM1gtM2=False,rescalefac=1.0,allow_blend=False):
        self.bestL=None
        self.forceM1gtM2=forceM1gtM2
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
            if period is not None: fperiod=period
            else:
                print("Computing folding period")
                frequency, power = LombScargle(data['time'].values,data['flux'].values).autopower()
                ilfcut=int(len(power)/40)+1
                if0=0
                for i,f in enumerate(frequency):
                    if 1/f < maxperiod:
                        if0=i
                        break
                fmax=frequency[if0:ilfcut][np.argmax(power[if0:ilfcut])]
                fperiod=1.0/fmax
                if rep:
                    import matplotlib.pyplot as plt
                    plt.plot(frequency,power)
                    plt.plot(frequency[if0:ilfcut],power[if0:ilfcut])
                    plt.show()
                sys.exit
                    
            doubler=1#set to 2 to fold on  double period
            if(fixperiod is not None):
                ffold=fixperiod*doubler
                fperiod=fixperiod
            else:ffold=fperiod*doubler
            self.fperiod=fperiod
            
            cycles=len(data['time'].values)/48.0/(fperiod/doubler)
            if self.rep:
                print('Folding period',ffold)
                print('Data has',cycles,'cycles')
            self.fphases,self.ffluxes,self.ferrs=fold_lc(data['time'].values,data['flux'].values,data['err'].values,ffold,downfac=downfac,rep=rep)
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
            print('logFmean',logFmean)
            print('ftimes',self.ftimes)
            print('ffluxes',self.ffluxes)

        ## Set up parameter space
        ## This piggybacks on the parameter space tool from  pyHB
        
        sp=copy.deepcopy(pyHB.sp)
        #Allow periods within a factor of just over 2% of specified
        sp.reset_range('logP',[np.log10(self.fperiod/1.02),np.log10(self.fperiod*1.02)])
        sp.pin('logP',np.log10(self.fperiod))
        #sp.reset_range('log(F+50)',[logFmean-dlogF,logFmean+dlogF])
        if not allow_blend: sp.pin('log_blendFlux',-3)
            
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
        
        ###Compute SNR
        #pars0=[-10,1,10000,0,0,0,0,logFmean,0]
        #logMlens, Mstar, Pdays, e, sini, omgf, T0overP,logFp50,Fblend=pars0

        #if lferr0 is None:
        #    llike0=pyHB.likelihood(self.ftimes,self.ffluxes,self.ferrs,pars0)
        #else:
        #    llike0=pyHB.likelihood_log10ferr0(self.ftimes,self.ffluxes,lferr0,pars0)
        self.lferr0=lferr0
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
        types=['uni']*npar
        types[names.index('inc')]='pol'
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
        

        #some rescaling for better Gaussian proposals
        rescales=[rescalefac]*npar
        
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
            if self.lferr0 is None:
                result=weighted_likelihood(self.ftimes,self.ffluxes,self.ferrs,params,self.sp,self.constraint_weight)
            else:
                result=weighted_likelihood_lferr0(self.ftimes,self.ffluxes,self.lferr0,params,self.sp,self.constraint_weight)
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
        models=[pyHB.lightcurve(self.ftimes,self.sp.get_pars(pars)) for pars in parslist]
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

def read_data_from_file(id,path,edgeskip=0.5,trueerr=1.0,tmin=None,tmax=None):
    print('Reading data from file:',path)
    data=np.genfromtxt(path,skip_header=1)
    #assumed format t flux other_flux flag
    data=data[data[:,3]==0] #clean out the flagged rows
    if tmin is not None: data=data[data[:,0]>=tmin]
    if tmax is not None: data=data[data[:,0]<=tmax]
    flux = data[:,1] 
    time = data[:,0]
    fluxerr = np.sqrt(flux)*trueerr
    dt=time[1]-time[0]
    iedgeskip=int(edgeskip/dt)
    print('iedgeskip',iedgeskip)
    if(iedgeskip>0):#process edge skips
        keeps=np.array([True]*len(time))
        keeps[0:iedgeskip]=False
        keeps[-iedgeskip:]=False
        for i in range(1,len(time)):
            if keeps[i] and time[i]-time[i-1]>0.5: #gap >1/2 day
                print('cut detected at t =',time[i])
                print(time[i-1],time[i],time[i]-time[i-1])
                keeps[i-iedgeskip:i]=False
                print('skipping from',time[i-iedgeskip],'to',time[i+iedgeskip])
                keeps[i:i+iedgeskip]=False
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
    ##opt.add("nchains","Number of consequtive chain runs. Default 1","1")
    opt.add("id","TIC_ID","")
    opt.add("noTIC","Set to 1 to skip any online query about TIC id.","0")
    opt.add("datadir","directory where processed sector data files are located",".")
    opt.add("datafile","Explicitly indicate the data file.","")
    opt.add("seed","Pseudo random number grenerator seed in [0,1). (Default=-1, use clock to seed.)","-1")
    ##opt.add("precision","Set output precision digits. (Default 13).","13")
    opt.add("outname","Base name for output files (Default 'mcmc_output').","mcmc_output")
    opt.add("period","Set fixed period for folding and model. (Default None)","None")
    opt.add("eMax","Set max value for eccentricity. (Default 0.95)","0.95")
    opt.add("Mstar","Override TIC star mass. (Default None)","None")
    opt.add("massTol","Uniform confidence width factor for TIC mass. (Default 0.2)","0.2")
    opt.add("plotSamples","File with samples to plot, (eg chain output)","")
    opt.add("nPlot","If plotting samples, how many to sample curves to include","20")
    opt.add("downfac","Extra downsampling factor in lightcurve folding.","1")
    opt.add("Roche_wt","Weight factor for Roche-limit constraint (def 10000).","10000")
    opt.add("secs","Only use these sectors (comma-separated)","")
    opt.add("trueerr","scalefactor of the error following JS's code. (def=1)","1")
    opt.add("M1gtM2","Set to 1 to force M1>M2. (def=0)","0")
    opt.add("tlimits","Set tmin,tmax time limits, outside which to ignore data. (def=none)","")    #int Nlead_args=1;
    opt.add('rescalefac','Rescale factor for gaussian proposals. Default=1','1')
    opt.add('blend','Set to 1 to vary the blending flux','0')
    #//Create the sampler
    #ptmcmc_sampler mcmc;
    s0=ptmcmc.sampler(opt)
    rep=s0.reporting()
    opt.parse(argv)
    
    #Get TIC catalog info:
    id=int(opt.value('id'))
    datadir=opt.value('datadir')
    outname=opt.value('outname')
    massTol=float(opt.value('massTol'))
    Roche_wt=float(opt.value('Roche_wt'))
    
    
    if opt.value('noTIC')!='0':
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
        Mstar=None
        if not np.isnan(float(TICData['mass'][0])):
            Mstar=TICData['mass'][0]
            #print('float(Mstar)',float(Mstar))
            if rep:print('Mstar(TIC)=',Mstar)
        if opt.value('Mstar')!='None':Mstar=float(opt.value('Mstar'))
        if rep:print('Mstar=',Mstar)

    allowsecs=None
    if opt.value('secs')!="":
        allowsecs=opt.value('secs').split(',')
        allowsecs=[int(sec) for sec in allowsecs]
    tmin=None
    tmax=None
    if opt.value('tlimits')!="":
        tlims=opt.value('tlimits').split(',')
        if len(tlims)<2:tlims.append('')
        if tlims[0].isnumeric():tmin=float(tlims[0])
        if tlims[1].isnumeric():tmax=float(tlims[1])
        print('Constraining',tmin,'< t <',tmax)
        
    #Prepare the data:
    trueerr=float(opt.value('trueerr'))
    if opt.value('datafile')=="":
        dfg=read_data_from_sector_files(id,datadir,edgeskip=0.5,allowsecs=allowsecs,trueerr=trueerr,tmin=tmin,tmax=tmax)
    else:
        filepath=datadir+'/'+opt.value('datafile')
        dfg=read_data_from_file(id,filepath,trueerr=trueerr,tmin=tmin,tmax=tmax)
    if rep:
        print('Trimmed data length is',len(dfg))
        dfg[['time','flux','err']].to_csv(outname+'_trimmed.dat')

    #//Create the likelihood
    
    fixperiod=None
    if opt.value('period')=="None":
        period=None
    else:
        period=float(opt.value('period'))
        if period<0:
            period=-period
            fixperiod=period
    eMax=float(opt.value('eMax'))
    #dlogF=float(opt.value('dlogF'))
    downfac=float(opt.value('downfac'))
    forceM1gtM2=(int(opt.value('M1gtM2'))==1)
    rescalefac=float(opt.value('rescalefac'))
    #lferr0=float(opt.value('l10ferr'))
    lferr0=None
    blend=int(opt.value('blend'))==1
    
    like=HB_likelihood(id,dfg,period,lferr0,Mstar,massTol=massTol,eMax=eMax,maxperiod=20,fixperiod=fixperiod,downfac=downfac,constraint_weight=Roche_wt,outname=outname,rep=rep,forceM1gtM2=forceM1gtM2,rescalefac=rescalefac,allow_blend=blend)

    if(opt.value('plotSamples')!="" or int(opt.value('nPlot'))==0):
        #Plot samples instead of running chains
        ncurves=int(opt.value('nPlot'))
        t=like.ftimes
        ts=np.linspace(t[0],t[-1],300)
        data=like.ffluxes
        if ncurves>0:
            sampfiles=opt.value('plotSamples')
            if sampfiles.startswith('[') and sampfiles.endswith(']'):
                if ',' in sampfiles:
                    sampfiles=sampfiles[1:-1].split(',')
                else:
                    sampfiles=sampfiles[1:-1].split()
            else:
                sampfiles=[sampfiles]
            modelsets=[]
            for sfile in sampfiles:
                print('Processing',sfile)
                chain=ptmcmc_analysis.chainData(sfile)
                n=min(1000000,chain.getSteps()/chain.dSdN)
                samples=chain.get_samples(ncurves,n)
                print("samples:")
                for sample in samples:print(sample)
                models=[pyHB.lightcurve(ts,like.sp.get_pars(pars)) for pars in samples]
                modelsets.append(models)
                roches=[pyHB.test_roche_lobe(like.sp.get_pars(pars),verbose=True) for pars in samples]
                print('roche fracs:',roches)
        else: modelsets =[]
        import matplotlib.pyplot as plt
        plt.errorbar(t,data,yerr=like.ferrs,ls='None',label='data')
        colors=['r','b','g','y','m','c','k']
        for i in range(len(modelsets)):
            label=sampfiles[i]
            col=colors[i]
            for model in modelsets[i]:
                plt.plot(ts,model,col,alpha=0.2,label=label)
                label=None
            rawftimes=like.data['time']%(like.fperiod)+int(like.data['time'][0]/like.fperiod)*like.fperiod
            #-0*like.data['time'][0]%(like.fperiod)+like.ftimes[0]
        
        plt.plot(rawftimes,like.data['flux'],'k.',ls='None',markersize=0.5,label='raw data')
        plt.legend()
        plt.show()
        return
        
        
    seed=float(opt.value('seed'))
    if seed<0:seed=np.random.random();
    #istringstream(opt.value("precision"))>>output_precision;
    #istringstream(opt.value("outname"))>>outname;
    
    #//report
    #cout.precision(output_precision);
    if rep:print("\noutname = '"+outname+"'")
    #cout<<"seed="<<seed<<endl; 
    #cout<<"Running on "<<omp_get_max_threads()<<" thread"<<(omp_get_max_threads()>1?"s":"")<<"."<<endl;
    
    #//Should probably move this to ptmcmc/bayesian
    ptmcmc.resetRNGseed(seed);
    
    #globalRNG.reset(ProbabilityDist::getPRNG());//just for safety to keep us from deleting main RNG in debugging.
          
    #//Get the space/prior for use here
    #stateSpace space;
    #shared_ptr<const sampleable_probability_function> prior;  
    space=like.getObjectStateSpace();
    if rep:print("like.nativeSpace=\n"+space.show())
    #prior=like->getObjectPrior();
    #cout<<"Prior is:\n"<<prior->show()<<endl;
    #valarray<double> scales;prior->getScales(scales);
    
    #//Read Params
    Npar=space.size();
    if rep:print("Npar=",Npar)
    
    #//Bayesian sampling [assuming mcmc]:
    #//Set the proposal distribution 
    #int Ninit;
    #proposal_distribution *prop=ptmcmc_sampler::new_proposal_distribution(Npar,Ninit,opt,prior.get(),&scales);
    #cout<<"Proposal distribution is:\n"<<prop->show()<<endl;
    #//set up the mcmc sampler (assuming mcmc)
    #//mcmc.setup(Ninit,*like,*prior,*prop,output_precision);
    #mcmc.setup(*like,*prior,output_precision);
    #mcmc.select_proposal();
    s0.setup(like)

    #//Testing (will break testsuite)
    #s=like.draw_from_prior();
    #print("test state:",s.get_string())
    #print("logL=",like.evaluate_log(s))
  
    
    #//Prepare for chain output
    #ss<<outname;
    #string base=ss.str();
    
    #//Loop over Nchains
    #for(int ic=0;ic<Nchain;ic++){
    s=s0.clone();
    s.initialize();
    print('initialization done')
    s.run(outname,0);
    #  //s->analyze(base,ic,Nsigma,Nbest,*like);
    #del s;
    #}
    
    #//Dump summary info
    #cout<<"best_post "<<like->bestPost()<<", state="<<like->bestState().get_string()<<endl;
    #//delete data;
    #//delete signal;
    #delete like;
    #}

if __name__ == "__main__":
    import sys
    argv=sys.argv[:]
    del argv[0]
    main(argv)
