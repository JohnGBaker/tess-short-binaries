from astropy.stats import LombScargle
import pandas as pd
import numpy as np
#import matplotlib as mpl
#import matplotlib.pyplot as plt
import astroquery
from astroquery.mast import Catalogs,Observations
#import re

import sys

dirp='../../../TessSLB/src/LightCurveCode'
if dirp not in sys.path: sys.path.append(dirp)
dirp='../../../TessSLB/src/LightCurveCode/ptmcmc/cython'
if dirp not in sys.path: sys.path.append(dirp)
dirp='../../MCMC/ptmcmc/python'
if dirp not in sys.path: sys.path.append(dirp)
import ptmcmc
import ptmcmc_analysis
import pyAvst
#import BrowseSLBs
import copy
#import warnings
import scipy
import glob
import pickle
import re
#import scipy.linalg
#import scipy.optimize as opt
#warnings.simplefilter("ignore")
#importlib.reload(pyAvst)


useM=True

def fold_lc(times,fluxes,errs,Pfold,downfac=1.0):
    phases=(np.array(times)/Pfold)%1
    isort=np.argsort(phases)
    phases=phases[isort]
    fluxes=np.array(fluxes)[isort]
    errs=np.array(errs)[isort]
    nold=len(times)
    groupwidth=(times[-1]-times[0])*(1+0.1/nold)/nold/Pfold
    groupwidth*=downfac
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

def weighted_likelihood(ftimes,ffluxes,ferrs,x,sp,Rstar,constraint_weight=10000):
    pars=sp.get_pars(x);
    if sp.out_of_bounds(pars): #Hopefully doesn't happen?
        lmeans=np.mean(sp.live_ranges(),axis=1)
        parwt=np.sum((pars-sp.get_pars(lmeans))**2)
        #print(lmeans,parwt)
        return 2e18*(1+parwt*0)
    else:
        mlike=pyAvst.likelihood(ftimes,ffluxes,ferrs,pars,ulambda=0)
        if constraint_weight > 0:
            roche_frac=pyAvst.test_roche_lobe(pars,Rstar=Rstar)
            mlike-=constraint_weight*max([0,roche_frac-0.8])
        #print(x,mlike)
        #print(roche_frac,pars)
        return mlike

def weighted_likelihood_lferr0(ftimes,ffluxes,lferr0,x,sp,Rstar,constraint_weight=10000):
    pars=sp.get_pars(x);
    if sp.out_of_bounds(pars): #Hopefully doesn't happen?
        print('pars out of bounds:',pars)
        lmeans=np.mean(sp.live_ranges(),axis=1)
        parwt=np.sum((pars-sp.get_pars(lmeans))**2)
        #print(lmeans,parwt)
        return 2e18*(1+parwt*0)
    else:
        mlike=pyAvst.likelihood_log10ferr0(ftimes,ffluxes,lferr0,pars,ulambda=0)
        if constraint_weight > 0:
            roche_frac=pyAvst.test_roche_lobe(pars,Rstar=Rstar)
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


class SLB_likelihood(ptmcmc.likelihood):
    
    def __init__(self,id,data,period=None,lferr0=None,Mstar=None,Rstar=None,massTol=0,lensMax=0,eMax=None,maxperiod=14,fixperiod=None,dlogF=0.01,downfac=1.0,constraint_weight=10000,outname=""):
        self.Rstar=Rstar
        self.bestL=None
        ## Prepare data ##
        if True:  data=adjust_sectors(data)
        self.data=data
        self.constraint_weight=constraint_weight
        
        #dofold=(period is None)
        dofold=True
        if dofold:    
            ## Compute period and fold data ##
            if period is not None: fperiod=period
            else:
                print("Computing folding period")
                frequency, power = LombScargle(data['time'].values,data['flux'].values).autopower()
                ilfcut=int(len(power)/20)+1
                if0=0
                for i,f in enumerate(frequency):
                    if 1/f < maxperiod:
                        if0=i
                        break
                fmax=frequency[if0:ilfcut][np.argmax(power[if0:ilfcut])]
                fperiod=1.0/fmax
            doubler=1#set to 2 to fold on  double period
            if(fixperiod is not None):
                ffold=fixperiod*doubler
                fperiod=fixperiod
            else:ffold=fperiod*doubler
            self.fperiod=fperiod
            
            print('Folding period',ffold)
            cycles=len(data['time'].values)/48.0/(fperiod/doubler)
            print('Data has',cycles,'cycles')
            self.fphases,self.ffluxes,self.ferrs=fold_lc(data['time'].values,data['flux'].values,data['err'].values,ffold,downfac=downfac)
            self.ftimes=self.fphases*ffold
            array=np.vstack((self.ftimes,self.ffluxes,self.ferrs)).T
            pd.DataFrame(data=array,columns=['ftime','fflux','ferr']).to_csv(outname+'_folded.dat')
        else:  #no fold
            self.ftimes=data['time'].values
            self.ffluxes=data['flux'].values
            self.fperiod=period
        #wts=1/self.ferrs**2
        wts=1+0*self.ffluxes
        ffmean=np.sum(self.ffluxes*wts)/np.sum(wts)
        print('ffmean',ffmean)
        logFmean=np.log10(ffmean+50)
        print('logFmean',logFmean)
        print('ftimes',self.ftimes)
        print('ffluxes',self.ffluxes)

        ## Set up parameter space
        ## This piggybacks on the parameter space tool from  pyAvst
        
        sp=copy.deepcopy(pyAvst.sp)
        #Allow periods within a factor of just over 2% of specified
        sp.reset_range('Pdays',[self.fperiod/1.02,self.fperiod*1.02])
        sp.pin('Pdays',self.fperiod)
        sp.reset_range('log(F+50)',[logFmean-dlogF,logFmean+dlogF])

            
        if(Mstar is not None):
            if massTol==0:
                sp.pin('Mstar',Mstar)
            else:
                sp.reset_range('Mstar',[Mstar/(1+massTol),Mstar*(1+massTol)])
        if lensMax>0:sp.reset_range('logMlens',[-1.0,np.log10(lensMax)])  
        if eMax is not None:
            if eMax>0:sp.reset_range('e',[0,eMax])
            else:sp.pin('e',0)
        self.sp=sp
        
        ###Compute SNR
        pars0=[-10,1,10000,0,0,0,0,logFmean,0]
        #logMlens, Mstar, Pdays, e, sini, omgf, T0overP,logFp50,Fblend=pars0
        #llike0=pyAvst.likelihood(self.ftimes,self.ffluxes,self.ferrs,pars0,ulambda=0)
        if lferr0 is None:
            llike0=pyAvst.likelihood(self.ftimes,self.ffluxes,self.ferrs,pars0,ulambda=0)
        else:
            llike0=pyAvst.likelihood_log10ferr0(self.ftimes,self.ffluxes,lferr0,pars0,ulambda=0)
        self.lferr0=lferr0
        SNR=np.sqrt(-llike0*2)

        print('sp:live',sp.live_names())
        print('mins/maxs',sp.live_ranges().T)
        print('pinvals',sp.pinvals)

        #Set up  stateSpace
        names=sp.live_names()
        ranges=sp.live_ranges()
        npar=len(names)
        space=ptmcmc.stateSpace(dim=npar);
        space.set_names(names);
        wraps=['omgf','T0overP']
        centers=[0]*npar
        scales=[1]*npar
        types=['uni']*npar
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

        print("SLB_likelihood::setup: space="+space.show())
        self.basic_setup(space, types, centers, scales);

    def evaluate_log(self,s):
        params=s.get_params()
        #print(params)
        if self.lferr0 is None:
            result=weighted_likelihood(self.ftimes,self.ffluxes,self.ferrs,params,self.sp,self.Rstar,self.constraint_weight)
        else:
            result=weighted_likelihood_lferr0(self.ftimes,self.ffluxes,self.lferr0,params,self.sp,self.Rstar,self.constraint_weight)
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
        models=[pyAvst.lightcurve(self.ftimes,self.sp.get_pars(pars)) for pars in parslist]
        return models
        
count=0

def read_data_from_sector_files(id,basepath,edgeskip=0.5,allowsecs=None):
    if allowsecs is None:allowsecs=range(1,20)
    #print('id=',id)
    datafiles=glob.glob(basepath+'/*/*/tesslc_'+str(id)+'.pkl')
    found_in_sectors=[]
    df=pd.DataFrame(columns=['sector','time','flux','err'])
    df=df.sort_values('time')
    for path in datafiles:
        data=pickle.load(open(path,'rb'))
        sector=int(re.findall(r'sector_(\d*)',path)[0])
        found_in_sectors+=[sector]
        if not sector in allowsecs:continue
        flux = data[6] 
        time = data[4]
        fluxerr = data[8]
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
        #print(df)
    print("Found in sectors",found_in_sectors)
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
    opt.add("datadir","directory where processed sector data files are located",".")
    opt.add("seed","Pseudo random number grenerator seed in [0,1). (Default=-1, use clock to seed.)","-1")
    ##opt.add("precision","Set output precision digits. (Default 13).","13")
    opt.add("outname","Base name for output files (Default 'mcmc_output').","mcmc_output")
    opt.add("period","Set fixed period for folding and model. (Default None)","None")
    opt.add("eMax","Set max value for eccentricity. (Default 0.2)","0.2")
    opt.add("dlogF","Prior halfwidth for log10(F). (Default 0.01)","0.01")
    opt.add("Mstar","Override TIC star mass. (Default None)","None")
    opt.add("massTol","Uniform confidence width factor for TIC mass. (Default 0.2)","0.2")
    opt.add("plotSamples","File with samples to plot, (eg chain output)","")
    opt.add("nPlot","If plotting samples, how many to sample curves to include","20")
    opt.add("downfac","Extra downsampling factor in lightcurve folding.","1")
    opt.add("Roche_wt","Weight factor for Roche-limit constraint (def 10000).","10000")
    opt.add("secs","Only use these sectors (comma-separated)","")
    opt.add("l10ferr","log10 of fractional flux err. (def =-3.25)","-3.25")
    #int Nlead_args=1;
    
    #//Create the sampler
    #ptmcmc_sampler mcmc;
    s0=ptmcmc.sampler(opt)

    opt.parse(argv)
    
    #Get TIC catalog info:
    id=int(opt.value('id'))
    datadir=opt.value('datadir')
    outname=opt.value('outname')
    massTol=float(opt.value('massTol'))
    Roche_wt=float(opt.value('Roche_wt'))
    

    try:
        TICData = Catalogs.query_object('TIC '+str(id),radius=0.0011,catalog='TIC')#0.011 deg is 2 px
        print(TICData['ID','Tmag','Vmag','ra','dec','d','objType','lumclass','Teff','mass','rad'][0])
        #print(TICData.columns)
    except: 
        print("**TIC Query Failed**")
        print("id=",id)
        TICData=None  
    
    if TICData is not None:
        print('Vmag',TICData['Vmag'][0], 'Teff',TICData['Teff'][0])

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
            print('Mstar(TIC)=',Mstar)
        if opt.value('Mstar')!='None':Mstar=float(opt.value('Mstar'))
        print('Mstar=',Mstar)

    allowsecs=None
    if opt.value('secs')!="":
        allowsecs=opt.value('secs').split(',')
        allowsecs=[int(sec) for sec in allowsecs]
                         
    #Prepare the data:
    dfg=read_data_from_sector_files(id,datadir,edgeskip=0.5,allowsecs=allowsecs)
    
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
    dlogF=float(opt.value('dlogF'))
    downfac=float(opt.value('downfac'))

    lferr0=float(opt.value('l10ferr'))
    
    like=SLB_likelihood(id,dfg,period,lferr0,Mstar,Rstar,massTol=massTol,lensMax=0,eMax=eMax,maxperiod=20,fixperiod=fixperiod,dlogF=dlogF,downfac=downfac,constraint_weight=Roche_wt,outname=outname)  

    if(opt.value('plotSamples')!="" or int(opt.value('nPlot'))==0):
        #Plot samples instead of running chains
        ncurves=int(opt.value('nPlot'))
        t=like.ftimes
        ts=np.linspace(t[0],t[-1],300)
        data=like.ffluxes
        if ncurves>0:
            chain=ptmcmc_analysis.chainData(opt.value('plotSamples'))
            samples=chain.get_samples(ncurves)
            print("samples:")
            for sample in samples:print(sample)
            models=[pyAvst.lightcurve(ts,like.sp.get_pars(pars)) for pars in samples]
            roches=[pyAvst.test_roche_lobe(like.sp.get_pars(pars),Rstar=like.Rstar) for pars in samples]
            print('roche fracs:',roches)
        else: models =[]
        import matplotlib.pyplot as plt
        plt.errorbar(t,data,yerr=like.ferrs,ls='None')
        for model in models:
            plt.plot(ts,model,'r',alpha=0.2)
        plt.plot(like.data['time']%(like.fperiod),like.data['flux'],'k.',ls='None',markersize=0.5)
        plt.show()
        return
        
        
    seed=float(opt.value('seed'))
    if seed<0:seed=random.random();
    #istringstream(opt.value("precision"))>>output_precision;
    #istringstream(opt.value("outname"))>>outname;
    
    #//report
    #cout.precision(output_precision);
    print("\noutname = '"+outname+"'")
    #cout<<"seed="<<seed<<endl; 
    #cout<<"Running on "<<omp_get_max_threads()<<" thread"<<(omp_get_max_threads()>1?"s":"")<<"."<<endl;
    
    #//Should probably move this to ptmcmc/bayesian
    ptmcmc.resetRNGseed(seed);
    
    #globalRNG.reset(ProbabilityDist::getPRNG());//just for safety to keep us from deleting main RNG in debugging.
          
    #//Get the space/prior for use here
    #stateSpace space;
    #shared_ptr<const sampleable_probability_function> prior;  
    space=like.getObjectStateSpace();
    print("like.nativeSpace=\n"+space.show())
    #prior=like->getObjectPrior();
    #cout<<"Prior is:\n"<<prior->show()<<endl;
    #valarray<double> scales;prior->getScales(scales);
    
    #//Read Params
    Npar=space.size();
    print("Npar=",Npar)
    
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
