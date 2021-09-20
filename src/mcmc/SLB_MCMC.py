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
import ptmcmc
import pyAvst
import BrowseSLBs
import copy
#import warnings
import scipy
#import scipy.linalg
#import scipy.optimize as opt
#warnings.simplefilter("ignore")
#importlib.reload(pyAvst)


useM=True

def weighted_likelihood(ftimes,ffluxes,ferrs,x,sp,Rstar):
    pars=sp.get_pars(x);
    if sp.out_of_bounds(pars): #Hopefully doesn't happen?
        lmeans=np.mean(sp.live_ranges(),axis=1)
        parwt=np.sum((pars-sp.get_pars(lmeans))**2)
        #print(lmeans,parwt)
        return 2e18*(1+parwt*0)
    else:
        roche_frac=pyAvst.test_roche_lobe(pars,Rstar=Rstar)
        mlike=-pyAvst.likelihood(ftimes,ffluxes,ferrs,pars,ulambda=0)+10000*max([0,roche_frac-0.8])
        #print(x,mlike)
        #print(roche_frac,pars)
        return -mlike

class SLB_likelihood(ptmcmc.likelihood):
    
    def __init__(self,id,data,Mstar=None,Rstar=None,massTol=0,lensMax=0,eMax=None,maxperiod=14,fixperiod=None,outname=""):
        self.Rstar=Rstar
        self.bestL=None
        ## Prepare data ##
        sector_tag='sector'
        sectors=data[sector_tag].unique()
        if(len(sectors)>1):
            medians=np.array([np.median(data.flux[data[sector_tag]==sec]) for sec in sectors])
            offsets=medians-medians.mean()
            #print('offsets',offsets)
            for i in range(len(sectors)):
                data.loc[data[sector_tag]==sectors[i],'flux']-=offsets[i]
            print('Adjusted sector levels:',offsets)

        ## Compute period and fold data ##
        
        frequency, power = LombScargle(data['time'].values,data['flux'].values).autopower()
        ilfcut=int(len(power)/20)+1
        if0=0
        for i,f in enumerate(frequency):
            if 1/f < maxperiod:
                if0=i
                break
        fmax=frequency[if0:ilfcut][np.argmax(power[if0:ilfcut])]
        fperiod=1.0/fmax
        if(fixperiod is not None):
            ffold=fixperiod*2
            fperiod=fixperiod
        else:ffold=fperiod*2

        
        print('Folding period',ffold)
        cycles=len(data['time'].values)/48.0/(fperiod/2.0)
        print('Data has',cycles,'cycles')
        self.fphases,self.ffluxes,self.ferrs=BrowseSLBs.fold_lc(data['time'].values,data['flux'].values,data['err'].values,ffold)
        self.ftimes=self.fphases*ffold
        array=np.vstack((self.ftimes,self.ffluxes,self.ferrs)).T
        pd.DataFrame(data=array,columns=['ftime','fflux','ferr']).to_csv(outname+'_folded.dat')
        wts=1/self.ferrs**2
        ffmean=np.sum(self.ffluxes*wts)/np.sum(wts)
        print('ffmean',ffmean)
        logFmean=np.log10(ffmean+50)

        ## Set up parameter space
        ## This piggybacks on the parameter space tool from  pyAvst
        
        sp=copy.deepcopy(pyAvst.sp)
        #if not sp.pin('Pdays',fperiod):
        #    print('period out of range, expanding to',fperiod+0.1)
        #    sp.reset_range('Pdays',[0.2,fperiod+0.1])
        #    sp.pin('Pdays',fperiod)
        #Folding period was twice peak max of lomb-scargle
        #but we allow periods within a factor of just over 2 of that peak
        sp.reset_range('Pdays',[fperiod/2.1,fperiod*2.1])
        sp.pin('Pdays',fperiod)
            
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
        llike0=pyAvst.likelihood(self.ftimes,self.ffluxes,self.ferrs,pars0,ulambda=0)
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
        result=weighted_likelihood(self.ftimes,self.ffluxes,self.ferrs,params,self.sp,self.Rstar)
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

count=0

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
    opt.add("Mstar","Override TIC star mass. (Default None)","None")
    opt.add("massTol","Uniform confidence width factor for TIC mass. (Default 0.2)","0.2")
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
    

    try: 
        TICData = Catalogs.query_object(str(id),radius=0.011,catalog='TIC')#0.011 deg is 2 px
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
    
    #Prepare the data:
    dfg=BrowseSLBs.read_data_from_sector_files(id,datadir,tag='*n15*',gapsize=0.75)
    
    #//Create the likelihood
    if opt.value('period')=="None":fixperiod=None
    else: fixperiod=float(opt.value('period'))
    eMax=float(opt.value('eMax'))
    
    like=SLB_likelihood(id,dfg,Mstar,Rstar,massTol=massTol,lensMax=0,eMax=eMax,maxperiod=13,fixperiod=fixperiod,outname=outname)  

    print('calling opt.parse')
    #bool parseBAD=opt.parse(argc,argv);
    #if(parseBAD) {
    #  cout << "Usage:\n mcmc [-options=vals] " << endl;
    #  cout <<opt.print_usage()<<endl;
    #  return 1;
    #}
    #cout<<"flags=\n"<<opt.report()<<endl;
    
    #//Setup likelihood
    #//data->setup();  
    #//signal->setup();  
    #like->setup();
    
    #double seed;
    #int Nchain,output_precision;
    #int Nsigma=1;
    #int Nbest=10;
    #string outname;
    #ostringstream ss("");
    #istringstream(opt.value("nchains"))>>Nchain;
    #istringstream(opt.value("seed"))>>seed;
    #//if seed<0 set seed from clock
    #if(seed<0)seed=fmod(time(NULL)/3.0e7,1);
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
