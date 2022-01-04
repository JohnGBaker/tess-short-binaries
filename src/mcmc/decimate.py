#Applies an algorithm to significantly decimate dense microlens data (eg from wfirst data challenge) without losing information.
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt

#The idea here is that we successively pass over the data, decimating progressive factors of two
#In each pass, we perform LLSF over surrounding N points, then compare likelihood of that fit
#both using the full data and using a factor of two decimation for the target points.
#If the difference is smaller than some tolerance (weighted by the original data rms err)
#then we keep the decimated version

# data are in a np array of the form [t,x,sigma^2,tmin,tmax]
# line info is in the form [t0, x0, slope]

def chi2(data,line):
    res=data[:,1]-line[1]-(data[:,0]-line[0])*line[2]
    return np.sum(res*res/data[:,2])

def DoLLSF(data,tref=None):
    sig2=1/np.sum(data[:,2]**(-1))
    t0=sig2*np.sum(data[:,0]/data[:,2])
    x0=sig2*np.sum(data[:,1]/data[:,2])
    t2sum=np.sum((data[:,0]-t0)**2/data[:,2])
    xtsum=np.sum((data[:,0]-t0)*(data[:,1]-x0)/data[:,2])
    slope=xtsum/t2sum;
    #print("\n1/sig2=",1/sig2,"t2sum=",t2sum,"xtsum=",xtsum)
    #print("t0,x0,slope",t0,x0,slope)
    #print(" data=",data)
    if(tref is None):tref=t0
    return np.array([tref,x0+(tref-t0)*slope,slope])

def subData(data,line,dchitol):
    #We will replace the data with a single point, requiring that
    # 1. llsf fit for this data + other data is unchanged
    #    -require slope and x0 variations of delta chi2 vanish
    # 2. the derivative of chi^2 wrt llsf intercept at mean time is preserved
    #
    # deltachi = sum[ (xi -x0 -(ti-t0)*s)^2 / sig2i ] - (xnew -x0 -(tnew-t0)*s)^2 / sig2new  
    #
    # d^2deltachi/dx0^2 = 0 -> 1/sig2new = sum(1/sig2i)
    # and
    # d deltachi/dx0 = 0    -> xnew -x0 -s*(tnew-t0) = sig2new * sum((xi-x0-s*(ti-t0))/sig2i) 
    #                                                = xd-x0
    # where xd=sig2new*sum(xi/sig2i), and we write the line setting t0=t0d=sig2new*sum(ti/sig2i)
    # and
    # d deltachi/ds = 0 = -sum((ti-t0)*(xi-x0-s*(ti-t0))/sig2i) + (tnew-t0)*(xnew-x0-s*(tnew-t0))/sig2new
    #                   = -sum((ti-t0)*ri/sig2i) + (tnew-t0)*(xd-x0)/sig2new
    # where ri = xi-x0-s*(ti-t0)
    #
    # For the last equation, if xd!=x0, we can set tnew to solve, but we constrain tnew to be within the
    # time limits of the data.
    # We also constrain the size of the resulting deltachi to be below some limit, after solving as above
    global nsub, nsubtfail,nsubchi2fail
    nsub+=1
    sig2new=1/np.sum(data[:,2]**(-1))
    t0d=sig2new*np.sum(data[:,0]/data[:,2])
    xd=sig2new*np.sum(data[:,1]/data[:,2])
    slope=line[2]
    x0=(t0d-line[0])*slope+line[1]
    #print("line0=",t0d,x0,slope)
    trel=data[:,0]-t0d;
    res=(data[:,1]-x0-trel*slope)
    #compute new t point to ensure that slope matches line
    trsum=np.sum(trel*res/data[:,2])
    #xsum=np.sum((data[:,1]-x0)/data[:,2])
    xsum=(xd-x0)/sig2new
    if(xsum==0):
        if(trsum==0):toff=0
        else: return data
    else: toff=trsum/xsum
    dataTmax=data[-1,4]
    dataTmin=data[0,3]
    if(dataTmax-t0d <= toff ):
        #print("fail tmax")
        nsubtfail+=1
        return data
    if(dataTmin-t0d >= toff ):
        #print("fail tmin")
        nsubtfail+=1
        return data
    tnew=t0d+toff
    #compute new xval
    xnew=xd+slope*(tnew-t0d)
    #print("xd,tnew,xnew",xd,tnew,xnew)
    dchi=(np.sum(res*res/data[:,2])-(xd-x0)**2/sig2new)
    if(dchi>dchitol):
        #print("fail dchi=",dchi,">",dchitol)
        nsubchi2fail+=1
        return data
    return np.array([[tnew,xnew,sig2new,dataTmin,dataTmax]])

def reduceDataChunk(segment,target,tol):
    line=DoLLSF(segment)
    n=len(segment)
    if( n - len(target) < 2):
        #in this case there is no real solution to the formal problem I pose
        #if there is just 1 remaining point, then a solution could be found, but it
        #will be set at the location of the remaining point and will not satisfy the
        #time-range condition
        return target 
    redchi2=chi2(segment,line)/(n-2)
    global nchi2,nchi2fail
    nchi2+=1
    if(redchi2>1+tol):
        #print("fail redchi2-1=",redchi2-1)
        nchi2fail+=1
        return target 
    return subData(target,line,tol*n)

       
def reduceDataPass(data,chunksize,tol,segwid=3):
    ndata=len(data)
    nchunk=int(ndata/chunksize)+1
    segsize=int(segwid*chunksize)
    noff=int((nchunk*chunksize-ndata)/2)
    #noff=int((nchunk*chunksize-ndata)*np.random.rand())
    nfirst=chunksize
    if(noff>0):nfirst-=noff
    for i in range(nchunk):
        #print("\n****\ni=",i)
        #set the range of the target chunk constraining within bounds
        itargleft=nfirst+(i-1)*chunksize
        if(itargleft<0):itargleft=0
        itargright=nfirst+i*chunksize
        if(itargright>ndata):itargright=ndata
        target=data[itargleft:itargright]
        #time grouping test:
        dtmax=0;dtmin=target[-1,0]-target[0,0]
        for k in range(len(target)-1):
            dt=target[k+1,0]-target[k,0]
            if(dt>dtmax):dtmax=dt
        #for the time grouping test dtmin we expand to the nearest neighbor points (if any)
        for k in range(max(0,itargleft-1),min(ndata-1,itargright+1)):
            dt=data[k+1,0]-data[k,0]
            if(dt<dtmin):dtmin=dt

        if(len(target)<2 or dtmax/dtmin > 30):
            #target too short or times not grouped
            replacement=target.copy()
        else: #passed test so continue
            #print("  target=",target)
            #set the range of the surrounding segment
            isegleft=int((itargleft+itargright-segsize)/2)
            if(isegleft<0):isegleft=0
            isegright=isegleft+segsize
            if(isegright>ndata):isegright=ndata
            #print(" ",isegleft,"--",itargleft,"++",itargright,"--",isegright)
            segment=data[isegleft:isegright]
            #print("  segment=",segment)
            replacement=reduceDataChunk(segment,target,tol).copy()
            #diagnostics:
            #newseg=np.concatenate((data[isegleft:itargleft],replacement,data[itargright:isegright]),axis=0)
            #llsf=DoLLSF(segment,tref=0)[1:3]
            #nllsf=DoLLSF(newseg,tref=0)[1:3]
        #print("  replacement=",replacement)        
        if(i==0):newdata=replacement
        else: newdata=np.append(newdata,replacement,axis=0)
        #print("  newdata=",newdata)        
        #print("  LLSF: ",llsf,"->",nllsf," delta=",llsf-nllsf)
    return newdata

def zeroCounters():
    global nchi2,nchi2fail,nsub,nsubtfail,nsubchi2fail
    nchi2=0
    nchi2fail=0
    nsub=0
    nsubtfail=0
    nsubchi2fail=0


def decimate(origdata, lev, maxpass=1000, ntemper=20, csscale=1000, npretemper=0,verbose=False):
    
    #first configure the data. Internally, we work with 5-column data:
    # t, flux, err**2, tmin, tmax
    #We also support 3 column data: t,flux,err
    data=origdata.copy()
    threecols=data.shape[1]==3
    if threecols:
        data=np.array([[d[0],d[1],d[2]**2,d[0],d[0]] for d in data])

    #first we tune some parameters based on 'lev' option

    #Note that I find similar levels of concentration [and net num of samples] on the peak region for segw=csmin*nwid~75 with csmin varying from 4->10
    #These tests are done with 
    #segw=75,tol=0.25         segw=150,tol=0.25  segw=150,tol=0.5  segw=75,tol=0.5
    #2: n=523 nev=321 F=.61   764 / 1182 = .64   533 / 799 = .67   338 / 476 = .71
    #3: n=736 nev=472 F=.64   704 / 1158 = .61   523 / 823 = .64   330 / 487 = .68
    #4: n=783 nev=421 F=.54   747 / 1196 = .62   536 / 909 = .59   368 / 659 = .56
    #5: n=900 nev=494 F=.55   784 / 1389 = .56   617 /1174 = .53   386 / 744 = .52
    #6: n=796 nev=425 F=.53   728 / 1306 = .62   670 /1140 = .59   437 / 782 = .56
    #7: n=877 nev=485 F=.55   812 / 1409 = .58
    #8: n=917 nev=512 F=.56   797 / 1324 = .60   684 /1253 = .55   384 / 769 = .50
    #9: n=908 nev=504 F=.55
    #10:n=908 nev=493 F=.54   787 / 1283 = .61   695 /1167 = .60
    #11:n=1022 nev=476 F=.46
    #12:n=926  nev=398 F=.43  753 / 1317 = .57   666 /1137 = .59
    #14:n=1109 nev=513 F=.46  819 / 1433 = .57   664 /1188 = .56
    segw=150;tol=0.2;csmin=10

    #here we set up some scalings for these params blending between the following guides
    #lev=0:segw=1000,tol=0.05,csmin=25 #few % red of lens reg. but next 10x reduced overall
    #lev=5:segw=150,tol=0.2,csmin=10   #reduction by factor of ~30 overall
    #lev=10:segw=60,tol=0.5,csmin=2    #reduction by factor ~100 overall
    #lev=15:segw=25,tol=1.0,csmin=2    #reduction by factor >200 overall

    if(lev<=5):
        x=lev/5.0
        segw=int(np.exp(np.log(1000)*(1-x)+np.log(150)*x))
        tol=np.exp(np.log(0.05)*(1-x)+np.log(0.2)*x)
        csmin=int(25*(1-x)+10*x)
        #csmin=10
    elif(lev<=10):
        x=(lev-5)/5.0
        segw=int(np.exp(np.log(150)*(1-x)+np.log(60)*x))
        tol=np.exp(np.log(0.2)*(1-x)+np.log(0.5)*x)
        csmin=int(10*(1-x)+2.0*x)
    else:
        x=(lev-10)/5.0
        segw=int(np.exp(np.log(60)*(1-x)+np.log(25)*x))
        tol=np.exp(np.log(0.5)*(1-x)+np.log(1.0)*x)
        csmin=2
    if(verbose):print("segw,csmin,tol:",segw,csmin,tol)
    nwid=int(segw/csmin)

    ##Now for the actual decimation algorithm

    lastcs=0
    doneAtSize=False
    for i in range(maxpass):
        zeroCounters()
        #with pretempering we begin with a pass of small chunk smoothing to make it less likely to cut small features.
        if(i<npretemper):
            chunksize=int(csmin*np.exp(np.log(csscale/csmin)*(i/(1.0+npretemper))))
            ieff=0
        else:
            ieff=i-npretemper
            chunksize=int(csmin+csscale/(ieff/ntemper*(1+ieff/ntemper)+1))
        if(chunksize==lastcs and doneAtSize):
            #already tried this case
            continue
        #print(i, "ieff=",ieff)
        #print("Trying chunksize=",chunksize)
        newdata = reduceDataPass(data,chunksize,tol,nwid)
        #print("data size ",len(data),"->",len(newdata))
        #print("fail rate: chi2:",nchi2fail/(nchi2+2e-18),"sub t:",nsubtfail/(nsub+2e-18),"sub chi2:",nsubchi2fail/(nsub+2e-18))
        #datallsf=DoLLSF(origdata,tref=0)
        #newdatallsf=DoLLSF(newdata,tref=0)
        #print("llsf:",datallsf[1:3],"->",newdatallsf[1:3]," delta=",(newdatallsf-datallsf)[1:3])

        #termination condition
        if(len(newdata)==len(data) and lastcs==chunksize and i>npretemper):
            if(chunksize<=csmin):
                break
            else: doneAtSize=True
        else:doneAtSize=False
        lastcs=chunksize
        data=newdata
    if threecols:
        data=np.array([[d[0],d[1],np.sqrt(d[2])] for d in data])
        
    return data


def main():

    parser = argparse.ArgumentParser(description='Attempt to decimate data losing minimal information.')
    parser.add_argument('fname', metavar='chain_file', type=str, help='Input file path')
    parser.add_argument('-lev', default="5",help='Level of aggressiveness in data reduction')
    parser.add_argument('-anscol', type=int, default="-1",help='Level of aggressiveness in data reduction')
    parser.add_argument('-plot', action="store_true", help='Plot results instead of saving to file.')
    parser.add_argument('-evalonly', action="store_true", help='Perform evaluation from precomputed results.')
    parser.add_argument('-esterr', action="store_true", help='Roughly estimate error bar level from the first few points.')
    parser.add_argument('-q', action="store_true", help='Run in quiet mode with minimal screen output.')


    args = parser.parse_args()

    lev=int(args.lev)
    tag="lev"+str(lev)

    data=np.loadtxt(args.fname) #Assume reading in t,x,sigma
    #data=np.array([[t,np.random.normal(),1] for t in range(300)])#fake data
    #data=np.array([[t,0.1*(t%2)+t,1] for t in range(10)])#fake data
    #data=np.array([[d[0],d[1],d[2]**2,d[0],d[0]] for d in data])
    #err=np.std([d[2] for d in data[:1600]])
    #print("err=",err)
    #err=np.std([d[2] for d in data[:400]])
    #print("err=",err)
    #err=np.std([d[2] for d in data[:100]])
    #print("err=",err)
    #print("err=",err)
    tcol=0
    dcol=1
    if(args.anscol>=0):
        ans=np.array([[d[0],d[args.anscol]] for d in data])
        if(args.anscol<=tcol):tcol+=1  
        if(args.anscol<=dcol):dcol+=1  
    #print("ans:",ans.shape)
    if(args.esterr):
        err=np.std([d[2] for d in data[:25]])
        if(not args.q):print("Using err=",err)
        data=np.array([[d[tcol],d[dcol],err**2,d[tcol],d[tcol]] for d in data])
    else:
        data=np.array([[d[tcol],d[dcol],d[dcol+1]**2,d[tcol],d[tcol]] for d in data])


    origdata=data.copy()
    data=decimate(data,lev,verbose=not args.q)

    if(not args.q):print("nsamples:",len(origdata),"->",len(newdata))

    if(plot):                                      
        plt.errorbar(origdata[:,0],origdata[:,1],yerr=np.sqrt(origdata[:,2]),fmt="+")
        plt.errorbar(newdata[:,0],newdata[:,1],yerr=np.sqrt(newdata[:,2]),fmt=".")
        icut=int(len(newdata)*9/10)
        plt.errorbar(newdata[:icut,0],newdata[:icut,1],yerr=np.sqrt(newdata[:icut,2]),fmt=".")
        icut=int(len(newdata)*4/5)
        plt.errorbar(newdata[:icut,0],newdata[:icut,1],yerr=np.sqrt(newdata[:icut,2]),fmt=".")
        icut=int(len(newdata)*3/5)
        plt.errorbar(newdata[:icut,0],newdata[:icut,1],yerr=np.sqrt(newdata[:icut,2]),fmt=".")
        icut=int(len(newdata)*2/5)
        plt.errorbar(newdata[:icut,0],newdata[:icut,1],yerr=np.sqrt(newdata[:icut,2]),fmt=".")
        icut=int(len(newdata)/5)
        plt.errorbar(newdata[:icut,0],newdata[:icut,1],yerr=np.sqrt(newdata[:icut,2]),fmt=".")
        icut=int(len(newdata)/10)
        plt.errorbar(newdata[:icut,0],newdata[:icut,1],yerr=np.sqrt(newdata[:icut,2]),fmt=".")
        if(args.anscol>=0 and False):
            plt.plot(ans[:,0],ans[:,1],"k-",linewidth=2)

    newdata=np.array([[d[0],d[1],np.sqrt(d[2])] for d in newdata])
    if(".txt" in args.fname):
        outfile=args.fname.replace(".txt","_"+tag+".dat")
    elif(".dat" in args.fname):
        outfile=args.fname.replace(".dat","_"+tag+".dat")
    elif(".out" in args.fname):
        outfile=args.fname.replace(".out","_"+tag+".dat")
    else:
        outfile=args.fname+".dat"


    if(not args.plot and args.anscol>=0):
        #Given a noise free 'answer' we estimate errors from the decimation in two ways.
        #Method 1: Estimate the reduced chi2 deviation of the true data from the decimated data
        #  In this case the true values are linearly interpolated to the decimated sample points
        #  and the variance comes from the decimated data estimate.
        diff=newdata[:,1]-np.interp(newdata[:,0],ans[:,0],ans[:,1])
        var=newdata[:,2]**2
        #print(diff,var)
        ncount=len(diff)
        rchi2a=np.sum(diff*diff/var)/ncount
        #Method 2: Estimate the reduced chi2 deviation of the decimated data from the true data
        #  In this case the decimated values are linearly interpolated to the original sample points
        #  and the variance comes from the original data err estimate.
        diff=ans[:,1]-np.interp(ans[:,0],newdata[:,0],newdata[:,1])
        var=origdata[:,2]
        #print(diff,var)
        ncount=len(diff)    
        rchi2b=np.sum(diff*diff/var)/ncount
        if(not args.q):
            print("err estimates")
            print(rchi2a,rchi2b)
        else:
            print(args.fname,len(origdata),len(newdata),rchi2a,rchi2b)

    if(args.plot):                                      
        plt.show()
    elif(not args.evalonly):
        if(not args.q):print("outfile=",outfile)
        np.savetxt(outfile,newdata)



if __name__ == "__main__": main()
