##Script to create and submit an MBHBmcmc_json.py job on slurm taking
##all relevant info from the configuration file.
hb_repo_dir='/gpfsm/dnb31/jgbaker//tess/heartbeats/repos/'
hb_mcmc_py=hb_repo_dir+'tess-short-binaries/src/mcmc/HB_MCMC2.py'
ptmcmc_dir=hb_repo_dir+'ptmcmc/'
data_dir=hb_repo_dir+'HB_MCMC/data/lightcurves/[oe]*/'
styles_dir='/gpfsm/dnb31/jgbaker/tess/heartbeats/runs_spr22/'

import sys
import os
import subprocess
import json
import pathlib
import shutil
import glob

debug=False

submission_template='''#!/bin/bash
#SBATCH --job-name=LABEL
#SBATCH --nodes=NODES --ntasks=NPROCS
#SBATCH --constraint=CONSTRAINT
#SBATCH --time=TIME:00:00
#SBATCH --account=s0982
#DEBUG

source /etc/profile.d/modules.sh
#module load python/GEOSpyD comp/gcc/9.3.0 mpi/impi/2021.4.0
module load python/GEOSpyD comp/intel/19.1.3.304 mpi/impi/19.1.3.304
#source /usr/local/other/python/GEOSpyD/4.9.2_py3.9/2021-05-05/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate hb.env

ulimit -s unlimited
export KMP_STACKSIZE=1G
export KMP_AFFINITY=compact
export I_MPI_PIN_DOMAIN=auto

np=NPROCS
export OMP_NUM_THREADS=1;

label=LABEL
cd RUN_DIR
ln -s $ptmcmc .

contflag=""
x=0
if ls -1rtd *-cp>/dev/null 2>&1; then  
    contflag="--restart_dir=`ls -1rtd *-cp|tail -n 1`";
    x=`ls -1rtd ${label}.out.X?|tail -n 1`
    #echo "x=$x"
    x="${x:$((${#x}-1)):1}"
    #echo "x=$x"
    (( x++ ))
    echo "x=$x"
fi

#preliminary set up
cmd="python3 -u HB_MCMC_PY --data_style=data_style.json --hb_style=hb_style.json --nPlot=0 --savePfig=period-fit.png --saveLfig=data-plot.png"
echo $cmd
$cmd > figs.out 2>&1

cmd="python3 -u HB_MCMC_PY --seed=0.082370SEED_N --outname=LABEL --data_style=data_style_Pfit.json --hb_style=hb_style.json --mcmc_style=mcmc_style.json ${contflag} "
echo $cmd
cp ${label}.out ${label}.out.X$x
(time mpirun -np $np $cmd)  1> ${label}.out  2>&1

cmd="python3 -u ${ptmcmc}/python/ess.py ${label}_t0.dat"
echo $cmd
(time mpirun -np $np $cmd)  1> ess.out  2>&1
cp ${label}.out ${label}.out.X$x
'''

argv=sys.argv[:]
if len(argv)!=7:
    print('Usage:\n python3 submit.py TICid data_style hb_style mcmc_style seed_n run_style')
    #print('got:',argv)
    sys.exit()
TICid=argv[1]
data_style=argv[2]
hb_style=argv[3]
mcmc_style=argv[4]
seed_n=int(argv[5])
run_style=argv[6]

#prep case dir and run dir
case_dir=TICid+'/'
pathlib.Path(case_dir).mkdir(exist_ok=True)
#create a symbolic link to the ptmcmc directory (a little hacky that the code needs this) 
if not os.path.exists(case_dir+'ptmcmc'):
    os.symlink(ptmcmc_dir,case_dir+'ptmcmc')
label=TICid+'_'+data_style+'_'+hb_style+'_'+mcmc_style+'_'+str(seed_n)+'_'+run_style
run_dir=case_dir+label+'/'
pathlib.Path(run_dir).mkdir(exist_ok=True)

#compute the datafile path
#dfname=TICid+'.txt'
#dfpath=data_dir+dfname
dfpathlist=glob.glob(data_dir+"*"+TICid+"*")
if len(dfpathlist)==0: raise FileNotFoundError(data_dir+"*"+TICid+"*")
if len(dfpathlist)>1:
    print("Multiple possible files with "+TICid+" in "+data_dir+". Looking for one  ending in .txt")
    newlist=[p for p in dfpathlist if p.endswith('.txt')]
    if len(newlist)>0:dfpathlist=newlist
dfpath=dfpathlist[0]

#prep data_styles
dsfname='data_styles.json'
dsfpath=case_dir+dsfname
dsf=pathlib.Path(dsfpath)
#if the data_styles file doesn't exist we create one from the template
if not dsf.is_file():
    with open(styles_dir+dsfname,'r') as sfile: dsdict=json.load(sfile)
    #populate info specific for TICid
    for style in dsdict:
        dsdict[style]['id']=TICid
        print('dfpath:',dfpath)
        dsdict[style]['datafile']=dfpath
    #write
    with open(dsfpath,'w') as sfile:        
        json.dump(dsdict,sfile,indent=4)
else:
    with open(dsfpath,'r') as sfile:        
        dsdict=json.load(sfile)
ds=dsdict[data_style]
with open(run_dir+'data_style.json','w') as sfile: json.dump(ds,sfile,indent=4)

#prep run pars
sfname='run_styles.json'
with open(styles_dir+sfname,'r') as sfile:        
    sdict=json.load(sfile)
runpars=sdict[run_style]
with open(run_dir+'run_style.json','w') as sfile: json.dump(runpars,sfile,indent=4)

#prep hb style:
sfname='hb_styles.json'
with open(styles_dir+sfname,'r') as sfile:        
    sdict=json.load(sfile)
hbs=sdict[hb_style]
with open(run_dir+'hb_style.json','w') as sfile: json.dump(hbs,sfile,indent=4)

#prep mcmc style
sfname='mcmc_styles.json'
with open(styles_dir+sfname,'r') as sfile:        
    sdict=json.load(sfile)
mcs=sdict[mcmc_style]
mcs['checkp_at_time']=mcs.get('checkp_at_time',str(float(runpars['TIME'])-.25))
with open(run_dir+'mcmc_style.json','w') as sfile: json.dump(mcs,sfile,indent=4)

#prep submission script tags
tags='LABEL,RUN_DIR,SEED_N,CONSTRAINT,NODESIZE,NPROCS,TIME,NODES,HB_MCMC_PY,DEBUG'.split(',')
defaults=[label,run_dir,str(seed_n)]+'hasw,28,8,12,,,,,,'.split(',')
tag_info=dict(zip(tags,defaults))
for tag in tags[1:-1]: tag_info[tag]=runpars.get(tag,tag_info[tag])
nodes=(int(tag_info['NPROCS'])-1)//int(tag_info['NODESIZE'])+1
tag_info['NODES']=runpars.get('NODES',nodes)
tag_info['HB_MCMC_PY']=hb_mcmc_py
if debug: tag_info['DEBUG']="SBATCH --time=1:00:00 --qos=debug"
print(tag_info)

subpath=run_dir+label+'.sub'
script=submission_template
for tag in tags: script=script.replace(tag,str(tag_info[tag]))
print('Writing slurm script',subpath)
with open(subpath,'w') as f: f.write(script)

print('Submitting job to slurm')
print(subprocess.check_output(['sbatch', subpath]).decode("utf-8"))
