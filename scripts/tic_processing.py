from astroquery.mast import Catalogs,Observations
from astropy.coordinates import Angle,SkyCoord
import numpy as np
from requests.models import HTTPError
from pathlib import Path
import pandas as pd

#Color photometry converters
#from https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html#Ch5.T7

def fVmag(G,GBP,GRP):
    #Based on forumla give for computing Tycho-2 Vmag from GAIA band magnitudes; scatter sigma=0.05501
    x=GBP-GRP
    return G - (-0.01842 - 0.06629*x - 0.2346*x**2 + 0.02157*x**3)
    
def fVmag_e(G_foe,GBP_foe,GRP_foe):
    #Based on forumla give for computing Tycho-2 Vmag from GAIA band magnitudes; scatter sigma=0.05501
    #To estimate uncertainty we add err estimates for each dependency in quadrature
    # dV**2 = dG**2 + |dV/dGBP|**2 dGBP**2 + |dV/dGRP|**2 dGRP**2 + sigma**2
    # We compute the magnitute errors from the GAIA flux over error values e.g gG_foe 
    # by mag_err = 2.5*log10(1+1/foe) ~ 1.0857/foe
    # We evaluate the derivatives at x=0 so
    # |dV/dGBP| = |dV/dGRP| = deriv
    # Note that if we should ultimately an error on the B-V difference, computed by these
    # formulae, then dropping the common G_e contribution would be more appropriate, though we
    # generally don't expect G_e to dominate
    deriv = 0.06629
    G_e=1.0857/G_foe
    GBP_e=1.0857/GBP_foe
    GRP_e=1.0857/GRP_foe
    return np.sqrt( G_e**2 + deriv**2 * ( GBP_e**2 + GRP_e**2 ) )
    
def fBmag(G,GBP,GRP):
    #Based on forumla give for computing Tycho-2 Vmag from GAIA band magnitudes; scatter sigma=0.07293
    x=GBP-GRP
    return G - (-0.02441 - 0.4899*x - 0.9740*x**2 + 0.2496*x**3)

def fBmag_e(G_foe,GBP_foe,GRP_foe):
    #Based on forumla give for computing Tycho-2 Vmag from GAIA band magnitudes; scatter sigma=0.05501
    #To estimate uncertainty we add err estimates for each dependency in quadrature
    # dB**2 = dG**2 + |dB/dGBP|**2 dGBP**2 + |dB/dGRP|**2 dGRP**2 + sigma**2
    # We compute the magnitute errors from the GAIA flux over error values e.g gG_foe 
    # by mag_err = 2.5*log10(1+1/foe) ~ 1.0857/foe    # We evaluate the derivatives at x=0 so
    # |dB/dGBP| = |dB/dGRP| = deriv
    # Note that if we should ultimately an error on e.g. the B-V difference, computed by these
    # formulae, then dropping the common G_e contribution would be more appropriate, though we
    # generally don't expect G_e to dominate
    deriv = 0.4899
    G_e=1.0857/G_foe
    GBP_e=1.0857/GBP_foe
    GRP_e=1.0857/GRP_foe
    return np.sqrt( G_e**2 + deriv**2 * ( GBP_e**2 + GRP_e**2 ) )

def fTmag(G,GBP,GRP):
    #Based on eq 1 from https://iopscience.iop.org/article/10.3847/1538-3881/ab3467
    #for TESS magnitude; scatter sigma=0.006
    #The text says something about dereddening of the GAIA colors, which we don't do.    
    x=GBP-GRP
    if np.isnan(x): return G - 0.430
    return G - 0.00522555*x**3 + 0.0891337*x**2 - 0.633923*x + 0.0324473

def fTmag_e(G_foe,GBP_foe,GRP_foe):
    #Based on forumla give for computing Tycho-2 Vmag from GAIA band magnitudes; scatter sigma=0.05501
    #To estimate uncertainty we add err estimates for each dependency in quadrature
    # dT**2 = dG**2 + |dT/dGBP|**2 dGBP**2 + |dT/dGRP|**2 dGRP**2 + sigma**2
    # We compute the magnitute errors from the GAIA flux over error values e.g gG_foe 
    # by mag_err = 2.5*log10(1+1/foe) ~ 1.0857/foe    # We evaluate the derivatives at x=0 so
    # |dT/dGBP| = |dT/dGRP| = deriv
    # Note that if we should ultimately an error on the B-V difference, computed by these
    # formulae, then dropping the common G_e contribution would be more appropriate, though we
    # generally don't expect G_e to dominate
    deriv = 0.633923
    G_e=1.0857/G_foe
    GBP_e=1.0857/GBP_foe
    GRP_e=1.0857/GRP_foe
    return np.sqrt( G_e**2 + deriv**2 * ( GBP_e**2 + GRP_e**2 ) )

def get_color_photometry_data(TICid):
    '''
    Queries MAST TIC/GAIA catalogs to get photometric information for blending and color models.
    '''
    flags=[]
    
    # Get the RA and Dec data
    TICData0 = Catalogs.query_object('TIC '+str(TICid),radius=0.001,catalog='TIC')[0]
    radec=' '.join([str(x) for x in TICData0['ra','dec']])
    GAIAData1px=Catalogs.query_object(radec,unit='deg',radius=0.00586,catalog='Gaia')
    GAIAData2px=Catalogs.query_object(radec,unit='deg',radius=0.00586*2,catalog='Gaia')
    GAIAData4px=Catalogs.query_object(radec,unit='deg',radius=0.00586*4,catalog='Gaia')
    #TICData4px=Catalogs.query_object(radec,unit='deg',radius=0.00586*4,catalog='TIC')
    
    if TICData0['disposition']!='--':flags.append('DISPOSITION='+TICData0['disposition'])
    
    #Get GAIA ID
    GID = TICData0['GAIA']
    print('GID',GID)
    if not GID or GID=='--':flags.append('NO_GAIA_ID')
    
        
    #display(GAIAData1px)
    Gmag0=TICData0['GAIAmag']
    if not Gmag0 or np.isnan(Gmag0):
        Gmag0=TICData0['Tmag']+0.43#No GAIA assoc. Use a default guess for Tmag->Gman from Stassun19
    #Gmag0=GAIAData1px['phot_g_mean_mag'].min()
    #brightestGID =  GAIAData1px[GAIAData1px['phot_g_mean_mag']==Gmag0]['source_id'].value
    #print('brightestGID',brightestGID)
    #if GID!=brightestGID: raise ValueError('TIC object GID='+str(GID)+' does not match the brightest object within 1px (GID='+str(brightestGID)+')')


    #estimate blending 1px,2px,4px blending fractions
    #  compute blending fracs using the sum of flux of GAIAobj over the region
    #  with blend_frac =  1-f0/fsum
    f0=10**(-0.4*Gmag0)
    bfs=[1-f0/np.sum(10**(-0.4*df['phot_g_mean_mag'].value)) for df in [GAIAData1px,GAIAData2px,GAIAData4px]]
    blending=np.mean(bfs)
    blending_err=np.std(bfs)
    print('bfs:',bfs)
    print("Estimated blending is: %f +/- %f" %(blending,blending_err))
    if blending<0 or blending>1: 
        blending=0.5
        blending_err=2
        flags.append("BLEND_BAD")
                     
    
    #Gather auxiliary photometry data
    TICrow = TICData0
    Tmag = TICrow['Tmag']
    Tmag_e = TICrow['e_Tmag']
    Bmag = TICrow['Bmag']
    Bmag_e = TICrow['e_Bmag']
    Vmag = TICrow['Vmag']
    Vmag_e = TICrow['e_Vmag']
    EBV = TICrow['ebv']
    EBV_e = TICrow['e_ebv']
    Gmag = TICrow['GAIAmag']
    Gmag_e = TICrow['e_GAIAmag']
    dist = TICrow['d']
    dist_e = TICrow['e_d']
    if np.isnan(dist_e): dist_e=None
    if np.isnan(Gmag): Gmag=None
    if np.isnan(Tmag): Tmag=None
    if np.isnan(Vmag): Vmag=None
    if np.isnan(EBV): EBV=None
    if not dist_e:dist_e=dist   #If there is no err estimate for the distance, set the error equal to the dist
    
    #If Vmag came from GAIA, then it isn't independent, so we drop it. The relevant color info is in G-T
    if TICrow['VmagFlag']=='gaia2': 
        Vmag=None
        flags.append("DROP_GAIA_VMAG")
    
    #Compute various colors and err estimates
    Gmag0=None
    BmV=None
    VmG=None
    GmT=None
    Gmag0_e=None
    BmV_e=None
    VmG_e=None
    GmT_e=None
    if Bmag and Vmag:
        BmV=Bmag-Vmag
        BmV_e=np.sqrt(Bmag_e**2+Vmag_e**2)
    if Vmag and Gmag:
        VmG=Vmag-Gmag
        VmG_e=np.sqrt(Gmag_e**2+Vmag_e**2)
    if Tmag and Gmag:
        GmT=Gmag-Tmag
        GmT_e=np.sqrt(Gmag_e**2+Tmag_e**2)
    
    #Estimate dereddened GAIA magnitude and colors
    #These come from the TIC's E(B-V) and scaling estimates in the TIC paper (Stessun2019) 
    #for A_G and A_T extinctions:
    #  A_G=2.72 E(B-V)
    #  A_T=2.06 E(B-V)
    #We use a generic value for V band Rv=3.1
    if not EBV: 
        #If we have no data on reddening, then we somewhat conservatively assume E(B-V)~1\pm1
        EBV=1
        EBV_e=1
        flags.append("GUESS_EBV")
    if Gmag:
        Gmag0   = Gmag - 2.72*EBV
        Gmag0_e = np.sqrt(Gmag_e**2+(2.72*EBV_e)**2+EBV**2/4) #We add 1/2 EBV unc for the unknown accuracy/scatter of the offset model
    if BmV:
        BmV     = BmV - EBV
        BmV_e   = np.sqrt(BmV_e**2+EBV_e**2+EBV**2/4)#We add 1/2 EBV unc for the unknown accuracy of the offset model
    if VmG:
        VmG     = VmG - 0.38*EBV
        VmG_e   = np.sqrt(VmG_e**2+0.38*(EBV_e) + EBV**2/4)#We add 1/2 EBV unc for the unknown accuracy of the offset model
    if GmT:
        GmT     = GmT - 0.66*EBV
        GmT_e   = np.sqrt(GmT_e**2+(0.66*EBV_e)**2)#We add 1/2 EBV unc for the unknown accuracy of the offset model 
    
    # Adjust Gmag0 for distance error 
    #(a little wonky, but we actually use the absolute magtinude and thus account for its error here)
    if Gmag0 and Gmag0_e and dist and dist_e:
        Gmag_e = Gmag_e + (5 * dist_e / dist / np.log(10))
        
    color_phot_data={}
    color_phot_data['dist']=dist
    color_phot_data['dist_e']=dist_e    
    color_phot_data['Gmag0']=Gmag0
    color_phot_data['Gmag0_e']=Gmag0_e
    color_phot_data['BmV0']=BmV
    color_phot_data['BmV0_e']=BmV_e
    color_phot_data['VmG0']=VmG
    color_phot_data['VmG0_e']=VmG_e
    color_phot_data['GmT0']=GmT
    color_phot_data['GmT0_e']=GmT_e
    color_phot_data['EBV']=EBV
    color_phot_data['EBV_e']=EBV_e
    color_phot_data["blend"]=blending
    color_phot_data["blend_e"]=blending_err
    color_phot_data["TIC_Gmag"]=Gmag
    color_phot_data["TIC_Teff"]=TICData0['Teff']
    color_phot_data["TIC_Teff_e"]=TICData0['e_Teff']
    color_phot_data["TIC_rad"]=TICData0['rad']
    color_phot_data["TIC_rad_e"]=TICData0['e_rad']
    color_phot_data["TIC_mass"]=TICData0['mass']
    color_phot_data["TIC_mass_e"]=TICData0['e_mass']
    color_phot_data["flags"]=flags
    
    return color_phot_data

def make_cp_data_df(ticids):
    '''
    Applies get_color_photometry_data() func to repeatedly query MAST to collect 
    color photometry data.
    If the query fails, then it returns with whatever it has so far.
    '''
    df=None
    
    for tic in ticids:
        try:
            cpdict=get_color_photometry_data(tic)
        except:
            return df
        #print('df')
        #display(df)
        keys=list(cpdict.keys())
        cols=['TIC_ID']+keys
        adddf=pd.DataFrame([[tic]+[cpdict[k] for k in keys]],columns=cols)
        #print('adddf')
        #display(adddf)
        if df is None:
            df=adddf
        else:
            df=df.append(adddf,ignore_index=True)
    return df

def append_cp_data(df,addeddf):
    '''
    Append a new cp_data DataFrame to an existing one, but replace any rows
    already present (as identified by TIC_ID)
    '''
    newdf=df
    if addeddf is None: return df
    for it,row in addeddf.iterrows():
        #print('looking for',row['TIC_ID'],' in ',newdf['TIC_ID'])
        if row['TIC_ID'] in newdf['TIC_ID'].values or int(row['TIC_ID']) in newdf['TIC_ID'].values: #replace row if present
            #print('found')
            newdf[newdf['TIC_ID']==row['TIC_ID']]=row
        else:
            #print('not found')
            newdf=newdf.append(row,ignore_index=True) #or else add a new row
    return newdf

def make_cp_data(ticids, filename,maxchunk=50):
    '''
    Collect cp_data on a number of tics and write/append to csv_file.
    If the file exists then use append_cp_data() to append, otherwise
    write a new csv file.
    Use make_cp_data_df() to make the new part of the cp_data. If the 
    function returns early, or maxchunk is exceeded, then write what we
    have so far and continue from there (by recursion)
    Return the dataframe
    '''
    fp=Path(filename)
    if not fp.parent.is_dir(): raise ValueError("No directory at '"+str(fp.parent)+"'")
    if fp.exists():
        df=pd.read_csv(filename)
    else:
        df=None
    iend=min(len(ticids),maxchunk)
    if iend==0 and df is None: raise ValueError('Empty list of TIC IDs')
    if iend==0: return df #Probably won't happen, but just in case.
    adddf=make_cp_data_df(ticids[:iend])
    if df is not None:
        newdf=append_cp_data(df,adddf)
    else:
        newdf=adddf
    #write result to file
    newdf=newdf.fillna(float('NaN'))
    newdf.to_csv(filename,index=False)

    #continue if more to do
    if adddf is None:
        print('Got a null cp_dataframe, retyring.')
        return make_cp_data(ticids,filename,maxchunk=maxchunk)
    elif len(adddf)<len(ticids):
        print('Added',len(adddf),'to cp_dataframe. Length is now',len(newdf))
    return make_cp_data(ticids[len(adddf):],filename,maxchunk=maxchunk)
    
    #return df if done
    return newdf

