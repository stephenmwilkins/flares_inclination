import timeit
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt

from FLARE.photom import lum_to_M, M_to_lum
from helpers import lum


def get_data(ii, tag, inp = 'FLARES'):

    num = str(ii)
    if inp == 'FLARES':
        if len(num) == 1:
            num =  '0'+num

        sim = rF"../flares_pipeline/data/flares.hdf5"
        num = num+'/'

    else:
        sim = rF"../flares_pipeline/data/EAGLE_{inp}_sp_info.hdf5"
        num=''

    with h5py.File(sim, 'r') as hf:
        mstar = np.array(hf[num+tag+'/Galaxy'].get('Mstar_30'), dtype = np.float64)*1e10
        S_len = np.array(hf[num+tag+'/Galaxy'].get('S_Length'), dtype = np.int64)
        S_mass = np.array(hf[num+tag+'/Particle'].get('S_MassInitial'), dtype = np.float64)*1e10
        S_Z =  np.array(hf[num+tag+'/Particle'].get('S_Z_smooth'), dtype = np.float64)
        S_age = np.array(hf[num+tag+'/Particle'].get('S_Age'), dtype = np.float64)*1e3

    begin = np.zeros(len(S_len), dtype = np.int64)
    end = np.zeros(len(S_len), dtype = np.int64)
    begin[1:] = np.cumsum(S_len)[:-1]
    end = np.cumsum(S_len)

    return mstar, S_mass, S_Z, S_age, begin, end

tag='010_z005p000'
sim_type='FLARES'
ii=0

num = str(ii)
if len(num) == 1:
    num = '0'+num

mstar, S_mass, S_Z, S_age, begin, end = get_data(num, tag, inp = 'FLARES')
z = float(tag[5:].replace('p','.'))
req_ind = np.where(mstar>10**9.5)[0]

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(5, 5), sharex=True, sharey=True, facecolor='w', edgecolor='k')

for kk, jj in enumerate(req_ind):
    print (F"Computing attenuations for task {kk}/{len(req_ind)}")
    #Coordinates and attributes for the jj galaxy in ii region
    this_smass = S_mass[begin[jj]:end[jj]]
    this_sZ = S_Z[begin[jj]:end[jj]]
    this_age = S_age[begin[jj]:end[jj]]
    L_FUV_int = lum(MetSurfaceDensities=np.zeros(len(this_smass)),Masses=this_smass, Ages=this_age, Metallicities=this_sZ, DTM=0., Type='Intrinsic')

    with h5py.File(F'data/Zlos_inclination_{num}.hdf5', 'r') as hf:
        # Zlos = np.array(hf[F"{tag}"].get(F'S_los_{jj}'), dtype = np.float64)
        LFUV = np.array(hf[F"{tag}/Luminosity"].get(F'LFUV_{jj}'), dtype = np.float64)
        LNUV = np.array(hf[F"{tag}/Luminosity"].get(F'LNUV_{jj}'), dtype = np.float64)


    att = -2.5*np.log10(LFUV/L_FUV_int) #np.log10(LFUV/LNUV)/np.log10(1500./2500.) - 2.0

    x = np.array([np.log10(L_FUV_int)])
    y = np.array([np.median(att)])
    y16 = np.array([y-np.percentile(att, 16)])
    y84 = np.array([np.percentile(att, 84)-y])

    ax.errorbar(x, y, yerr=[y16,y84], marker='o', color='red')


# ax.set_ylabel(r'A$_{\mathrm{FUV}}$=-2.5 log$_{10}$(L$_{\mathrm{FUV}}^{\mathrm{Observed}}$/L$_{\mathrm{FUV}}^{\mathrm{Intrinsic}}$)', fontsize=12)
#ax.set_ylabel(r"$\beta$", fontsize=12)

#ax.set_xlabel(r'log$_{10}$(M$_{*}$/M$_{\odot}$)', fontsize=12)
ax.set_xlabel(r'log$_{10}$(L$_{FUV\,int}$/(ergs/s/Hz))', fontsize=12)

plt.savefig(F'LFUV_int_att_orient_{num}.png', dpi = 300, bbox_inches='tight')

plt.show()
