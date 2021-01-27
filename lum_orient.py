import timeit
import numpy as np
import h5py
from functools import partial
import schwimmbad
import matplotlib
import matplotlib.pyplot as plt

import SynthObs
from SynthObs.SED import models
import FLARE
import FLARE.filters
from FLARE.photom import lum_to_M, M_to_lum

from helpers import lum
from phot_modules import DTM_fit
import flares


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
        G_len = np.array(hf[num+tag+'/Galaxy'].get('G_Length'), dtype = np.int64)
        S_mass = np.array(hf[num+tag+'/Particle'].get('S_MassInitial'), dtype = np.float64)*1e10
        S_Z =  np.array(hf[num+tag+'/Particle'].get('S_Z_smooth'), dtype = np.float64)
        S_age = np.array(hf[num+tag+'/Particle'].get('S_Age'), dtype = np.float64)*1e3
        G_Z = np.array(hf[num+tag+'/Particle'].get('G_Z_smooth'), dtype = np.float64)

    begin = np.zeros(len(S_len), dtype = np.int64)
    end = np.zeros(len(S_len), dtype = np.int64)
    begin[1:] = np.cumsum(S_len)[:-1]
    end = np.cumsum(S_len)

    gbegin = np.zeros(len(G_len), dtype = np.int64)
    gend = np.zeros(len(G_len), dtype = np.int64)
    gbegin[1:] = np.cumsum(G_len)[:-1]
    gend = np.cumsum(G_len)


    return mstar, S_mass, S_Z, S_age, G_Z, begin, end, gbegin, gend


tag='010_z005p000'
sim_type='FLARES'
ii=0

num = str(ii)
if len(num) == 1:
    num = '0'+num

mstar, S_mass, S_Z, S_age, G_Z, begin, end, gbegin, gend = get_data(num, tag, inp = 'FLARES')


z = float(tag[5:].replace('p','.'))
req_ind = np.where(mstar>10**9.5)[0]

filename = F'data/Zlos_inclination_{num}.hdf5'
fl = flares.flares(fname = filename, sim_type = sim_type)
fl.create_group(F'{tag}/Luminosity')


for kk, jj in enumerate(req_ind):
    print (F"Computing attenuations for task {kk}/{len(req_ind)}")
    #Coordinates and attributes for the jj galaxy in ii region

    this_smass = S_mass[begin[jj]:end[jj]]

    this_sZ = S_Z[begin[jj]:end[jj]]
    this_gZ = G_Z[gbegin[jj]:gend[jj]]

    this_age = S_age[begin[jj]:end[jj]]

    Mage = np.nansum(this_smass*this_age)/np.nansum(this_smass)
    Z = np.nanmean(this_gZ)
    DTM = DTM_fit(Z, Mage)

    with h5py.File(F'data/Zlos_inclination_{num}.hdf5', 'r') as hf:
        Zlos = np.array(hf[tag].get(F'S_los_{jj}'), dtype = np.float64)


    calc_lum = partial(lum, Masses=this_smass, Ages=this_age, Metallicities=this_sZ, DTM=DTM, filters = ['FAKE.TH.FUV', 'FAKE.TH.NUV'])
    pool = schwimmbad.MultiPool(processes=28)
    dat = np.array(list(pool.map(calc_lum, Zlos)))
    pool.close()


    fl.create_dataset(dat[:,0], F'LFUV_{jj}', F'{tag}/Luminosity',
            desc = "Dust corrected FUV luminosity (using ModelI) of the galaxy",
            unit = 'ergs/s/Hz')
    fl.create_dataset(dat[:,1], F'LNUV_{jj}', F'{tag}/Luminosity',
            desc = "Dust corrected NUV luminosity (using ModelI) of the galaxy",
            unit = 'ergs/s/Hz')
