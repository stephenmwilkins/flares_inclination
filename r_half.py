import timeit, sys
import numpy as np
import pandas as pd
import healpy as hp
import h5py
from functools import partial
import schwimmbad
import matplotlib
import matplotlib.pyplot as plt

import flares
from phot_modules import DTM_fit
from FLARE.photom import lum_to_M, M_to_lum
from helpers import lum_from_stars, calc_halflightradius


def get_data(ii, tag, inp = 'FLARES'):

    num = str(ii)
    if inp == 'FLARES':
        if len(num) == 1:
            num =  '0'+num

        sim = rF"../flares_pipeline/data/FLARES_{num}_sp_info.hdf5"
        num = ''#num+'/'

    else:
        sim = rF"../flares_pipeline/data/EAGLE_{inp}_sp_info.hdf5"
        num=''

    with h5py.File(sim, 'r') as hf:
        cop = np.array(hf[num+tag+'/Galaxy'].get('COP'), dtype = np.float64)
        mstar = np.array(hf[num+tag+'/Galaxy'].get('Mstar_30'), dtype = np.float64)*1e10
        S_len = np.array(hf[num+tag+'/Galaxy'].get('S_Length'), dtype = np.int64)
        G_len = np.array(hf[num+tag+'/Galaxy'].get('G_Length'), dtype = np.int64)
        S_coords = np.array(hf[num+tag+'/Particle'].get('S_Coordinates'), dtype = np.float64)
        G_coords = np.array(hf[num+tag+'/Particle'].get('G_Coordinates'), dtype = np.float64)
        S_mass = np.array(hf[num+tag+'/Particle'].get('S_MassInitial'), dtype = np.float64)*1e10
        G_mass = np.array(hf[num+tag+'/Particle'].get('G_Mass'), dtype = np.float64)*1e10
        S_Z =  np.array(hf[num+tag+'/Particle'].get('S_Z_smooth'), dtype = np.float64)
        S_age = np.array(hf[num+tag+'/Particle'].get('S_Age'), dtype = np.float64)*1e3
        S_los = np.array(hf[num+tag+'/Particle'].get('S_los'), dtype = np.float64)
        S_sml = np.array(hf[num+tag+'/Particle'].get('S_sml'), dtype = np.float64)
        G_sml = np.array(hf[num+tag+'/Particle'].get('G_sml'), dtype = np.float64)
        G_Z = np.array(hf[num+tag+'/Particle'].get('G_Z_smooth'), dtype = np.float64)
        lfuv = lum_to_M(np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI'].get('FUV'), dtype = np.float64))

    begin = np.zeros(len(S_len), dtype = np.int64)
    end = np.zeros(len(S_len), dtype = np.int64)
    begin[1:] = np.cumsum(S_len)[:-1]
    end = np.cumsum(S_len)

    gbegin = np.zeros(len(G_len), dtype = np.int64)
    gend = np.zeros(len(G_len), dtype = np.int64)
    gbegin[1:] = np.cumsum(G_len)[:-1]
    gend = np.cumsum(G_len)


    return S_len, cop, mstar, S_coords, G_coords, S_mass, G_mass, S_Z, S_age, S_los, S_sml, G_sml, G_Z, begin, end, gbegin, gend, S_len, lfuv

def get_half_lr(jj, S_mass, S_Z, G_Z, S_age, S_los, S_coords, S_sml, begin, end, z):

    this_smass = S_mass[begin[jj]:end[jj]]

    this_sZ = S_Z[begin[jj]:end[jj]]
    this_gZ = G_Z[gbegin[jj]:gend[jj]]

    this_age = S_age[begin[jj]:end[jj]]
    this_los = S_los[begin[jj]:end[jj]]

    this_scoord = (S_coords[:, begin[jj]:end[jj]].T - cop[:,jj])/(1+z)
    this_sml = S_sml[begin[jj]:end[jj]]

    Mage = np.nansum(this_smass*this_age)/np.nansum(this_smass)
    Z = np.nanmean(this_gZ)
    DTM = DTM_fit(Z, Mage)

    lum = np.concatenate(lum_from_stars(MetSurfaceDensities = this_los, Masses=this_smass, Ages=this_age, Metallicities=this_sZ, DTM=DTM, filters = ['FAKE.TH.FUV']))
    lum_intrinsic = np.concatenate(lum_from_stars(MetSurfaceDensities = this_los, Masses=this_smass, Ages=this_age, Metallicities=this_sZ, DTM=0., filters = ['FAKE.TH.FUV'], Type = 'Intrinsic'))

    hlr = calc_halflightradius(this_scoord*1e3, lum, this_sml*1e3, z)

    return hlr, lum, lum_intrinsic


sim_type='FLARES'
ii = int(sys.argv[1])
tag = str(sys.argv[2])
z = float(tag[5:].replace('p','.'))

num = str(ii)
if len(num) == 1:
    num = '0'+num


S_len, cop, mstar, S_coords, G_coords, S_mass, G_mass, S_Z, S_age, S_los, S_sml, G_sml, G_Z, begin, end, gbegin, gend, S_len, lfuv = get_data(num, tag, inp = 'FLARES')






req_ind = np.where(S_len>=1000)[0]
# for jj in range(len(req_ind)):
#     kk = req_ind[jj]
#     if lfuv[kk]<-21:
#
#
#         with h5py.File(F"./data/FLARES_{num}_sp_info.hdf5", 'r') as hf:
#             rhalf = np.array(hf[tag].get('R_half'), dtype = np.float64)
#             lum = np.array(hf[tag+F'/Luminosity/{jj}'].get('FUV'))
#             lum_int = np.array(hf[tag+F'/Luminosity/{jj}'].get('FUV_intrinsic'))
#
#         this_scoord = S_coords[:, begin[kk]:end[kk]].T - cop[:,kk]
#         dist_r = np.linalg.norm(this_scoord, axis=1)
#         r_bins = np.arange(0,4,0.25)*rhalf[jj]
#         r_bins_cen = (r_bins[1:]+r_bins[:-1])/2.
#         att = np.array([])
#
#         for ii in range(len(r_bins)-1):
#             ok=np.logical_and(dist_r>=r_bins[ii], dist_r<r_bins[ii+1])
#             att = np.append(att, np.sum(lum[ok])/np.sum(lum_int[ok]))
#
#         plt.plot(r_bins_cen/rhalf[jj], -2.5*np.log10(att))
#
# # plt.scatter(dist_r/rhalf[jj], lum/lum_int)
#
# plt.show()


func = partial(get_half_lr, S_mass=S_mass, S_Z=S_Z, G_Z=G_Z, S_age=S_age, S_los=S_los, S_coords=S_coords, S_sml=S_sml, begin=begin, end=end, z=z)

pool = schwimmbad.MultiPool(processes=8)
dat = np.array(list(pool.map(func, req_ind)))
pool.close()

hlr, lum, lum_intrinsic =  dat[:,0], dat[:,1], dat[:,2]

R_half = np.asarray(dat[:,0], dtype=np.float64)


data_folder = 'data'
filename = './{}/FLARES_{}_sp_info.hdf5'.format(data_folder,num)
sim_type = 'FLARES'

print(F"Wrting out to {filename}")

fl = flares.flares(fname = filename,sim_type = sim_type)

fl.create_group('{}'.format(tag))
fl.create_group('{}/Luminosity'.format(tag))

fl.create_dataset(R_half, 'R_half', '{}'.format(tag),
    desc = 'Half mass radius for galaxies with number of star particles >= 1000', unit = 'pkpc')
for ii in range(len(lum)):
    fl.create_dataset(np.asarray(lum[ii], dtype=np.float64), F'{ii}/FUV', '{}/Luminosity'.format(tag),
    desc = 'FUV luminosities of star particles in galaxies with number of star particles >= 1000', unit = 'ergs/s/Hz')
    fl.create_dataset(np.asarray(lum_intrinsic[ii], dtype=np.float64), F'{ii}/FUV_intrinsic', '{}/Luminosity'.format(tag),
    desc = 'FUV luminosities of star particles in galaxies with number of star particles >= 1000', unit = 'ergs/s/Hz')
