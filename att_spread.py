import timeit
import numpy as np
import pandas as pd
import healpy as hp
import h5py
from functools import partial
import schwimmbad
import matplotlib
import matplotlib.pyplot as plt

from phot_modules import DTM_fit
from helpers import lum_from_stars


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
        cop = np.array(hf[num+tag+'/Galaxy'].get('COP'), dtype = np.float64)
        mstar = np.array(hf[num+tag+'/Galaxy'].get('Mstar_30'), dtype = np.float64)*1e10
        cop_vel = np.array(hf[num+tag+'/Galaxy'].get('Velocity'), dtype = np.float64)
        L_FUV = np.array(hf[num+tag+'/Galaxy'].get('BPASS_2.2.1/Chabrier300/Luminosity/DustModelI/FUV'), dtype = np.float64)
        L_FUV_int = np.array(hf[num+tag+'/Galaxy'].get('BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic/FUV'), dtype = np.float64)
        S_len = np.array(hf[num+tag+'/Galaxy'].get('S_Length'), dtype = np.int64)
        G_len = np.array(hf[num+tag+'/Galaxy'].get('G_Length'), dtype = np.int64)
        S_mass = np.array(hf[num+tag+'/Particle'].get('S_MassInitial'), dtype = np.float64)*1e10
        G_mass = np.array(hf[num+tag+'/Particle'].get('G_Mass'), dtype = np.float64)*1e10
        S_coords = np.array(hf[num+tag+'/Particle'].get('S_Coordinates'), dtype = np.float64)
        G_coords = np.array(hf[num+tag+'/Particle'].get('G_Coordinates'), dtype = np.float64)
        S_mass_curr = np.array(hf[num+tag+'/Particle'].get('S_Mass'), dtype = np.float64)*1e10
        S_vel = np.array(hf[num+tag+'/Particle'].get('S_Vel'), dtype = np.float64)
        G_vel = np.array(hf[num+tag+'/Particle'].get('G_Vel'), dtype = np.float64)
        S_Z =  np.array(hf[num+tag+'/Particle'].get('S_Z_smooth'), dtype = np.float64)
        S_age = np.array(hf[num+tag+'/Particle'].get('S_Age'), dtype = np.float64)*1e3
        S_los = np.array(hf[num+tag+'/Particle'].get('S_los'), dtype = np.float64)
        G_Z = np.array(hf[num+tag+'/Particle'].get('G_Z_smooth'), dtype = np.float64)

    begin = np.zeros(len(S_len), dtype = np.int64)
    end = np.zeros(len(S_len), dtype = np.int64)
    begin[1:] = np.cumsum(S_len)[:-1]
    end = np.cumsum(S_len)

    gbegin = np.zeros(len(G_len), dtype = np.int64)
    gend = np.zeros(len(G_len), dtype = np.int64)
    gbegin[1:] = np.cumsum(G_len)[:-1]
    gend = np.cumsum(G_len)


    return cop, mstar, cop_vel, L_FUV, L_FUV_int, S_coords, G_coords, S_mass, S_mass_curr, G_mass, S_Z, S_age, S_los, S_vel, G_vel, G_Z, begin, end, gbegin, gend

def get_theta(phi_vals, normal):

    x, y, z = normal

    theta_vals = - np.arctan2(z, x*np.cos(phi_vals) + y*np.sin(phi_vals))
    theta_vals[theta_vals<0]+=np.pi

    return theta_vals


tag='010_z005p000'
sim_type='FLARES'
ii=0


num = str(ii)
if len(num) == 1:
    num = '0'+num

cop, mstar, cop_vel, L_FUV, L_FUV_int, S_coords, G_coords, S_mass, S_mass_curr, G_mass, S_Z, S_age, S_los, S_vel, G_vel, G_Z, begin, end, gbegin, gend = get_data(num, tag, inp = 'FLARES')
z = float(tag[5:].replace('p','.'))
cop = cop/(1+z)
S_coords/=(1+z)
G_coords/=(1+z)

ATT = -2.5*np.log10(L_FUV/L_FUV_int)
FESC = L_FUV/L_FUV_int

req_ind = np.where(mstar>10**9.5)[0]
att_16s, att_84s = np.zeros(len(req_ind)), np.zeros(len(req_ind))

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(5, 5), sharex=True, sharey=True, facecolor='w', edgecolor='k')
norm = matplotlib.colors.Normalize(vmin=0., vmax=1)
# choose a colormap
c_m = matplotlib.cm.coolwarm
# create a ScalarMappable and initialize a data structure
s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
s_m.set_array([])


for kk, jj in enumerate(req_ind):
    print (F"Computing attenuations for task {kk+1}/{len(req_ind)}")
    #Coordinates and attributes for the jj galaxy in ii region

    this_smass = S_mass[begin[jj]:end[jj]]

    this_sZ = S_Z[begin[jj]:end[jj]]
    this_gZ = G_Z[gbegin[jj]:gend[jj]]

    this_age = S_age[begin[jj]:end[jj]]
    this_los = S_los[begin[jj]:end[jj]]

    Mage = np.nansum(this_smass*this_age)/np.nansum(this_smass)
    Z = np.nanmean(this_gZ)
    DTM = DTM_fit(Z, Mage)

    this_ATT = np.array([ATT[jj]])
    this_mstar = np.array([np.log10(mstar[jj])])

    # with h5py.File(F'data/Zlos_inclination_{num}.hdf5', 'r') as hf:
    #     Zlos = np.array(hf[tag].get(F'S_los_{jj}'), dtype = np.float64)
    #
    #
    # calc_lum = partial(lum_from_stars, Masses=this_smass, Ages=this_age, Metallicities=this_sZ, DTM=DTM, filters = ['FAKE.TH.FUV'])
    # pool = schwimmbad.MultiPool(processes=28)
    # dat = np.array(list(pool.map(calc_lum, Zlos)))
    # pool.close()

    lum = lum_from_stars(MetSurfaceDensities = this_los, Masses=this_smass, Ages=this_age, Metallicities=this_sZ, DTM=DTM, filters = ['FAKE.TH.FUV'])
    lum_int = lum_from_stars(MetSurfaceDensities = np.zeros(len(this_smass)), Masses=this_smass, Ages=this_age, Metallicities=this_sZ, DTM=0, filters = ['FAKE.TH.FUV'])
    # att = -2.5*np.log10(lum[0]/lum_int[0])
    #
    # att_84 = np.percentile(att, 84)
    # att_84s[kk] = att_84
    # att_16 = np.percentile(att, 16)
    # att_16s[kk] = att_16
    fesc = lum[0]/lum_int[0]


    ax.scatter(this_mstar, np.max(fesc)-np.min(fesc), s=10, marker='o', color=s_m.to_rgba(FESC[jj]))


ax.set_xlabel(r'log$_{10}$(M$_{\star}$/M$_{\odot}$)', fontsize=12)
ax.set_ylabel(r'f$_{\mathrm{esc, max-min}}$', fontsize=12)
#ax.set_ylim(0,10)
ax.grid()

fig.subplots_adjust(right=0.9, wspace=0, hspace=0)
cbaxes = fig.add_axes([0.91, 0.2, 0.015, 0.6])
fig.colorbar(s_m, cax=cbaxes)
cbaxes.set_ylabel(r'f$_{\mathrm{esc}}$ (Galaxy)', fontsize = 12)
for label in (cbaxes.get_yticklabels()):
    label.set_fontsize(10)

plt.savefig(F'fesc_spread_min_max_reg{ii}.png', dpi = 300, bbox_inches='tight')
plt.show()
