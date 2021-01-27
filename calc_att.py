import timeit
import numpy as np
import healpy as hp
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

from helpers import lum, ang_mom_vector, get_spherical_from_cartesian, kappa
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
        cop = np.array(hf[num+tag+'/Galaxy'].get('COP'), dtype = np.float64)
        mstar = np.array(hf[num+tag+'/Galaxy'].get('Mstar_30'), dtype = np.float64)*1e10
        cop_vel = np.array(hf[num+tag+'/Galaxy'].get('Velocity'), dtype = np.float64)
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
        G_Z = np.array(hf[num+tag+'/Particle'].get('G_Z_smooth'), dtype = np.float64)

    begin = np.zeros(len(S_len), dtype = np.int64)
    end = np.zeros(len(S_len), dtype = np.int64)
    begin[1:] = np.cumsum(S_len)[:-1]
    end = np.cumsum(S_len)

    gbegin = np.zeros(len(G_len), dtype = np.int64)
    gend = np.zeros(len(G_len), dtype = np.int64)
    gbegin[1:] = np.cumsum(G_len)[:-1]
    gend = np.cumsum(G_len)


    return cop, mstar, cop_vel, S_coords, G_coords, S_mass, S_mass_curr, G_mass, S_Z, S_age, S_vel, G_vel, G_Z, begin, end, gbegin, gend


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

#viewing angles
nside=8
hp_theta, hp_phi = hp.pix2ang(nside, range(hp.nside2npix(nside)))
angles = np.vstack([hp_theta, hp_phi]).T

pixel_indices = hp.ang2pix(nside, hp_theta, hp_phi)
m = np.zeros(hp.nside2npix(nside))


cop, mstar, cop_vel, S_coords, G_coords, S_mass, S_mass_curr, G_mass, S_Z, S_age, S_vel, G_vel, G_Z, begin, end, gbegin, gend = get_data(num, tag, inp = 'FLARES')
z = float(tag[5:].replace('p','.'))
cop = cop/(1+z)
S_coords/=(1+z)
G_coords/=(1+z)

req_ind = np.where(mstar>10**9.5)[0]

#jj = req_ind[4]
for kk, jj in enumerate(req_ind):
    print (F"Computing attenuations for task {kk}/{len(req_ind)}")
    #Coordinates and attributes for the jj galaxy in ii region
    scoords = S_coords[:, begin[jj]:end[jj]].T - cop[:,jj]
    gcoords = G_coords[:, gbegin[jj]:gend[jj]].T - cop[:,jj]

    this_smass = S_mass[begin[jj]:end[jj]]
    this_smass_curr = S_mass_curr[begin[jj]:end[jj]]
    this_svel = S_vel[begin[jj]:end[jj]] - cop_vel[:,jj]

    this_gmass = G_mass[gbegin[jj]:gend[jj]]
    this_gvel = G_vel[gbegin[jj]:gend[jj]] - cop_vel[:,jj]

    this_sZ = S_Z[begin[jj]:end[jj]]
    this_gZ = G_Z[gbegin[jj]:gend[jj]]

    this_age = S_age[begin[jj]:end[jj]]

    Mage = np.nansum(this_smass*this_age)/np.nansum(this_smass)
    Z = np.nanmean(this_gZ)
    DTM = DTM_fit(Z, Mage)

    Kco, Krot = kappa(this_smass_curr, scoords, this_svel)
    ang_vector_stars = ang_mom_vector(this_smass_curr, scoords, this_svel)
    ang_vector_sp_stars = get_spherical_from_cartesian(ang_vector_stars)

    ang_vector_gas = ang_mom_vector(this_gmass, gcoords, this_gvel)
    ang_vector_sp_gas = get_spherical_from_cartesian(ang_vector_gas)

    print (F'Kco = {Kco}, Krot = {Krot}')

    with h5py.File(F'data/Zlos_inclination_{num}.hdf5', 'r') as hf:
        Zlos = np.array(hf[tag].get(F'S_los_{jj}'), dtype = np.float64)


    calc_lum = partial(lum, Masses=this_smass, Ages=this_age, Metallicities=this_sZ, DTM=DTM)
    pool = schwimmbad.MultiPool(processes=28)
    dat = np.array(list(pool.map(calc_lum, Zlos)))
    pool.close()

    L_FUV_int = lum(MetSurfaceDensities=np.zeros(len(this_smass)),Masses=this_smass, Ages=this_age, Metallicities=this_sZ, DTM=0., Type='Intrinsic')

    att = -2.5*np.log10(dat/L_FUV_int)

    #fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(5, 5), sharex=True, sharey=True, facecolor='w', edgecolor='k')

    # ax.scatter(hp_phi/np.pi,att, s=4)
    # ax.set_ylabel(r'A$_{\mathrm{FUV}}$=-2.5 log$_{10}$(L$_{\mathrm{FUV}}^{\mathrm{Observed}}$/L$_{\mathrm{FUV}}^{\mathrm{Intrinsic}}$)', fontsize=12)
    # ax.set_xlabel(r'$\phi\,/\pi$', fontsize=14)
    # plt.savefig('test_0_0_phi.png', dpi = 300, bbox_inches='tight')

    pixel_indices = hp.ang2pix(nside, hp_theta, hp_phi)
    m = np.zeros(hp.nside2npix(nside))
    m[pixel_indices] = att
    hp.mollview(m, cmap=matplotlib.cm.coolwarm, title="Kco=%0.3f, Krot=%0.3f"%(Kco, Krot))

    phi = np.linspace(0, 2*np.pi,10000)

    theta = get_theta(phi, ang_vector_stars)
    hp.projplot(theta, phi, ls='solid', color='orange')
    hp.projscatter(ang_vector_sp_stars[1], ang_vector_sp_stars[2], marker='X', color='orange')

    theta = get_theta(phi, ang_vector_gas)
    hp.projplot(theta, phi, ls='solid', color='blue')
    hp.projscatter(ang_vector_sp_gas[1], ang_vector_sp_gas[2], marker='X', color='blue')


    plt.savefig(F'mollview_plots_0/test_{jj}_mollview.png', dpi = 300, bbox_inches='tight')

    plt.close()
