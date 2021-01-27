import timeit, sys
import numpy as np
import healpy as hp
import h5py
from functools import partial
import schwimmbad
import matplotlib.pyplot as plt
from numba import jit
from astropy.cosmology import Planck13 as cosmo
from astropy import units as u

from helpers import get_Z_LOS, get_rotation_matrix, get_spherical_from_cartesian, get_cartesian_from_spherical
sys.path.append('/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline')
import flares

conv = (u.solMass/u.Mpc**2).to(u.solMass/u.pc**2)



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
        S_len = np.array(hf[num+tag+'/Galaxy'].get('S_Length'), dtype = np.int64)
        G_len = np.array(hf[num+tag+'/Galaxy'].get('G_Length'), dtype = np.int64)
        S_coords = np.array(hf[num+tag+'/Particle'].get('S_Coordinates'), dtype = np.float64)
        G_coords = np.array(hf[num+tag+'/Particle'].get('G_Coordinates'), dtype = np.float64)
        S_mass = np.array(hf[num+tag+'/Particle'].get('S_MassInitial'), dtype = np.float64)*1e10
        S_Z =  np.array(hf[num+tag+'/Particle'].get('S_Z_smooth'), dtype = np.float64)
        S_age = np.array(hf[num+tag+'/Particle'].get('S_Age'), dtype = np.float64)*1e3
        G_mass = np.array(hf[num+tag+'/Particle'].get('G_Mass'), dtype = np.float64)*1e10
        G_sml = np.array(hf[num+tag+'/Particle'].get('G_sml'), dtype = np.float64)
        G_Z = np.array(hf[num+tag+'/Particle'].get('G_Z_smooth'), dtype = np.float64)

    begin = np.zeros(len(S_len), dtype = np.int64)
    end = np.zeros(len(S_len), dtype = np.int64)
    begin[1:] = np.cumsum(S_len)[:-1]
    end = np.cumsum(S_len)

    gbegin = np.zeros(len(G_len), dtype = np.int64)
    gend = np.zeros(len(G_len), dtype = np.int64)
    gbegin[1:] = np.cumsum(G_len)[:-1]
    gend = np.cumsum(G_len)


    return cop, mstar, S_coords, G_coords, S_mass, S_Z, S_age, G_mass, G_sml, G_Z, begin, end, gbegin, gend



def get_ZLOS(angle, scoords, gcoords, this_gmass, this_gZ, this_gsml, lkernel, kbins):

    vector = get_cartesian_from_spherical(angle)
    rot = get_rotation_matrix(vector)
    this_scoords = (rot @ scoords.T).T
    this_gcoords = (rot @ gcoords.T).T

    Z_los_SD = get_Z_LOS(this_scoords, this_gcoords, this_gmass, this_gZ, this_gsml, lkernel, kbins)*conv

    return Z_los_SD


if __name__ == "__main__":

    ii, tag, sim_type = sys.argv[1], sys.argv[2], sys.argv[3]

    # tag='010_z005p000'
    # sim_type='FLARES'

    #sph kernel approximations
    kinp = np.load('./data/kernel_sph-anarchy.npz', allow_pickle=True)
    lkernel = kinp['kernel']
    header = kinp['header']
    kbins = header.item()['bins']

    # Generate different viewing angles
    nside=8
    hp_theta, hp_phi = hp.pix2ang(nside, range(hp.nside2npix(nside)))
    angles = np.vstack([hp_theta, hp_phi]).T

    #For galaxies in region `num`
    num = str(ii)
    if len(num) == 1:
        num = '0'+num
    cop, mstar, S_coords, G_coords, S_mass, S_Z, S_age, G_mass, G_sml, G_Z, begin, end, gbegin, gend = get_data(num, tag, inp = 'FLARES')
    z = float(tag[5:].replace('p','.'))
    cop = cop/(1+z)
    S_coords/=(1+z)
    G_coords/=(1+z)

    req_ind = np.where(mstar>10**9.5)[0]
    print ("Number of selected galaxies = ", len(req_ind))
    filename = F'data/Zlos_inclination_{num}.hdf5'
    fl = flares.flares(fname = filename, sim_type = sim_type)
    fl.create_group(F'{tag}')

    for kk, jj in enumerate(req_ind):

        #Coordinates and attributes for the jj galaxy in ii region
        scoords = S_coords[:, begin[jj]:end[jj]].T - cop[:,jj]
        gcoords = G_coords[:, gbegin[jj]:gend[jj]].T - cop[:,jj]

        this_smass = S_mass[begin[jj]:end[jj]]
        this_gmass = G_mass[gbegin[jj]:gend[jj]]

        this_sZ = S_Z[begin[jj]:end[jj]]
        this_gZ = G_Z[gbegin[jj]:gend[jj]]

        this_age = S_age[begin[jj]:end[jj]]
        this_gsml = G_sml[gbegin[jj]:gend[jj]]


        start = timeit.default_timer()
        print (F"Computing Zlos's for task {kk}/{len(req_ind)}")
        calc_Zlos = partial(get_ZLOS, scoords=scoords, gcoords=gcoords, this_gmass=this_gmass, this_gZ=this_gZ, this_gsml=this_gsml, lkernel=lkernel, kbins=kbins)
        pool = schwimmbad.MultiPool(processes=16)
        Zlos = np.array(list(pool.map(calc_Zlos, angles)))
        pool.close()

        fl.create_dataset(Zlos, F'S_los_{jj}', F'{tag}',
                desc = F'Star particle line-of-sight metal column density along the z-axis for galaxy index {jj} for different viewing angles',
                unit = 'Msun/pc^2')

        stop = timeit.default_timer()
        print (F"Took {np.round(stop - start, 6)/60} minutes")
