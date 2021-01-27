import timeit
import numpy as np
import pandas as pd
import healpy as hp
import h5py
from functools import partial
import schwimmbad
import matplotlib
import matplotlib.pyplot as plt


from helpers import lum, ang_mom_vector, get_spherical_from_cartesian, get_cartesian_from_spherical, kappa, calc_axes


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

# Generate different viewing angles
nside=8
hp_theta, hp_phi = hp.pix2ang(nside, range(hp.nside2npix(nside)))
hp_cartesian = np.array([get_cartesian_from_spherical([hp_theta[ii], hp_phi[ii]]) for ii in range(len(hp_theta))])


num = str(ii)
if len(num) == 1:
    num = '0'+num

cop, mstar, cop_vel, S_coords, G_coords, S_mass, S_mass_curr, G_mass, S_Z, S_age, S_vel, G_vel, G_Z, begin, end, gbegin, gend = get_data(num, tag, inp = 'FLARES')
z = float(tag[5:].replace('p','.'))
cop = cop/(1+z)
S_coords/=(1+z)
G_coords/=(1+z)

req_ind = np.where(mstar>10**9.5)[0]

with h5py.File('../flares_pipeline/data/flares.hdf5','r') as hf:

    L_FUV_int = np.array(hf[F"{num}/{tag}/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic/FUV"])


# norm = matplotlib.colors.Normalize(vmin=0., vmax=1.)
# # choose a colormap
# c_m = matplotlib.cm.coolwarm
# # create a ScalarMappable and initialize a data structure
# s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
# s_m.set_array([])
#
# df = pd.DataFrame({})
# colors = np.array([])
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
    this_age = S_age[begin[jj]:end[jj]]

    Kco, Krot = kappa(this_smass_curr, scoords, this_svel)
    ang_vector_stars = ang_mom_vector(this_smass_curr, scoords, this_svel)
    ang_vector_gas = ang_mom_vector(this_gmass, gcoords, this_gvel)

    angle_g_los = np.arccos((hp_cartesian@ang_vector_gas.T).T)/np.pi
    angle_s_los = np.arccos((hp_cartesian@ang_vector_stars.T).T)/np.pi
    angle_g_s = np.arccos(np.dot(ang_vector_stars, ang_vector_gas))/np.pi

    s_axis, s_ev = calc_axes(scoords)
    g_axis, g_ev = calc_axes(gcoords)

    s_s, s_q, s_p = s_axis[2]/s_axis[0], s_axis[1]/s_axis[0], s_axis[2]/s_axis[1]
    g_s, g_q, g_p = g_axis[2]/g_axis[0], g_axis[1]/g_axis[0], g_axis[2]/g_axis[1]

    # L_FUV_int = lum(MetSurfaceDensities=np.zeros(len(this_smass)),Masses=this_smass, Ages=this_age, Metallicities=this_sZ, DTM=0., Type='Intrinsic')

    with h5py.File(F'data/Zlos_inclination_{num}.hdf5', 'r') as hf:
        LFUV = np.array(hf[F"{tag}/Luminosity"].get(F'LFUV_{jj}'), dtype = np.float64)
        # LNUV = np.array(hf[F"{tag}/Luminosity"].get(F'LNUV_{jj}'), dtype = np.float64)

    att = -2.5*np.log10(LFUV/L_FUV_int[jj])

    #x = np.ones(len(att))*Kco


    if Kco>=0.4:
        fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(7, 4), sharex=True, sharey=True, facecolor='w', edgecolor='k')
        axs=axs.ravel()

        axs[0].scatter(angle_g_los, att, s=1)
        axs[1].scatter(angle_s_los, att, s=1)

        fig.suptitle(r'$log_{10}(M_{*}/M_{\odot})=%0.3f,\,log_{10}(L_{FUV\,int}/ergs/s/Hz)=%0.3f,\,\kappa_{co}=%0.3f,\,\mathrm{Angle}_{g.s}/\pi=%0.3f, \,\mathrm{stars:(s,q,p)}=[%0.3f,%0.3f,%0.3f],\,\mathrm{gas:(s,q,p)}=[%0.3f,%0.3f,%0.3f]$'%(np.log10(mstar[jj]), np.log10(L_FUV_int[jj]), Kco, angle_g_s, s_s, s_q, s_p, g_s, g_q, g_p), fontsize=9)

        axs[0].set_ylabel(r'$A_{FUV}$', fontsize=12)
        axs[0].set_xlabel(r'$Angle_{g.los}/\pi$', fontsize=12)
        axs[1].set_xlabel(r'$Angle_{s.los}/\pi$', fontsize=12)
        for ll in [0,1]:
            axs[ll].grid(True)
        plt.savefig(F'plots/att_angle_los_anglegs{num}_{jj}.png', dpi = 300, bbox_inches='tight')

        plt.close()



# ax.set_xlabel(r'$A_{FUV}$', fontsize=12)
# ax.set_ylabel(r'$Angle_{g.los}/\pi$', fontsize=12)
# # ax.legend(fontsize=12)
#
# fig.subplots_adjust(right=0.9, wspace=0, hspace=0)
# cbaxes = fig.add_axes([0.91, 0.2, 0.015, 0.6])
# fig.colorbar(s_m, cax=cbaxes)
# cbaxes.set_ylabel(r'$Angle_{g.s}$', fontsize = 12)
# for label in (cbaxes.get_yticklabels()):
#     label.set_fontsize(10)


#plt.savefig(F'att_angle_glos_anglegs{num}.png', dpi = 300, bbox_inches='tight')

# plt.show()
