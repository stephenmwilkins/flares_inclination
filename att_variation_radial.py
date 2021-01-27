import timeit, sys
import numpy as np
import pandas as pd
import healpy as hp
import h5py
from functools import partial
import schwimmbad
import matplotlib
import matplotlib.pyplot as plt

from FLARE.photom import lum_to_M, M_to_lum



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
        lfuv = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI'].get('FUV'), dtype = np.float64)
        lfuv_int = np.array(hf[num+tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic'].get('FUV'), dtype = np.float64)

    begin = np.zeros(len(S_len), dtype = np.int64)
    end = np.zeros(len(S_len), dtype = np.int64)
    begin[1:] = np.cumsum(S_len)[:-1]
    end = np.cumsum(S_len)

    gbegin = np.zeros(len(G_len), dtype = np.int64)
    gend = np.zeros(len(G_len), dtype = np.int64)
    gbegin[1:] = np.cumsum(G_len)[:-1]
    gend = np.cumsum(G_len)


    return cop, mstar, S_coords, G_coords, S_mass, G_mass, S_Z, S_age, S_los, S_sml, G_sml, G_Z, begin, end, gbegin, gend, S_len, lfuv, lfuv_int

def get_att(num, tag, r_bins, plt_type, inp='FLARES'):


    r_cen = (r_bins[1:]+r_bins[:-1])/2.
    atts = r_cen
    z = float(tag[5:].replace('p','.'))

    num = str(num)
    if len(num) == 1:
        num = '0'+num

    try:
        cop, mstar, S_coords, G_coords, S_mass, G_mass, S_Z, S_age, S_los, S_sml, G_sml, G_Z, begin, end, gbegin, gend, S_len, lfuv, lfuv_int = get_data(num, tag, inp = 'FLARES')

        req_ind = np.where(S_len>=1000)[0]
        mstar = mstar[req_ind]
        lfuv = lfuv[req_ind]
        lfuv_int = lfuv_int[req_ind]
        att_gal = -2.5*np.log10(lfuv/lfuv_int)

        lums, lum_ints = np.array([]), np.array([])

        for jj, kk in enumerate(req_ind):

            with h5py.File(F"./data/FLARES_{num}_sp_info.hdf5", 'r') as hf:
                rhalf = np.array(hf[tag].get('R_half'), dtype = np.float64)
                lum = np.array(hf[tag+F'/Luminosity/{jj}'].get('FUV'))
                lum_int = np.array(hf[tag+F'/Luminosity/{jj}'].get('FUV_intrinsic'))

            lums = np.append(lums, np.sum(lum))
            lum_ints = np.append(lum_ints, np.sum(lum_int))

            this_scoord = (S_coords[:, begin[kk]:end[kk]].T - cop[:,kk])*1e3/(1+z)
            dist_r = np.linalg.norm(this_scoord, axis=1)
            r_bins_this = r_bins*rhalf[jj]
            r_bins_cen = (r_bins_this[1:]+r_bins_this[:-1])/2.

            att = np.array([])
            for ii in range(len(r_bins_this)-1):
                if plt_type == 0:
                    tmp = r_bins_this[ii]
                else:
                    tmp = 0.
                ok=np.logical_and(dist_r>=tmp, dist_r<r_bins_this[ii+1])
                att = np.append(att, np.sum(lum[ok])/np.sum(lum_int[ok]))

            atts = np.vstack([atts, att])
        atts = atts[1:]

    except:
        print (F"No data available in {num}")

        atts, mstar, lfuv, att_gal = np.array([]), np.array([]), np.array([]), np.array([])

    #return atts, mstar, lum_to_M(lfuv), att_gal
    return atts, mstar, lfuv, lfuv_int, att_gal, lums, lum_ints


tag='010_z005p000'
z = float(tag[5:].replace('p','.'))
sim_type='FLARES'

plt_type = int(sys.argv[1])
if plt_type==0:
    ylabel = r"A$_{\mathrm{FUV}}$($<$r)"
    savename = 'att_radial_var_aperture.png'
else:
    ylabel = r"r$_1 \le A$_{\mathrm{FUV}}<$r$_2$"
    savename = 'att_radial_var_annuli.png'

r_bins = np.arange(0, 5, 0.25)
r_cen = (r_bins[1:]+r_bins[:-1])/2.


func = partial(get_att, tag=tag, r_bins=r_bins, plt_type=plt_type)
pool = schwimmbad.MultiPool(processes=8)
dat = np.array(list(pool.map(func, np.arange(0,40))))
pool.close()

for ii in range(40):
    att, mstar, lfuv, att_gal = dat[ii][0], dat[ii][1], dat[ii][2], dat[ii][4]

    if len(mstar)>0:
        if ii>0:
            atts = np.append(atts, att, axis=0)
            mstars = np.append(mstars, mstar)
            lfuvs = np.append(lfuvs, lfuv)
            att_gals = np.append(att_gals, att_gal)
        else:
            atts, mstars, lfuvs, att_gals = att, mstar, lfuv, att_gal



mstars = np.log10(mstars)
mbins = np.arange(9., 11.75, 0.5)
mbins_label = np.array([F'${mbins[ii]}$ - ${mbins[ii+1]}$' for ii in range(len(mbins)-1)])
lbins = -np.arange(19.5, 25, 1)[::-1]
lbins_label = np.array([F'${lbins[ii+1]}$ - ${lbins[ii]}$' for ii in range(len(lbins)-1)])

atts = -2.5*np.log10(atts)
lfuvs = lum_to_M(lfuvs)


fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(10, 5), sharex=True, sharey=True, facecolor='w', edgecolor='k')
axs = axs.ravel()

colors = ['red', 'brown', 'magenta', 'orange', 'indigo']

for ii in range(len(mbins)-1):
    ok = np.logical_and(mstars>=mbins[ii], mstars<mbins[ii+1])
    if np.sum(ok)>=1:

        axs[0].plot(r_cen, np.nanmedian(atts[ok], axis=0), label=rF'{mbins_label[ii]} (N={np.sum(ok)})', color=colors[::-1][ii])
        axs[0].axhline(y=np.nanmedian(att_gals[ok]), color=colors[::-1][ii], ls='dashed')
        # axs[0].plot(r_cen, np.nanpercentile(atts[ok], 84, axis=0)-np.nanpercentile(atts[ok], 16, axis=0), label=rF'{mbins_label[ii]} (N={np.sum(ok)})', color=colors[::-1][ii])

for ii in range(len(lbins)-1):
    ok = np.logical_and(lfuvs>=lbins[ii], lfuvs<lbins[ii+1])
    if np.sum(ok)>=1:

        axs[1].plot(r_cen, np.nanmedian(atts[ok], axis=0), label=rF'{lbins_label[ii]} (N={np.sum(ok)})', color=colors[ii])
        axs[1].axhline(y=np.nanmedian(att_gals[ok]), color=colors[ii], ls='dashed')
        # axs[1].plot(r_cen, np.nanpercentile(atts[ok], 84, axis=0)-np.nanpercentile(atts[ok], 16, axis=0), label=rF'{lbins_label[ii]} (N={np.sum(ok)})', color=colors[ii])


for ii in [0,1]:
    axs[ii].grid()
    #axs[ii].set_ylim(0,20)
    axs[ii].set_xlim(left=0)
    axs[ii].set_xlabel(r"r/r$_{1/2}$", fontsize=12)


axs[0].legend(title=r'Mass range (log$_{10}$)', fontsize=10)
# axs[1].legend(title=r'M$_{\mathrm{FUV}}$ range', fontsize=10)
handles, labels = axs[1].get_legend_handles_labels()
axs[1].legend(handles[::-1], labels[::-1], title=r'M$_{\mathrm{FUV}}$ range', fontsize=10)

axs[0].set_ylabel(ylabel, fontsize=12)
#axs[0].set_ylabel(r"A$_{\mathrm{FUV,84}}$ $-$ A$_{\mathrm{FUV,16}}$", fontsize=12)

fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig(savename, bbox_inches='tight', dpi=300)
plt.show()
