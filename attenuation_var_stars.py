import timeit, sys
import numpy as np
import pandas as pd
import healpy as hp
import h5py
from functools import partial
import schwimmbad
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from FLARE.photom import lum_to_M, M_to_lum


def get_data(ii, tag, inp = 'FLARES'):
    try:
        num = str(ii)
        if len(num) == 1:
            num =  '0'+num

        sim = rF"../flares_pipeline/data/FLARES_{num}_sp_info.hdf5"

        with h5py.File(sim, 'r') as hf:
            mstar   = np.array(hf[tag+'/Galaxy'].get('Mstar_30'), dtype = np.float64)*1e10
            lfuv    = np.array(hf[tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI'].get('FUV'), dtype = np.float64)
            lfuv_int    = np.array(hf[tag+'/Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic'].get('FUV'), dtype = np.float64)

            S_len = np.array(hf[tag+'/Galaxy'].get('S_Length'), dtype = np.int64)


        req_ind     = np.where(S_len>=1000)[0]
        mstar       = mstar[req_ind]
        lfuv        = lfuv[req_ind]
        lfuv_int    = lfuv_int[req_ind]
        S_len       = S_len[req_ind]

        begin       = np.zeros(len(S_len), dtype = np.int64)
        end         = np.zeros(len(S_len), dtype = np.int64)
        begin[1:]   = np.cumsum(S_len)[:-1]
        end         = np.cumsum(S_len)

        with h5py.File(F"./data/FLARES_{num}_sp_info.hdf5", 'r') as hf:
            for jj in range(len(req_ind)):
                if jj==0:
                    lum     = np.array(hf[tag+F'/Luminosity/{jj}'].get('FUV'))
                    lum_int = np.array(hf[tag+F'/Luminosity/{jj}'].get('FUV_intrinsic'))
                else:
                    lum     = np.append(lum, np.array(hf[tag+F'/Luminosity/{jj}'].get('FUV')))
                    lum_int = np.append(lum_int, np.array(hf[tag+F'/Luminosity/{jj}'].get('FUV_intrinsic')))

        att = lum/lum_int


        return mstar, lfuv, lfuv_int, att, begin, end, S_len

    except:
        print (F"No data available in {ii}")

        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

tag='010_z005p000'
z = float(tag[5:].replace('p','.'))
sim_type='FLARES'

inp = int(sys.argv[1])

# mstar, lfuv, lfuv_int, att, begin, end, slen = get_data(0, tag, inp = 'FLARES')

func = partial(get_data, tag=tag, inp = 'FLARES')
pool = schwimmbad.MultiPool(processes=8)
dat = np.array(list(pool.map(func, np.arange(0,40))))
pool.close()

for ii in range(40):
    if ii==0:
        mstar       = dat[ii][0]
        lfuv        = dat[ii][1]
        lfuv_int    = dat[ii][2]
        slen        = dat[ii][-1]
        num         = np.append(np.array([0]), len(dat[ii][0]))
        part_num    = np.arange(0, len(dat[ii][0]))
    else:
        mstar       = np.append(mstar, dat[ii][0])
        lfuv        = np.append(lfuv, dat[ii][1])
        lfuv_int    = np.append(lfuv_int, dat[ii][2])
        slen        = np.append(slen, dat[ii][-1])
        num         = np.append(num, num[-1]+len(dat[ii][0]))
        part_num    = np.append(part_num, np.arange(0, len(dat[ii][0])))


slen = np.asarray(slen, dtype=np.int)

violin_att = np.ones((len(mstar), np.max(slen)))*np.nan
for ii in range(len(mstar)):
    jj = np.where(ii>=num)[0][-1]
    begin   = dat[jj][4][part_num[ii]]
    end     = dat[jj][5][part_num[ii]]
    violin_att[ii][:slen[ii]] = dat[jj][3][begin:end]

att_gal = -2.5*np.log10(lfuv/lfuv_int)
lfuv    = lum_to_M(lfuv)
mstars = np.log10(mstar)

mbins = np.arange(9., 11.75, 0.5)
mbins_label = np.array([F'${mbins[ii]}$ - ${mbins[ii+1]}$' for ii in range(len(mbins)-1)])
lbins = -np.arange(19, 25, 1)[::-1]
lbins_label = np.array([F'${lbins[ii]}$ - ${lbins[ii+1]}$' for ii in range(len(lbins)-1)])


if inp == 0:
    fig, axs = plt.subplots(nrows = 2, ncols = 5, figsize=(11, 5), sharex=True, sharey=True, facecolor='w', edgecolor='k')
    axs = axs.ravel()


    for ii in range(len(mbins)-1):
        ok = np.logical_and(mstars>=mbins[ii], mstars<mbins[ii+1])
        if np.sum(ok)>=1:

            tmp = violin_att[ok].flatten()
            finite = np.isfinite(tmp)
            axs[ii].hist(tmp[finite], bins=np.arange(0,1.1,0.1), density=True, label=rF'{mbins_label[ii]} (N={np.sum(ok)})')

            axs[ii].grid()
            axs[ii].set_xticks([0,1])
            axs[ii].legend(title=r'Mass range (log$_{10}$)', frameon=False, fontsize=8)

    for ii in range(len(lbins)-1):
        ok = np.logical_and(lfuv>=lbins[ii], lfuv<lbins[ii+1])
        if np.sum(ok)>=1:

            tmp = violin_att[ok].flatten()
            finite = np.isfinite(tmp)
            axs[ii+5].hist(tmp[finite], bins=np.arange(0,1.1,0.1), density=True, label=rF'{lbins_label[ii]} (N={np.sum(ok)})')

            axs[ii+5].grid()
            axs[ii+5].set_xticks([0,1])
            axs[ii+5].legend(title=r'M$_{\mathrm{FUV}}$ range', frameon=False, fontsize=8)

    axs[0].set_ylabel('PDF', fontsize=10)
    axs[5].set_ylabel('PDF', fontsize=10)
    axs[7].set_xlabel('f$_{\mathrm{esc}}$', fontsize=10)

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(F'fesc_distr_mass_fuv_z{z}.png', bbox_inches='tight', dpi=300)
    plt.show()

elif inp == 1:


    norm = matplotlib.colors.Normalize(vmin=0., vmax=4.5)
    # choose a colormap
    c_m = matplotlib.cm.coolwarm
    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)



    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(7, 4), sharex=False, sharey=True, facecolor='w', edgecolor='k')
    axs = axs.ravel()

    y = np.nanmedian(violin_att, axis=1)
    y_up = np.nanpercentile(violin_att, 84, axis=1)
    y_lo = np.nanpercentile(violin_att, 16, axis=1)

    # axs[0].errorbar(mstars, y, yerr=[y_lo, y_up], ls='None')
    axs[0].scatter(mstars, y_up-y_lo, c=att_gal, s=5, cmap=c_m)

    # axs[1].errorbar(lfuv, y, yerr=[y_lo, y_up], ls='None')
    axs[1].scatter(lfuv, y_up-y_lo, c=att_gal, s=5, cmap=c_m)

    fig.subplots_adjust(right=0.92, wspace=0, hspace=0)

    cbaxes = fig.add_axes([0.95, 0.3, 0.015, 0.4])
    fig.colorbar(s_m, cax=cbaxes, orientation='vertical')
    cbaxes.set_xlabel(r'A$_{\mathrm{FUV}}$')
    axs[0].set_xlim(9,11.8)
    axs[1].set_xlim(-18,-24.5)

    for ii in [0,1]:
        axs[ii].grid()

    axs[0].set_xlabel(r"log$_{10}$(M$_{\star}$/M$_{\odot}$)", fontsize=11)
    axs[1].set_xlabel(r"M$_{\mathrm{FUV}}$", fontsize=11)

    axs[0].set_ylabel(r"f$_{\mathrm{esc,84}}$ $-$ f$_{\mathrm{esc,16}}$", fontsize=11)

    plt.savefig(F'fesc_precentile_diff_mass_fuv_z{z}.png', bbox_inches='tight', dpi=300)
    plt.show()
