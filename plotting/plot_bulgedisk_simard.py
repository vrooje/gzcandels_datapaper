import sys
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
#import pandas as pd
import astropy
from astropy.table import Table
from scipy import stats, interpolate, special


infile_CANDELS = '/Users/vrooje/Documents/Astro/Zooniverse/gz_reduction_sandbox/data/gz_and_megamorph_cosmos_withrestframeV.fits'

#infile_SDSS = '/Users/vrooje/Documents/Astro/GalaxyZoo/GZ2_paper/lackner2012_orig/gz2_table3_lackner_gunn_2012_withBtot.fits'
#infile_SDSS = '/Users/vrooje/Documents/Astro/GalaxyZoo/GZ_CANDELS/gzcandels_datapaper/plotting/simard_etal_SD_matched_to_gz2_tables3_4.fits'
infile_SDSS = '/Users/vrooje/Documents/Astro/GalaxyZoo/GZ_CANDELS/gzcandels_datapaper/plotting/gz2_BTfits_meert_simard_withabsmag.fits'

# column labels for the quantities of interest - SDSS
gMag = "ggMag"
redshift = "z"
btot = "__B_T_g"

# column labels for quantities of interest - CANDELS
# use the VIJH summed B/Tot
#gzbt = 'BTot_galfit_sum'
# use the rest-frame g B/Tot (which doesn't exist yet, so create it below)
gzbt = 'BTot_rest_g'
#gzbt = 'BTot_galfit_F606W_BD'
#gzbt = 'BTot_galfit_F814W_BD'
#gzbt = 'BTot_galfit_F125W_BD'
#gzbt = 'BTot_galfit_F160W_BD'


###########################################################





def get_dMg(q):
    try:
        return float(q) - Mstar_g
    except:
        return -99.



def plot_dM_bymorph():
    sdss_col = btot
    figM = plt.figure(figsize=(8,6))
    #thebins = np.arange(-4.35,5.55,.3)
    thebins = np.arange(-4.2,5.8,.4)

    # TOP LEFT
    axq = figM.add_subplot(2,2,1)

    axq.hist(gzmm['dM_V'][smooth_select],   histtype='step', color='red', label='Smooth', linestyle='dotted', linewidth=2., bins=thebins)
    axq.hist(gzmm['dM_V'][featured_select], weights=np.ones_like(gzmm['dM_V'][featured_select])*10., histtype='step', color='blue', label='Featured ($\\times 10$)', bins=thebins)

    ylim1 = (0., 460.)

    #axq.set_xlim(xlimits)
    axq.set_ylim(ylim1)
    axq.set_xlabel("CANDELS $M_V - M^*_V$")
    axq.legend(loc='upper left', frameon=False) #loc=2


    # TOP RIGHT
    axr = figM.add_subplot(2,2,2)

    axr.hist(sdss['dM_g'][sdss_smooth & np.invert(np.isnan(sdss[sdss_col]))],   histtype='step', color='red', label='Smooth', linestyle='dotted', linewidth=2., bins=thebins)
    axr.hist(sdss['dM_g'][sdss_featured & np.invert(np.isnan(sdss[sdss_col]))], histtype='step', color='blue', label='Featured', bins=thebins)

    ylim2 = (0., 270.)

    #axr.set_xlim(xlimits)
    axr.set_ylim(ylim2)
    axr.set_xlabel("SDSS $M_g - M^*_g$")
    axr.legend(loc='upper left', frameon=False) #loc=2




    # BOTTOM LEFT
    axs = figM.add_subplot(2,2,3)

    axs.hist(gzmm['dM_V'][smooth_select],   weights=gzmm['Mweight_smooth'][smooth_select],   histtype='step', color='red', label='CANDELS Smooth', bins=thebins)
    axs.hist(sdss['dM_g'][sdss_smooth & np.invert(np.isnan(sdss[sdss_col]))],   weights=sdss['Mweight_smooth'][sdss_smooth & np.invert(np.isnan(sdss[sdss_col]))],   histtype='step', color='red', label='SDSS Smooth', linestyle='dashed', bins=thebins)


    ylim3 = (0., 47.)

    #axs.set_xlim(xlimits)
    axs.set_ylim(ylim3)
    axs.set_xlabel("$M - M^*$")
    axs.legend(loc='upper left', frameon=False) #loc=2


    # BOTTOM RIGHT
    axt = figM.add_subplot(2,2,4)

    axt.hist(gzmm['dM_V'][featured_select], weights=gzmm['Mweight_feat'][featured_select], histtype='step', color='blue', label='CANDELS Featured', bins=thebins)
    axt.hist(sdss['dM_g'][sdss_featured & np.invert(np.isnan(sdss[sdss_col]))], weights=sdss['Mweight_feat'][sdss_featured & np.invert(np.isnan(sdss[sdss_col]))], histtype='step', color='blue', label='SDSS Featured', linestyle='dashed', bins=thebins)

    ylim4 = (0., 17.9)

    #axt.set_xlim(xlimits)
    axt.set_ylim(ylim4)
    axt.set_xlabel("$M - M^*$")
    axt.legend(loc='upper left', frameon=False) #loc=2





    plt.tight_layout()

    plt.savefig('dM_histogram_all_x10a_Simard.png', facecolor='None', edgecolor='None')
    plt.savefig('dM_histogram_all_x10a_Simard.eps', facecolor='None', edgecolor='None')

    plt.clf()
    plt.cla()
    plt.close('All')
    plt.close()
    plt.close()
    plt.close()
    plt.close()











def plot_dM():
    figM = plt.figure(figsize=(8,4))
    axq = figM.add_subplot(1,2,1)

    thebins = np.arange(-4.,4.25,.25)

    #axq.hist(gzmm['dM_V'][smooth_select],   weights=smooth_wt,   histtype='step', color='red', label='Smooth', linestyle='dashed')
    #axq.hist(gzmm['dM_V'][featured_select], weights=featured_wt, histtype='step', color='blue', label='Featured')

    #axq.hist(gzmm['dM_V'][smooth_select],   histtype='step', color='red', label='Smooth', linestyle='dashed')
    #axq.hist(gzmm['dM_V'][featured_select], histtype='step', color='blue', label='Featured')

    axq.hist(gzmm['dM_V'][smooth_select],   weights=gzmm['Mweight_feat'][smooth_select],   histtype='step', color='red', label='Smooth', linestyle='dashed', bins=thebins)
    axq.hist(gzmm['dM_V'][featured_select], weights=gzmm['Mweight_feat'][featured_select], histtype='step', color='blue', label='Featured', bins=thebins)


    #axq.hist(gzmm['dM_V'][smooth_select],   histtype='step', color='red', label='Smooth', linestyle='dashed', cumulative=True)
    #axq.hist(gzmm['dM_V'][featured_select], histtype='step', color='blue', label='Featured', cumulative=True)


    #axq.set_xlim(xlimits)
    #axq.set_ylim(ylimits)
    axq.set_xlabel("CANDELS $M_V - M^*_V$")
    axq.legend(loc='upper left', frameon=False) #loc=2


    axr = figM.add_subplot(1,2,2)


    #axr.hist(sdss['dM_g'][sdss_smooth],   weights=sdss_smooth_wt,   histtype='step', color='red', label='Smooth', linestyle='dashed')
    #axr.hist(sdss['dM_g'][sdss_featured], weights=sdss_featured_wt, histtype='step', color='blue', label='Featured')

    axr.hist(sdss['dM_g'][sdss_smooth],   weights=sdss['Mweight_feat'][sdss_smooth],   histtype='step', color='red', label='Smooth', linestyle='dashed', bins=thebins)
    axr.hist(sdss['dM_g'][sdss_featured], weights=sdss['Mweight_feat'][sdss_featured], histtype='step', color='blue', label='Featured', bins=thebins)

    #axr.hist(sdss['dM_g'][sdss_smooth],   histtype='step', color='red', label='Smooth', linestyle='dashed')
    #axr.hist(sdss['dM_g'][sdss_featured], histtype='step', color='blue', label='Featured')

    axr.set_xlabel("SDSS $M_g - M^*_g$")
    axr.legend(loc='upper left', frameon=False) #loc=2


    plt.tight_layout()

    plt.savefig('dM_histogram_adj_all_to_CANDELS.png', facecolor='None', edgecolor='None')
    # plt.savefig('dM_histogram_adj.eps', facecolor='None', edgecolor='None')

    plt.clf()
    plt.cla()
    plt.close('All')





###########################################
# B/Tot plots

def plot_btot_hist_bymorph_cumul(whichsdss='Simard'):

    sdss_col = btot
    pval_s = p_smooth
    pval_f = p_featured
    filelabel = 'Simard'


    # first initialise the WHOLE figure
    fig2 = plt.figure(figsize=(8, 4))

    xlimits = (-0.1, 1.1)
    #ylimits = (0.0, 0.275)
    ylimits_f = (0., 1.3*num_featured)
    ylimits_s = (0., 1.3*num_smooth)
    sbins_smooth   = np.arange(-0.15, 1.15, 0.001)
    sbins_featured = np.arange(-0.15, 1.15, 0.001)

    plabel_height = 0.1


    axq = fig2.add_subplot(1,2,1)

    axq.hist(gzmm[gzbt][smooth_select],   weights=gzmm['Mweight_smooth'][smooth_select],   bins=sbins_smooth,   histtype='step', color='red', label='CANDELS Smooth', cumulative=True)
    axq.hist(sdss[sdss_col][sdss_smooth & np.invert(np.isnan(sdss[sdss_col]))],   weights=sdss['Mweight_smooth'][sdss_smooth & np.invert(np.isnan(sdss[sdss_col]))],   bins=sbins_smooth,   histtype='step', color='red', label='SDSS Smooth', linestyle='dashed', cumulative=True)

    if pval_s > 0.005:
        axq.text(0.7, plabel_height*ylimits_s[1], '$p_{KS} = %.3f$' % pval_s)
    elif pval_s > 0.0005:
        axq.text(0.7, plabel_height*ylimits_s[1], '$p_{KS} = %.4f$' % pval_s)
    else:
        axq.text(0.6, plabel_height*ylimits_s[1], '$p_{KS} = %.1e$' % pval_s)

    axq.set_xlim(xlimits)
    axq.set_ylim(ylimits_s)
    axq.set_xlabel("B/Tot")
    axq.legend(loc='upper left', frameon=False) #loc=2


    axr = fig2.add_subplot(1,2,2)

    axr.hist(gzmm[gzbt][featured_select], weights=gzmm['Mweight_feat'][featured_select], bins=sbins_featured, histtype='step', color='blue', label='CANDELS Featured', cumulative=True)
    axr.hist(sdss[sdss_col][sdss_featured & np.invert(np.isnan(sdss[sdss_col]))], weights=sdss['Mweight_feat'][sdss_featured & np.invert(np.isnan(sdss[sdss_col]))], bins=sbins_featured, histtype='step', color='blue', label='SDSS Featured', linestyle='dashed', cumulative=True)

    if pval_f > 0.005:
        axr.text(0.7, plabel_height*ylimits_f[1], '$p_{KS} = %.3f$' % pval_f)
    elif pval_f > 0.0005:
        axr.text(0.7, plabel_height*ylimits_f[1], '$p_{KS} = %.4f$' % pval_f)
    else:
        axr.text(0.6, plabel_height*ylimits_f[1], '$p_{KS} = %.1e$' % pval_f)


    axr.set_xlim(xlimits)
    axr.set_ylim(ylimits_f)
    axr.set_xlabel("B/Tot")
    axr.legend(loc='upper left', frameon=False) #loc=2


    plt.tight_layout()

    plt.savefig('BTot_histogram_bymorph_Lmatched_cumul_%s.png' % filelabel, facecolor='None', edgecolor='None')
    #plt.savefig('BTot_histogram.eps', facecolor='None', edgecolor='None')

    plt.clf()
    plt.cla()
    plt.close('All')













###########################################
# B/Tot plots, not cumulative

def plot_btot_hist_bymorph(whichsdss='Simard'):
    sdss_col = btot
    pval_s = p_smooth
    pval_f = p_featured
    filelabel = 'Simard'
    hists_featured = gzmm_featured_hist[0] + sdss_featured_hist[0]
    hists_smooth   = gzmm_smooth_hist[0]   + sdss_smooth_hist[0]
    ylim_shi = 45.5
    ylim_fhi = 21.5


    # first initialise the WHOLE figure
    fig2 = plt.figure(figsize=(8, 4))

    xlimits = (-0.1, 1.1)
    #ylimits = (0.0, 0.275)
    ylimits_f = (0., ylim_fhi)
    ylimits_s = (0., ylim_shi)
    sbins_smooth   = np.arange(-0.15, 1.15, 0.1)
    sbins_featured = np.arange(-0.15, 1.15, 0.1)

    plabel_height = 0.71


    axq = fig2.add_subplot(1,2,1)

    axq.hist(gzmm[gzbt][smooth_select],   weights=gzmm['Mweight_smooth'][smooth_select],   bins=sbins_smooth,   histtype='step', color='red', label='CANDELS Smooth')
    axq.hist(sdss[sdss_col][sdss_smooth & np.invert(np.isnan(sdss[sdss_col]))],   weights=sdss['Mweight_smooth'][sdss_smooth & np.invert(np.isnan(sdss[sdss_col]))],   bins=sbins_smooth,   histtype='step', color='red', label='SDSS Smooth', linestyle='dashed')

    if pval_s > 0.005:
        axq.text(0.16, plabel_height*ylimits_s[1], '$p_{KS} = %.3f$' % pval_s)
    elif pval_s > 0.0005:
        axq.text(0.16, plabel_height*ylimits_s[1], '$p_{KS} = %.4f$' % pval_s)
    else:
        axq.text(0.16, plabel_height*ylimits_s[1], '$p_{KS} = %.1e$' % pval_s)

    axq.set_xlim(xlimits)
    axq.set_ylim(ylimits_s)
    axq.set_xlabel("B/Tot")
    axq.legend(loc='upper left', frameon=False) #loc=2


    axr = fig2.add_subplot(1,2,2)

    axr.hist(gzmm[gzbt][featured_select], weights=gzmm['Mweight_feat'][featured_select], bins=sbins_featured, histtype='step', color='blue', label='CANDELS Featured')
    axr.hist(sdss[sdss_col][sdss_featured & np.invert(np.isnan(sdss[sdss_col]))], weights=sdss['Mweight_feat'][sdss_featured & np.invert(np.isnan(sdss[sdss_col]))], bins=sbins_featured, histtype='step', color='blue', label='SDSS Featured', linestyle='dashed')

    if pval_f > 0.005:
        axr.text(0.16, plabel_height*ylimits_f[1], '$p_{KS} = %.3f$' % pval_f)
    elif pval_f > 0.0005:
        axr.text(0.16, plabel_height*ylimits_f[1], '$p_{KS} = %.4f$' % pval_f)
    else:
        axr.text(0.16, plabel_height*ylimits_f[1], '$p_{KS} = %.1e$' % pval_f)


    axr.set_xlim(xlimits)
    axr.set_ylim(ylimits_f)
    axr.set_xlabel("B/Tot")
    axr.legend(loc='upper left', frameon=False) #loc=2


    plt.tight_layout()

    plt.savefig('BTot_histogram_bymorph_Lmatched_%s.png' % filelabel, facecolor='None', edgecolor='None')
    plt.savefig('BTot_histogram_bymorph_Lmatched_%s.eps' % filelabel, facecolor='None', edgecolor='None')

    plt.clf()
    plt.cla()
    plt.close('All')
    plt.close()
    plt.close()
    plt.close()





















# you're not using a super-new version of astropy so this doesn't work yet
#gzmm = Table.read(infile).to_pandas()

gzmm = Table.read(infile_CANDELS)
sdss_all = Table.read(infile_SDSS)

# get rid of nan values
sdss = sdss_all[np.invert(np.isnan(sdss_all[btot]))]

#there is probably a better way to do this with transposes or something?
gzmm['N_GALFIT_BAND_V'] = [q[0] for q in gzmm['N_GALFIT_BAND']]
gzmm['N_GALFIT_BAND_I'] = [q[1] for q in gzmm['N_GALFIT_BAND']]
gzmm['N_GALFIT_BAND_J'] = [q[2] for q in gzmm['N_GALFIT_BAND']]
gzmm['N_GALFIT_BAND_H'] = [q[3] for q in gzmm['N_GALFIT_BAND']]
gzmm['RE_GALFIT_BAND_V'] = [q[0] for q in gzmm['RE_GALFIT_BAND']]
gzmm['RE_GALFIT_BAND_I'] = [q[1] for q in gzmm['RE_GALFIT_BAND']]
gzmm['RE_GALFIT_BAND_J'] = [q[2] for q in gzmm['RE_GALFIT_BAND']]
gzmm['RE_GALFIT_BAND_H'] = [q[3] for q in gzmm['RE_GALFIT_BAND']]
gzmm['N_GALFIT_BAND_B_V'] = [q[0] for q in gzmm['N_GALFIT_BAND_B']]
gzmm['N_GALFIT_BAND_B_I'] = [q[1] for q in gzmm['N_GALFIT_BAND_B']]
gzmm['N_GALFIT_BAND_B_J'] = [q[2] for q in gzmm['N_GALFIT_BAND_B']]
gzmm['N_GALFIT_BAND_B_H'] = [q[3] for q in gzmm['N_GALFIT_BAND_B']]
gzmm['N_GALFIT_BAND_D_V'] = [q[0] for q in gzmm['N_GALFIT_BAND_D']]
gzmm['N_GALFIT_BAND_D_I'] = [q[1] for q in gzmm['N_GALFIT_BAND_D']]
gzmm['N_GALFIT_BAND_D_J'] = [q[2] for q in gzmm['N_GALFIT_BAND_D']]
gzmm['N_GALFIT_BAND_D_H'] = [q[3] for q in gzmm['N_GALFIT_BAND_D']]
gzmm['RE_GALFIT_BAND_B_V'] = [q[0] for q in gzmm['RE_GALFIT_BAND_B']]
gzmm['RE_GALFIT_BAND_B_I'] = [q[1] for q in gzmm['RE_GALFIT_BAND_B']]
gzmm['RE_GALFIT_BAND_B_J'] = [q[2] for q in gzmm['RE_GALFIT_BAND_B']]
gzmm['RE_GALFIT_BAND_B_H'] = [q[3] for q in gzmm['RE_GALFIT_BAND_B']]
gzmm['RE_GALFIT_BAND_D_V'] = [q[0] for q in gzmm['RE_GALFIT_BAND_D']]
gzmm['RE_GALFIT_BAND_D_I'] = [q[1] for q in gzmm['RE_GALFIT_BAND_D']]
gzmm['RE_GALFIT_BAND_D_J'] = [q[2] for q in gzmm['RE_GALFIT_BAND_D']]
gzmm['RE_GALFIT_BAND_D_H'] = [q[3] for q in gzmm['RE_GALFIT_BAND_D']]

if gzbt == 'BTot_rest_g':
    # well, crap, we were asked to use this and we haven't yet created it
    # so do that

    central_wavelength = 4770. # sloan g-band
    # the observed central wavelength of rest-g at each redshift
    gzmm['obs_g'] = central_wavelength*(1. + gzmm['z_best'])

    cwave_band = np.array([6060., 8140., 12500., 16000.])

    gzmm['BTot_rest_g'] = gzmm['obs_g']*0.0 - 1.0

    # populate the BTot_rest_g values

    for i, bt in enumerate(gzmm['BTot_galfit_band_BD']):
        # we don't have a full SED of B/Tot, just between observed V and H
        # and interp1d will yell about bounds, so let's be explicit
        if gzmm['z_best'][i] > 2.3:
            # just use F160W
            gzmm['BTot_rest_g'][i] = bt[3]
        elif gzmm['z_best'][i] < 0.3:
            # just use F606W
            gzmm['BTot_rest_g'][i] = bt[0]
        else:
            f = interpolate.interp1d(cwave_band, bt)
            gzmm['BTot_rest_g'][i] = f(gzmm['obs_g'][i])
    # end creating rest-g B/Tot value


# set up basic selections, e.g. not artifact, not potentially mismatched, high enough surface brightness, right redshift ranges
min_n = 0.22
max_n = 7.8
min_r = 0.33
max_r = 390.0

not_artifact = (gzmm['t00_smooth_or_featured_a2_artifact_weighted_frac'] < 0.5) & (gzmm['Separation'] <= 1.0) & (gzmm['FLAG_GALFIT_BD'] == 2)
fit_not_constrained = (gzmm['FLAG_GALFIT_BD'] == 2) & (gzmm['N_GALFIT_BAND_V'] > min_n) & (gzmm['N_GALFIT_BAND_I'] > min_n) & (gzmm['N_GALFIT_BAND_J'] > min_n) & (gzmm['N_GALFIT_BAND_H'] > min_n) & (gzmm['N_GALFIT_BAND_V'] < max_n) & (gzmm['N_GALFIT_BAND_I'] < max_n) & (gzmm['N_GALFIT_BAND_J'] < max_n) & (gzmm['N_GALFIT_BAND_H'] < max_n) & (gzmm['N_GALFIT_BAND_B_V'] > min_n) & (gzmm['N_GALFIT_BAND_B_I'] > min_n) & (gzmm['N_GALFIT_BAND_B_J'] > min_n) & (gzmm['N_GALFIT_BAND_B_H'] > min_n) & (gzmm['N_GALFIT_BAND_B_V'] < max_n) & (gzmm['N_GALFIT_BAND_B_I'] < max_n) & (gzmm['N_GALFIT_BAND_B_J'] < max_n) & (gzmm['N_GALFIT_BAND_B_H'] < max_n) & (gzmm['RE_GALFIT_BAND_V'] > min_r) & (gzmm['RE_GALFIT_BAND_I'] > min_r) & (gzmm['RE_GALFIT_BAND_J'] > min_r) & (gzmm['RE_GALFIT_BAND_H'] > min_r) & (gzmm['RE_GALFIT_BAND_V'] < max_r) & (gzmm['RE_GALFIT_BAND_I'] < max_r) & (gzmm['RE_GALFIT_BAND_J'] < max_r) & (gzmm['RE_GALFIT_BAND_H'] < max_r) & (gzmm['RE_GALFIT_BAND_B_V'] > min_r) & (gzmm['RE_GALFIT_BAND_B_I'] > min_r) & (gzmm['RE_GALFIT_BAND_B_J'] > min_r) & (gzmm['RE_GALFIT_BAND_B_H'] > min_r) & (gzmm['RE_GALFIT_BAND_B_V'] < max_r) & (gzmm['RE_GALFIT_BAND_B_I'] < max_r) & (gzmm['RE_GALFIT_BAND_B_J'] < max_r) & (gzmm['RE_GALFIT_BAND_B_H'] < max_r) & (gzmm['RE_GALFIT_BAND_D_V'] > min_r) & (gzmm['RE_GALFIT_BAND_D_I'] > min_r) & (gzmm['RE_GALFIT_BAND_D_J'] > min_r) & (gzmm['RE_GALFIT_BAND_D_H'] > min_r) & (gzmm['RE_GALFIT_BAND_D_V'] < max_r) & (gzmm['RE_GALFIT_BAND_D_I'] < max_r) & (gzmm['RE_GALFIT_BAND_D_J'] < max_r) & (gzmm['RE_GALFIT_BAND_D_H'] < max_r)
sb_okay = gzmm['SB_AUTO'] <= 24.5
z_range = (gzmm['z_best'] >= 1.0) & (gzmm['z_best'] < 3.0)


# be strict on the smooth sample; I'd like to do this 0.5 and 0.5 but that's too strict for features
# and obviously you don't want the samples to overlap at all
smooth_select   = (gzmm['t00_smooth_or_featured_a0_smooth_weighted_frac'] > 0.6) & (not_artifact) & (fit_not_constrained) & (sb_okay) & (z_range)
featured_select = (gzmm['t00_smooth_or_featured_a1_features_weighted_frac'] >= 0.4) & (not_artifact) & (fit_not_constrained) & (sb_okay) & (z_range)

# 1/N_tot_in_subset so you can do normalized histograms later
smooth_wt   = np.ones_like(np.array(gzmm[gzbt][smooth_select]))/float(len(gzmm['bulgesersic_fluxweighted_tot'][smooth_select]))
featured_wt = np.ones_like(np.array(gzmm[gzbt][featured_select]))/float(len(gzmm['bulgesersic_fluxweighted_tot'][featured_select]))


# interpolate to the knee of the LF at each source's redshift
# Marchesini et al. (2012) rest-frame V-band LF evolution
# these are M* values (the knee)
# bins are 0.4-0.7, 0.76-1.1, 1.1-1.5, 1.5-2.1, 2.1-2.7, 2.7-3.3
zLF_V = np.array([  0.4,   0.55,  0.93,  1.3,  1.8,   2.4,   3.0,   3.3])
MLF_V = np.array([-21.76,-21.76,-22.15,-22.2,-22.47,-22.59,-22.83,-22.83])

# interpolate and just use NaN for anything outside the bound
f = interpolate.interp1d(zLF_V, MLF_V, bounds_error=False)
gzmm['Mstar_V'] = [f(q) for q in gzmm['z_best']]

# g-band M* for SDSS: -20.04 (Blanton et al. 2001)
Mstar_g = -20.04


# where does each actual luminosity sit relative to
# the knee of the LF at that z?
# this is defined so that fainter than knee == positive number
# which is backwards, but then again so are magnitudes
# up yours, magnitude system
# (anyway it doesn't matter because this will never be plotted etc.)
# (except that whole function below that plots it)
# (but that's just for sanity checks etc, not publication)
# (okay nevermind I decided it's useful to publish it)
# (I explained myself in the caption of the paper, minus "up yours")
gzmm['dM_V'] = gzmm['V_rest'] - gzmm['Mstar_V']

sdss['dM_g'] = [get_dMg(q) for q in sdss[gMag]]

sdss_z_use    = (sdss[redshift] >= 0.0418) & (sdss[redshift] <= 0.04355)
sdss_smooth   = sdss_z_use & (sdss['t01_smooth_or_features_a01_smooth_flag'] == 1)
sdss_featured = sdss_z_use & (sdss['t01_smooth_or_features_a02_features_or_disk_flag'] == 1)

sdss_smooth_wt   = np.ones_like(np.array(sdss[redshift][sdss_smooth]))/float(len(sdss[redshift][sdss_smooth]))
sdss_featured_wt = np.ones_like(np.array(sdss[redshift][sdss_featured]))/float(len(sdss[redshift][sdss_featured]))




bins_dM = np.arange(-7.0,4.8,0.4)
gzmm['Mweight_feat'] = gzmm['dM_V'] * 0.0 + 1.0
sdss['Mweight_feat'] = sdss['dM_g'] * 0.0 + 1.0

gzmm['Mweight_smooth'] = gzmm['dM_V'] * 0.0 + 1.0
sdss['Mweight_smooth'] = sdss['dM_g'] * 0.0 + 1.0

dbins = bins_dM[1] - bins_dM[0]


for this_bin in bins_dM:
    print("----------------\n   Bin: %.1f" % this_bin)
    #print("Checking |dM - %.1f| < %.2f\n" % (this_bin, dbins/2.))
    this_gzmm = (np.abs(gzmm['dM_V'] - this_bin) < (dbins/2.))
    this_sdss = (np.abs(sdss['dM_g'] - this_bin) < (dbins/2.))

    this_f_gzmm = this_gzmm & featured_select
    this_s_gzmm = this_gzmm & smooth_select

    this_f_sdss = this_sdss & sdss_featured
    this_s_sdss = this_sdss & sdss_smooth

    n_this_f_gzmm = sum(this_f_gzmm)
    n_this_s_gzmm = sum(this_s_gzmm)

    n_this_f_sdss = sum(this_f_sdss)
    n_this_s_sdss = sum(this_s_sdss)


    print("CANDELS: %d featured, %d smooth in this bin" % (n_this_f_gzmm, n_this_s_gzmm))
    print("SDSS:    %d featured, %d smooth in this bin\n" % (n_this_f_sdss, n_this_s_sdss))

    # if any of them have 0 objects in the bin then we weight by 0
    if (n_this_f_gzmm == 0) | (n_this_s_gzmm == 0) | (n_this_f_sdss == 0) | (n_this_s_sdss == 0):
        gzmm['Mweight_feat'][this_f_gzmm | this_s_gzmm] = gzmm['Mweight_feat'][this_f_gzmm | this_s_gzmm]*0.0
        sdss['Mweight_feat'][this_f_sdss | this_s_sdss] = sdss['Mweight_feat'][this_f_sdss | this_s_sdss]*0.0
        gzmm['Mweight_smooth'][this_f_gzmm | this_s_gzmm] = gzmm['Mweight_smooth'][this_f_gzmm | this_s_gzmm]*0.0
        sdss['Mweight_smooth'][this_f_sdss | this_s_sdss] = sdss['Mweight_smooth'][this_f_sdss | this_s_sdss]*0.0
    else:
    # otherwise we compute the weights, but have to figure out which is smallest?
    # or maybe we could do this in a more clever way
        nbin_min = min([n_this_f_gzmm, n_this_s_gzmm, n_this_f_sdss, n_this_s_sdss])
        # w == 1 if it's the minimum bin, <1 otherwise
        w_f_gzmm = float(nbin_min)/float(n_this_f_gzmm)
        w_s_gzmm = float(nbin_min)/float(n_this_s_gzmm)
        w_f_sdss = float(nbin_min)/float(n_this_f_sdss)
        w_s_sdss = float(nbin_min)/float(n_this_s_sdss)

        gzmm['Mweight_feat'][this_f_gzmm] = gzmm['Mweight_feat'][this_f_gzmm]*0.0 + w_f_gzmm
        sdss['Mweight_feat'][this_f_sdss] = sdss['Mweight_feat'][this_f_sdss]*0.0 + w_f_sdss

        gzmm['Mweight_smooth'][this_s_gzmm] = gzmm['Mweight_smooth'][this_s_gzmm]*0.0 + w_s_gzmm
        sdss['Mweight_smooth'][this_s_sdss] = sdss['Mweight_smooth'][this_s_sdss]*0.0 + w_s_sdss

'''
once the binning is done, we end up with the smooth and featured weights for gzmm and sdss summing to ~the number of galaxies in the smallest sample of all 4 samples.

But that isn't exactly right, because there might actually be more galaxies in both the smooth samples than in one of the featured samples, for example. And we're going to do a K-S test so we need to have the right numbers.

So figure out which set of weights can be renormalised.

'''

max_weight_gzmm_f = max(gzmm['Mweight_feat'][featured_select])
max_weight_sdss_f = max(sdss['Mweight_feat'][sdss_featured])

max_weight_gzmm_s = max(gzmm['Mweight_smooth'][smooth_select])
max_weight_sdss_s = max(sdss['Mweight_smooth'][sdss_smooth])

max_featured = max([max_weight_gzmm_f, max_weight_sdss_f])
max_smooth   = max([max_weight_gzmm_s, max_weight_sdss_s])

if max_featured > max_smooth:
    # in this case, the smooth weights are all too low, so adjust them
    # so that the max weight for a galaxy is 1 for smooth galaxies
    gzmm['Mweight_smooth'][smooth_select] = gzmm['Mweight_smooth'][smooth_select]/max_smooth
    sdss['Mweight_smooth'][sdss_smooth] = sdss['Mweight_smooth'][sdss_smooth]/max_smooth
elif max_smooth > max_featured:
    # in this case it's the featured galaxies that are too low
    gzmm['Mweight_feat'][featured_select] = gzmm['Mweight_feat'][featured_select]/max_featured
    sdss['Mweight_feat'][sdss_featured] = sdss['Mweight_feat'][sdss_featured]/max_featured
else:
    # if they are already equal, don't do anything
    # the max values should both be 1 in this case
    pass


# we are going to plot these later but we also need to do a K-S test,
# and because of the weights we need to do it by hand. (ick)
#
sbins_smooth   = np.arange(-0.0015, 1.0015, 0.001)
sbins_featured = np.arange(-0.0015, 1.0015, 0.001)

# these will make histograms, not cumulative histograms
# also they're not normalized to have sum(histogram) == 1
# (numpy.histogram has a normed=boolean keyword but it doesn't actually do that)
gzmm_smooth_hist = np.histogram(gzmm[gzbt][smooth_select],   weights=gzmm['Mweight_smooth'][smooth_select],   bins=sbins_smooth)
sdss_smooth_hist = np.histogram(sdss[btot][sdss_smooth],   weights=sdss['Mweight_smooth'][sdss_smooth],     bins=sbins_smooth)
#sdss_smooth_hist_nb = np.histogram(sdss['BULGE_TO_TOT_NB_SUM'][sdss_smooth],   weights=sdss['Mweight_smooth'][sdss_smooth],     bins=sbins_smooth)

gzmm_featured_hist = np.histogram(gzmm[gzbt][featured_select],   weights=gzmm['Mweight_feat'][featured_select],   bins=sbins_featured)
sdss_featured_hist = np.histogram(sdss[btot][sdss_featured & np.invert(np.isnan(sdss[btot]))],   weights=sdss['Mweight_feat'][sdss_featured & np.invert(np.isnan(sdss[btot]))],     bins=sbins_featured)
#sdss_featured_hist_nb = np.histogram(sdss['BULGE_TO_TOT_NB_SUM'][sdss_featured],   weights=sdss['Mweight_feat'][sdss_featured],     bins=sbins_featured)

# but K-S needs a cumulative normalised histogram


gzmm_sh_norm    = gzmm_smooth_hist[0]/float(sum(gzmm_smooth_hist[0]))
sdss_sh_norm    = sdss_smooth_hist[0]/float(sum(sdss_smooth_hist[0]))
#sdss_sh_norm_nb = sdss_smooth_hist_nb[0]/float(sum(sdss_smooth_hist_nb[0]))
gzmm_sh_cnorm    = gzmm_sh_norm*0.0
sdss_sh_cnorm    = sdss_sh_norm*0.0
#sdss_sh_cnorm_nb = sdss_sh_norm_nb*0.0

gzmm_fh_norm    = gzmm_featured_hist[0]/float(sum(gzmm_featured_hist[0]))
sdss_fh_norm    = sdss_featured_hist[0]/float(sum(sdss_featured_hist[0]))
#sdss_fh_norm_nb = sdss_featured_hist_nb[0]/float(sum(sdss_featured_hist_nb[0]))
gzmm_fh_cnorm    = gzmm_fh_norm*0.0
sdss_fh_cnorm    = sdss_fh_norm*0.0
#sdss_fh_cnorm_nb = sdss_fh_norm_nb*0.0

for i, bin in enumerate(sbins_smooth):
    try:
        gzmm_sh_cnorm[i]    = sum(gzmm_sh_norm[0:i+1])
        sdss_sh_cnorm[i]    = sum(sdss_sh_norm[0:i+1])
        #sdss_sh_cnorm_nb[i] = sum(sdss_sh_norm_nb[0:i+1])
    except IndexError:
        pass # why is it trying to go from i=0 to i=len(arr) here,
        # instead of i=len(arr)-1? It's not single-indexed! +1 issue, ugh

for i, bin in enumerate(sbins_featured):
    try:
        gzmm_fh_cnorm[i]    = sum(gzmm_fh_norm[0:i+1])
        sdss_fh_cnorm[i]    = sum(sdss_fh_norm[0:i+1])
        #sdss_fh_cnorm_nb[i] = sum(sdss_fh_norm_nb[0:i+1])
    except IndexError:
        pass

#get the max distance between the cumulative, normalised curves
# this is the K-S statistic
# https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Two-sample_Kolmogorov.E2.80.93Smirnov_test
#
# see http://sparky.rice.edu/astr360/kstest.pdf for how to calculate the critical value at a particular level of significance
dist_smooth   = max(np.abs(gzmm_sh_cnorm - sdss_sh_cnorm))
dist_featured = max(np.abs(gzmm_fh_cnorm - sdss_fh_cnorm))

#dist_smooth_nb   = max(np.abs(gzmm_sh_cnorm - sdss_sh_cnorm_nb))
#dist_featured_nb = max(np.abs(gzmm_fh_cnorm - sdss_fh_cnorm_nb))

# we've set it up so n_gzmm and n_sdss are the same within these categories
num_smooth   = sum(gzmm['Mweight_smooth'][smooth_select])
num_featured = sum(gzmm['Mweight_feat'][featured_select])

'''
for a given p-value alpha, the "critical" K-S statistic value, i.e. the value the statistic must have for the p-value to equal alpha, is given by the lookup table at the rice.edu link above, or this:
    d_crit(alpha) = c(alpha)*sqrt((n1+n2)/(n1*n2))
    where n1, n2 are the dimensions of the 2 samples
    and c(alpha) is the inverse of the kolmogorov distribution at that alpha,
    and is kind of a bitch to calculate for any given p-value but here's a table
    for some specific ones:
        alpha           c(alpha)
        ------------------------
        0.1             1.22
        0.05            1.36      (p=0.05 is ~2 sigma)
        0.025           1.48
        0.01            1.63      (p=0.01 is ~3 sigma)
        0.005           1.73
        0.001           1.95

    Okay so it was 4 am when I was writing the above and I didn't have the remaining mental capacity to google "scipy inverse kolmogorov" but it does exist: scipy.special.kolmogi(alpha) == c(alpha)

    but also there's scipy.special.kolmogorov(c) == alpha
    i.e. we can calculate the c value we actually have and get a p-value!
    Yet another lesson in not wasting time trying to do work when exhausted.
    Sleep on it and all will be clear(er) in the morning.

    Also the special.erfc(x) is the complementary error function, where x = sigma/sqrt(2)
    i.e. the table of p-value-to-sigma at https://en.wikipedia.org/wiki/Normal_distribution#Standard_deviation_and_tolerance_intervals is actually a table of erf(x), erfc(x), 1./erfc(x) for sigma values from 1 to 6

    and special.erfcinv(alpha)*np.sqrt(2.) will return the significance level in sigma.

'''

smooth_term   = np.sqrt((2.0*num_smooth)/(num_smooth*num_smooth))
featured_term = np.sqrt((2.0*num_featured)/ (num_featured*num_featured))

cval_smooth    = dist_smooth/smooth_term
#cval_smooth_nb = dist_smooth_nb/smooth_term
cval_featured    = dist_featured/featured_term
#cval_featured_nb = dist_featured_nb/featured_term

p_smooth    = special.kolmogorov(cval_smooth)
#p_smooth_nb = special.kolmogorov(cval_smooth_nb)
p_featured    = special.kolmogorov(cval_featured)
#p_featured_nb = special.kolmogorov(cval_featured_nb)

sigma_smooth    = special.erfcinv(p_smooth)*np.sqrt(2.)
#sigma_smooth_nb = special.erfcinv(p_smooth_nb)*np.sqrt(2.)
sigma_featured    = special.erfcinv(p_featured)*np.sqrt(2.)
#sigma_featured_nb = special.erfcinv(p_featured_nb)*np.sqrt(2.)


c_2sig = 1.36
dcrit_smooth_2sig   = c_2sig*smooth_term
dcrit_featured_2sig = c_2sig*featured_term

c_3sig = 1.63
dcrit_smooth_3sig   = c_3sig*smooth_term
dcrit_featured_3sig = c_3sig*featured_term


print("Smooth: %.2f effective objects" % num_smooth )
print("K-S distance: %.3f" % (dist_smooth))
print("  D critical, 2-sigma: %.3f, 3-sigma: %.3f" % (dcrit_smooth_2sig, dcrit_smooth_3sig))
print(" ... p-value: %.3e (%.1f sigma)\n" % (p_smooth, sigma_smooth))


print("Featured: %.2f effective objects" % num_featured )
print("K-S distance: %.3f" % (dist_featured))
print("  D critical, 2-sigma: %.3f, 3-sigma: %.3f" % (dcrit_featured_2sig, dcrit_featured_3sig))
print(" ... p-value: %.3e (%.1f sigma)\n" % (p_featured, sigma_featured))




plot_btot_hist_bymorph_cumul('Simard')
plot_btot_hist_bymorph('Simard')
plot_dM_bymorph()

plt.clf()
plt.cla()
plt.close()
plt.close()












###########################################
# B/Tot plots

def plot_btot_hist():
    # first initialise the WHOLE figure
    fig2 = plt.figure(figsize=(8, 4))

    xlimits = (-0.1, 1.1)
    ylimits = (0.0, 0.275)
    sbins_smooth   = np.arange(-0.1, 1.1, 0.1)
    sbins_featured = np.arange(-0.1, 1.1, 0.1)


    axq = fig2.add_subplot(1,2,1)

    axq.hist(gzmm[gzbt][smooth_select],   weights=gzmm['Mweight_feat'][smooth_select],   bins=sbins_smooth,   histtype='step', color='red', label='Smooth', linestyle='dashed')
    axq.hist(gzmm[gzbt][featured_select], weights=gzmm['Mweight_feat'][featured_select], bins=sbins_featured, histtype='step', color='blue', label='Featured')


    axq.set_xlim(xlimits)
    #axq.set_ylim(ylimits)
    axq.set_xlabel("CANDELS B/Tot")
    axq.legend(loc='upper right', frameon=False) #loc=2


    axr = fig2.add_subplot(1,2,2)

    axr.hist(sdss['BULGE_TO_TOT_NB_SUM'][sdss_smooth],   weights=sdss['Mweight_feat'][sdss_smooth],   bins=sbins_smooth,   histtype='step', color='red', label='Smooth', linestyle='dashed')
    axr.hist(sdss['BULGE_TO_TOT_NB_SUM'][sdss_featured], weights=sdss['Mweight_feat'][sdss_featured], bins=sbins_featured, histtype='step', color='blue', label='Featured')


    axr.set_xlim(xlimits)
    #axr.set_ylim(ylimits)
    axr.set_xlabel("SDSS B/Tot")
    axr.legend(loc='upper right', frameon=False) #loc=2


    plt.tight_layout()

    plt.savefig('BTot_histogram_both_allsameLum.png', facecolor='None', edgecolor='None')
    #plt.savefig('BTot_histogram.eps', facecolor='None', edgecolor='None')

    plt.clf()
    plt.cla()
    plt.close('All')







def plot_hists_all():


    # first initialise the WHOLE figure
    fig = plt.figure(figsize=(12, 4))


    smooth_wt = np.ones_like(np.array(gzmm['sersic_fluxweighted_tot'][smooth_select]))/float(len(gzmm['bulgesersic_fluxweighted_tot'][smooth_select]))
    featured_wt = np.ones_like(np.array(gzmm['sersic_fluxweighted_tot'][featured_select]))/float(len(gzmm['bulgesersic_fluxweighted_tot'][featured_select]))

    ks_bulgesersic = stats.ks_2samp(gzmm['bulgesersic_fluxweighted_tot'][smooth_select], gzmm['bulgesersic_fluxweighted_tot'][featured_select])
    #Out[158]: (0.19472597597597596, 0.00059887162386288904)
    ks_sersic = stats.ks_2samp(gzmm['sersic_fluxweighted_tot'][smooth_select], gzmm['sersic_fluxweighted_tot'][featured_select])
    #Out[159]: (0.1943036786786787, 0.00062029840567241214)
    ks_btot = stats.ks_2samp(gzmm[gzbt][smooth_select], gzmm[gzbt][featured_select])
    #Out[160]: (0.21928178178178176, 6.8012923870561459e-05)

    print(sum(smooth_select), 'smooth galaxies')
    print(sum(featured_select), 'featured galaxies')
    print('p_KS_sersic:', ks_sersic[1])
    print('p_KS_BTot:', ks_btot[1])
    print('p_KS_bulgesersic:', ks_bulgesersic[1])


    ###########################################
    # Single-sersic plots
    xlimits = (-0.2, 8.5)
    sbins_smooth   = np.arange(-1.0, 13.0, .5)
    sbins_featured = np.arange(-1.0, 13.0, .5)

    #smooth_wt1   = np.ones_like(np.array(gzmm['sersic_fluxweighted_tot'][smooth_select]))/float(len(gzmm['bulgesersic_fluxweighted_tot'][smooth_select]))
    #featured_wt1 = np.ones_like(np.array(gzmm['sersic_fluxweighted_tot'][featured_select]))/float(len(gzmm['bulgesersic_fluxweighted_tot'][featured_select]))


    ax1 = fig.add_subplot(1,3,1)

    ax1.hist(gzmm['sersic_fluxweighted_tot'][smooth_select],   weights=smooth_wt,   bins=sbins_smooth,   histtype='step', color='red', label='Smooth', linestyle='dashed')
    ax1.hist(gzmm['sersic_fluxweighted_tot'][featured_select], weights=featured_wt, bins=sbins_featured, histtype='step', color='blue', label='Featured')

    ax1.set_xlim(xlimits)
    ax1.set_xlabel('Sersic n') # notice how with ax the command changes from plt.xlabel() to ax.set_xlabel()
    ax1.set_ylabel('N/Ntot')
    ax1.legend(frameon=False)



    ###########################################
    # B/Tot plots
    xlimits = (-0.1, 1.1)
    sbins_smooth   = np.arange(-0.1, 1.1, 0.1)
    sbins_featured = np.arange(-0.1, 1.1, 0.1)

    smooth_wt2   = np.ones_like(np.array(gzmm[gzbt][smooth_select]))/float(len(gzmm['bulgesersic_fluxweighted_tot'][smooth_select]))
    featured_wt2 = np.ones_like(np.array(gzmm[gzbt][featured_select]))/float(len(gzmm['bulgesersic_fluxweighted_tot'][featured_select]))

    ax2 = fig.add_subplot(1,3,2)

    ax2.hist(gzmm[gzbt][smooth_select],   weights=smooth_wt,   bins=sbins_smooth,   histtype='step', color='red', label='Smooth', linestyle='dashed')
    ax2.hist(gzmm[gzbt][featured_select], weights=featured_wt, bins=sbins_featured, histtype='step', color='blue', label='Featured')


    ax2.set_xlim(xlimits)
    ax2.set_xlabel("B/Tot")
    #ax2.legend(loc='upper right', frameon=False) #loc=2




    ###########################################
    # Bulge Sersic plots
    xlimits = (-0.2, 8.5)
    sbins_smooth   = np.arange(-1.0, 13.0, .5)
    sbins_featured = np.arange(-1.0, 13.0, .5)

    #smooth_wt1   = np.ones_like(np.array(gzmm['sersic_fluxweighted_tot'][smooth_select]))/float(len(gzmm['bulgesersic_fluxweighted_tot'][smooth_select]))
    #featured_wt1 = np.ones_like(np.array(gzmm['sersic_fluxweighted_tot'][featured_select]))/float(len(gzmm['bulgesersic_fluxweighted_tot'][featured_select]))


    ax3 = fig.add_subplot(1,3,3)

    ax3.hist(gzmm['bulgesersic_fluxweighted_tot'][smooth_select],   weights=smooth_wt,   bins=sbins_smooth,   histtype='step', color='red', label='Smooth', linestyle='dashed')
    ax3.hist(gzmm['bulgesersic_fluxweighted_tot'][featured_select], weights=featured_wt, bins=sbins_featured, histtype='step', color='blue', label='Featured')

    ax3.set_xlim(xlimits)
    ax3.set_xlabel('Bulge Sersic n')




    ###########################################
    # Bulge sersic plots IN LOG
    # xlimits = (0.15, 12.2)
    # ylimits = (0.0, 0.3)
    # sbins_smooth   = np.logspace(np.log10(0.2),np.log10(12.5),12)
    # sbins_featured = np.logspace(np.log10(0.2),np.log10(12.5),12)
    #
    # smooth_wt3   = np.ones_like(np.array(gzmm['bulgesersic_fluxweighted_tot'][smooth_select]))/float(len(gzmm['bulgesersic_fluxweighted_tot'][smooth_select]))
    # featured_wt3 = np.ones_like(np.array(gzmm['bulgesersic_fluxweighted_tot'][featured_select]))/float(len(gzmm['bulgesersic_fluxweighted_tot'][featured_select]))
    #
    # ax3 = fig.add_subplot(1,3,3)
    #
    # ax3.hist(gzmm['bulgesersic_fluxweighted_tot'][smooth_select],   weights=smooth_wt,   bins=sbins_smooth,   histtype='step', color='red', label='Smooth', linestyle='dashed')
    # ax3.hist(gzmm['bulgesersic_fluxweighted_tot'][featured_select], weights=featured_wt, bins=sbins_featured, histtype='step', color='blue', label='Featured')
    #
    #
    # ax3.set_xlim(xlimits)
    # ax3.set_ylim(ylimits)
    # ax3.set_xscale("log")
    # ax3.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax3.set_xlabel("Bulge Sersic n")



    plt.tight_layout()

    plt.savefig('histograms.png', facecolor='None', edgecolor='None')
    plt.savefig('histograms.eps', facecolor='None', edgecolor='None')


    plt.clf()
    plt.cla()
    plt.close('All')






# SB_AUTO WFC3_F160W_MAG
def plot_Hmag_bymorph():

    gzc_nanmag  = np.isnan(np.array(gzmm['WFC3_F160W_MAG']))
    # ugh these are strings and I don't really need them anyway, nevermind
    #sdss_nanmag = np.isnan(np.array(sdss['TOTAL_MAG_G_1']))


    figM = plt.figure(figsize=(8,6))
    #thebins = np.arange(-4.35,5.55,.3)
    thebins = np.arange(16.167,26.5,0.33)
    thebinssb = np.arange(16.67,25.33,0.33)

    # TOP LEFT
    axq = figM.add_subplot(2,2,1)

    axq.hist(gzmm['WFC3_F160W_MAG'][smooth_select & np.invert(gzc_nanmag)],   histtype='step', color='red', label='Smooth', bins=thebins)
    axq.hist(gzmm['WFC3_F160W_MAG'][featured_select & np.invert(gzc_nanmag)], weights=np.ones_like(gzmm['dM_V'][featured_select & np.invert(gzc_nanmag)])*10., histtype='step', color='blue', label='Featured ($\\times 10$)', bins=thebins)

    ylim1 = (0., 340.)

    #axq.set_xlim(xlimits)
    axq.set_ylim(ylim1)
    axq.set_xlabel("CANDELS $F160W$")
    axq.legend(loc='upper left', frameon=False) #loc=2


    # TOP RIGHT
    axr = figM.add_subplot(2,2,2)

    axr.hist(gzmm['SB_AUTO'][smooth_select & np.invert(gzc_nanmag)],   histtype='step', color='red', label='Smooth', bins=thebinssb)
    axr.hist(gzmm['SB_AUTO'][featured_select & np.invert(gzc_nanmag)], weights=np.ones_like(gzmm['dM_V'][featured_select & np.invert(gzc_nanmag)])*10., histtype='step', color='blue', label='Featured ($\\times 10$)', bins=thebinssb)

    #axr.hist(sdss['TOTAL_MAG_G_1'][sdss_smooth & np.invert(sdss_nanmag)],   histtype='step', color='red', label='Smooth', linestyle='dotted', linewidth=2., bins=thebins)
    #axr.hist(sdss['TOTAL_MAG_G_1'][sdss_featured & np.invert(sdss_nanmag)], histtype='step', color='blue', label='Featured', bins=thebins)

    ylim2 = (0., 490.)

    #axr.set_xlim(xlimits)
    axr.set_ylim(ylim2)
    axr.set_xlabel("$\mu_{SB}$")
    axr.legend(loc='upper left', frameon=False) #loc=2




    # BOTTOM LEFT
    axs = figM.add_subplot(2,2,3)

    axs.hist(gzmm['WFC3_F160W_MAG'][smooth_select & np.invert(gzc_nanmag)],   weights=gzmm['Mweight_smooth'][smooth_select & np.invert(gzc_nanmag)],   histtype='step', color='red', label='Weighted Smooth', bins=thebins)
    axs.hist(gzmm['WFC3_F160W_MAG'][featured_select & np.invert(gzc_nanmag)], weights=gzmm['Mweight_feat'][featured_select & np.invert(gzc_nanmag)], histtype='step', color='blue', label='Weighted Featured', bins=thebins)
    #axs.hist(sdss['TOTAL_MAG_G_1'][sdss_smooth & np.invert(sdss_nanmag)],   weights=sdss['Mweight_smooth'][sdss_smooth & np.invert(sdss_nanmag)],   histtype='step', color='red', label='SDSS Smooth', linestyle='dashed', bins=thebins)


    ylim3 = (0., 29.)

    #axs.set_xlim(xlimits)
    axs.set_ylim(ylim3)
    axs.set_xlabel("$F160W$")
    axs.legend(loc='upper left', frameon=False) #loc=2


    # BOTTOM RIGHT
    axt = figM.add_subplot(2,2,4)

    axt.hist(gzmm['SB_AUTO'][smooth_select & np.invert(gzc_nanmag)],   weights=gzmm['Mweight_smooth'][smooth_select & np.invert(gzc_nanmag)],   histtype='step', color='red', label='Weighted Smooth', bins=thebinssb)
    axt.hist(gzmm['SB_AUTO'][featured_select & np.invert(gzc_nanmag)], weights=gzmm['Mweight_feat'][featured_select & np.invert(gzc_nanmag)], histtype='step', color='blue', label='Weighted Featured', bins=thebinssb)
    #axt.hist(sdss['TOTAL_MAG_G_1'][sdss_featured & np.invert(sdss_nanmag)], weights=sdss['Mweight_feat'][sdss_featured & np.invert(sdss_nanmag)], histtype='step', color='blue', label='SDSS Featured', linestyle='dashed', bins=thebins)

    ylim4 = (0., 34.)

    #axt.set_xlim(xlimits)
    axt.set_ylim(ylim4)
    axt.set_xlabel("$\mu_{SB}$")
    axt.legend(loc='upper left', frameon=False) #loc=2





    plt.tight_layout()

    plt.savefig('mag_histogram_all_x10a.png', facecolor='None', edgecolor='None')
    # plt.savefig('dM_histogram_adj.eps', facecolor='None', edgecolor='None')

    plt.clf()
    plt.cla()
    plt.close('All')
    plt.close()
    plt.close()
    plt.close()
    plt.close()












#booya
