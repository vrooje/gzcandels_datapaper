import sys
import numpy as np
import pandas as pd
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize
import functools
from scipy.interpolate import interp1d
from scipy.stats.stats import pearsonr, spearmanr

from collections import Counter
#from pymongo import MongoClient

try:
    from astropy.io import fits as pyfits
    from astropy.io.fits import Column
    from astropy.io import ascii
except ImportError:
    import pyfits
    from pyfits import Column

path_class = './'


def plot_class_viscandels():
    matched_class_file = '/Users/vrooje/Astro/Zooniverse/gz_reduction_sandbox/data/gz_candelsteam_ukmc_visclass_compare_all_radecmatch_gds_uds.csv'
    matched_class_all = pd.read_csv(matched_class_file,low_memory=False)

    # select based on surface brightness etc
    #For all comparisons below we have compared the subset of sources in CANDELS above the surface brightness limit mu_SB < 24.2 which have visual classifications from both teams, which have not been deemed “unclassifiable” by the CANDELS team, and which have not been rejected as stars or artifacts by more than 50% of classifiers for either project.

    to_keep = (matched_class_all.SB_AUTO < 24.5) & (matched_class_all.f_Unc < 0.3) & (matched_class_all.f_PS < 0.5) & (matched_class_all.t00_smooth_or_featured_a2_artifact_weighted_frac < 0.5)

    matched_class = matched_class_all[to_keep]

    # note we've already removed likely artifacts from the file above, otherwise we'd add that as a requirement below
    is_smooth = matched_class.t00_smooth_or_featured_a0_smooth_weighted_frac >= 0.3
    is_featured = matched_class.t00_smooth_or_featured_a1_features_weighted_frac >= 0.3
    is_clumpy = matched_class.t02_clumpy_appearance_a0_yes_weighted_frac >= 0.7 # this is really really conservative but matches S14
    is_edge_on = matched_class.t09_disk_edge_on_a0_yes_weighted_frac >= 0.5
    not_clumpy = np.invert(is_clumpy)
    not_edge_on = np.invert(is_edge_on)

    is_featured_clumpy                = is_featured & is_clumpy
    is_featured_not_clumpy            = is_featured & not_clumpy
    is_featured_not_clumpy_edgeon     = is_featured_not_clumpy & is_edge_on
    is_featured_not_clumpy_not_edgeon = is_featured_not_clumpy & not_edge_on

    # candels team morphologies have between 3 and 7 people looking at each galaxy
    # so the vote fractions that are possible are limited to any fractions out of 3-7
    #f3 = np.arange(0,4)/3.
    #f4 = np.arange(1,4)/4.
    #f5 = np.arange(1,5)/5.
    #f6 = np.arange(1,6)/6.
    #f7 = np.arange(1,7)/7.

    #f_all = np.concatenate((f3,f4,f5,f6,f7))
    #f_all.sort()



    candels_int      = matched_class.f_merger + matched_class.f_Int1 + matched_class.f_Int2 + matched_class.f_Tidal #+matched_class.f_Tadpole
    # this thing with the clumpy/patchy vote is not quite the same as but similar to Guo et al 2014's comparison with Kartaltepe et al 2014
    candels_clumpy   = (0.25*matched_class.f_C0P1) + (0.5*matched_class.f_C0P2) + (0.25*matched_class.f_C1P0) + (0.5*matched_class.f_C1P1) + (0.75*matched_class.f_C1P2) + (0.5*matched_class.f_C2P0) + (0.75*matched_class.f_C2P1) + matched_class.f_C2P2
    candels_clumpy = candels_clumpy/np.max(candels_clumpy)
    candels_featured = matched_class.f_Bar + matched_class.f_Arms + candels_clumpy
    # instead of summing, use the max featured vote for any given feature, which is theoretically more similar to GZ
    #candels_featured = np.maximum(np.maximum(matched_class.f_Bar, matched_class.f_Arms), candels_clumpy)
    #candels_featured += matched_class.f_Asym + matched_class.f_Tidal + matched_class.f_Chain + matched_class.f_Tadpole + matched_class.f_Irr
    candels_edgeon   = matched_class.f_Edgeon[is_featured_not_clumpy]
    candels_barred   = matched_class.f_Bar[is_featured_not_clumpy_not_edgeon]
    candels_spiral   = matched_class.f_Arms[is_featured_not_clumpy_not_edgeon]

    merger_or_interacting = np.array([matched_class.t16_merging_tidal_debris_a0_merging_weighted_frac + matched_class.t16_merging_tidal_debris_a1_tidal_debris_weighted_frac + matched_class.t16_merging_tidal_debris_a2_both_weighted_frac, candels_int, candels_int + np.random.uniform(-0.0125,0.0125,len(candels_int))])

    featured = np.array([matched_class.t00_smooth_or_featured_a1_features_weighted_frac, candels_featured, candels_featured + np.random.uniform(-0.0125,0.0125,len(candels_featured))])

    edge_on = np.array([matched_class.t09_disk_edge_on_a0_yes_weighted_frac[is_featured_not_clumpy], candels_edgeon, candels_edgeon + np.random.uniform(-0.0125,0.0125,len(candels_edgeon))])

    barred = np.array([matched_class.t11_bar_feature_a0_yes_weighted_frac[is_featured_not_clumpy_not_edgeon], candels_barred, candels_barred +  + np.random.uniform(-0.0125,0.0125,len(candels_barred))])

    spiral = np.array([matched_class.t12_spiral_pattern_a0_yes_weighted_frac[is_featured_not_clumpy_not_edgeon], candels_spiral, candels_spiral +  + np.random.uniform(-0.0125,0.0125,len(candels_spiral))])

    # define the eventual return data
    theavg   = 0
    thesig   = 1
    themed   = 2
    the5pct  = 3 #2 sigma (kinda), low
    the16pct = 4 #1 sigma, low
    the25pct = 5 #lower IQR
    the75pct = 6 #upper IQR
    the84pct = 7 #1 sigma, hi
    the95pct = 8 #2 sigma, hi
    miderr   = 9 #midpt btwn lo&hi 1-sigma
    theerr   = 10#half dist btwn lo&hi 1-sigma
    thecount = 11
    theavgx  = 12


    # Note the binning does < for the high bin so set the boundaries slightly higher
    gz_minmax      = [0., 1.001]
    feat_minmax    = [0., 2.501]
    merger_minmax  = [0., 2.001]
    edgeon_minmax  = [0., 1.001]
    spiral_minmax  = [0., 1.001]


    #################### Diagonal Binning - use with a LOT of caution ###################
    # In fact don't use it at all; it's a TERRIBLE IDEA that I needed to
    # try out, like really commit to, in order to prove to myself that
    # it is a TERRIBLE IDEA.
    # it makes uncorrelated data look like perfect 1:1 correlations.
    # seriously DO NOT DO THIS
    # (but at least you know, hence my leaving this here)

    featured_stats_x, featured_stats_y = diagbins_stats(featured[0], featured[1], [gz_minmax, feat_minmax], 10.)
    merger_stats_x, merger_stats_y     = diagbins_stats(merger_or_interacting[0], merger_or_interacting[1], [gz_minmax, merger_minmax], 10.)
    edgeon_stats_x, edgeon_stats_y     = diagbins_stats(edge_on[0], edge_on[1],   [gz_minmax, edgeon_minmax], 7.)
    spiral_stats_x, spiral_stats_y     = diagbins_stats(spiral[0], spiral[1],     [gz_minmax, spiral_minmax], 7.)
    #####################################################################################





    #####################   Normal Binning     ##########################################

    featured_statsbygz  = binned_stats(             featured[0],              featured[1], gz_minmax, 10.)
    merger_statsbygz    = binned_stats(merger_or_interacting[0], merger_or_interacting[1], gz_minmax, 10.)
    edgeon_statsbygz    = binned_stats(              edge_on[0],               edge_on[1], gz_minmax, 7.)
    spiral_statsbygz    = binned_stats(               spiral[0],                spiral[1], gz_minmax, 7.)

    featured_statsbyk12 = binned_stats(             featured[1],              featured[0], feat_minmax,   10.)
    merger_statsbyk12   = binned_stats(merger_or_interacting[1], merger_or_interacting[0], merger_minmax, 10.)
    edgeon_statsbyk12   = binned_stats(              edge_on[1],               edge_on[0], edgeon_minmax, 7.)
    spiral_statsbyk12   = binned_stats(               spiral[1],                spiral[0], spiral_minmax, 7.)

    #####################################################################################



    featured_r, featured_p = pearsonr(np.array(featured[0]),                          np.array(featured[1]))
    merger_r,     merger_p = pearsonr(np.array(merger_or_interacting[0]), np.array(merger_or_interacting[1]))
    edge_on_r,   edge_on_p = pearsonr(np.array(edge_on[0]),                            np.array(edge_on[1]))
    spiral_r,     spiral_p = pearsonr(np.array(spiral[0]),                              np.array(spiral[1]))


# Pearson r
# In [282]: print("%.4f %.2e (%d subjects)\n" % (featured_r, featured_p, len(featured[0])))
# 0.5679 0.00e+00 (13145 subjects)
#
# In [283]: print("%.4f %.2e (%d subjects)\n" % (merger_r, merger_p, len(merger_or_interacting[0])))
# 0.6717 0.00e+00 (13145 subjects)
#
# In [284]: print("%.4f %.2e (%d subjects)\n" % (edge_on_r, edge_on_p, len(edge_on[0])))
# 0.7771 0.00e+00 (1611 subjects)
#
# In [285]: print("%.4f %.2e (%d subjects)\n" % (spiral_r, spiral_p, len(spiral[0])))
# 0.6095 3.64e-122 (1192 subjects)

# Spearman r
# In [299]: spearmanr(featured[0], featured[1])
# Out[299]: (0.44550532412648242, 0.0)
#
# In [300]: spearmanr(merger_or_interacting[0], merger_or_interacting[1])
# Out[300]: (0.66843746334447618, 0.0)
#
# In [301]: spearmanr(edge_on[0], edge_on[1])
# Out[301]: (0.7192386952518216, 9.0519935179250988e-257)
#
# In [302]: spearmanr(spiral[0], spiral[1])
# Out[302]: (0.61284675890750606, 7.6620130174538219e-124)





    #####################################################################
    #####################################################################
    #####################################################################
    #####################################################################
    #####################################################################

    ptcolor_k12 = '#d30202' #slightly darker than fire-engine red

    ptcolor_gz = '#007aa3' #slightly darker than teal blue

    #which_plot = themed
    which_plot = theavg


    # ALL in subplots, with hexgrid to prevent huge eps files
    #plt.clf()
    #plt.cla()
    #plt.figure(1)
    #fig = plt.gcf()

    fig = plt.figure(figsize=(8, 8))

    ###########################################
    # FEATURED
    plt.subplot(221)

    xlimits = (-0.025,1.025)
    ylimits = (-0.05,2.5)

    plt.xlim(xlimits)
    plt.ylim(ylimits)

    # underlying points, binned in 2d
    plt.hexbin(featured[0], featured[1], gridsize=40, bins='log', cmap='Greys')

    # to plot normal binning
    # binned by k12
    plt.plot(featured_statsbyk12[which_plot][featured_statsbyk12[thecount] > 0], featured_statsbyk12[theavgx][featured_statsbyk12[thecount] > 0], marker='s', linestyle='-', color=ptcolor_k12, ms=10.)
    #plt.errorbar(featured_statsbyk12[which_plot], featured_statsbyk12[theavgx], xerr=featured_statsbyk12[thesig], ecolor=ptcolor_k12, linewidth=2, linestyle='None')
    plt.errorbar(featured_statsbyk12[miderr][featured_statsbyk12[thecount] > 0], featured_statsbyk12[theavgx][featured_statsbyk12[thecount] > 0], xerr=featured_statsbyk12[theerr][featured_statsbyk12[thecount] > 0], ecolor=ptcolor_k12, linewidth=2, linestyle='None')

    # binned by gz
    plt.plot(featured_statsbygz[theavgx][featured_statsbygz[thecount] > 0], featured_statsbygz[which_plot][featured_statsbygz[thecount] > 0], marker='o', linestyle='-', color=ptcolor_gz, ms=10.)
    #plt.errorbar(featured_statsbygz[theavgx], featured_statsbygz[which_plot], yerr=featured_statsbygz[thesig], ecolor=ptcolor_gz, linewidth=2, linestyle='None')
    plt.errorbar(featured_statsbygz[theavgx][featured_statsbygz[thecount] > 0], featured_statsbygz[miderr][featured_statsbygz[thecount] > 0], yerr=featured_statsbygz[theerr][featured_statsbygz[thecount] > 0], ecolor=ptcolor_gz, linewidth=2, linestyle='None')


#    # to plot diagonal binning
# J/K lolz no we're not. Diagonal binning is a TERRIBLE IDEA
# I'll leave this ONE part below here instead of deleting, but if you
# want to do this and plot this you'll have to transfer it to all the
# other plots yourself.
# but don't do it.
#    plt.plot(featured_stats_x[which_plot][featured_stats_x[thecount] > 0], featured_stats_y[which_plot][featured_stats_x[thecount] > 0], marker='None', linestyle='--', color=ptcolor_gz, ms=10.)
#    #plt.errorbar(featured_stats_x[miderr], featured_stats_y[miderr], xerr=featured_stats_x[theerr], yerr=featured_stats_y[theerr], ecolor=ptcolor_gz, linewidth=2, linestyle='None')

#    plt.plot(featured_stats_x[the16pct][featured_stats_x[thecount] > 0], featured_stats_y[the84pct][featured_stats_x[thecount] > 0], color=ptcolor_gz)
#    plt.plot(featured_stats_x[the84pct][featured_stats_x[thecount] > 0], featured_stats_y[the16pct][featured_stats_x[thecount] > 0], color=ptcolor_gz)


    plt.xlabel("Galaxy Zoo: Featured Vote Fraction")
    plt.ylabel("K14: (Bar + Spiral + Clumpy) Vote Fractions")
    plt.text(0.05, 2.35, '(a)', fontsize=14, va='top')
    cb = plt.colorbar()
    cb.set_label("log(N)")


    ###########################################
    # MERGER
    plt.subplot(222)

    xlimits = (-0.05,1.05)
    ylimits = (-0.1,2.1)

    plt.xlim(xlimits)
    plt.ylim(ylimits)

    # underlying points, binned in 2d
    plt.hexbin(merger_or_interacting[0], merger_or_interacting[1], gridsize=40, bins='log', cmap='Greys')

    # to plot normal binning
    # binned by k12
    plt.plot(merger_statsbyk12[which_plot][merger_statsbyk12[thecount] > 0], merger_statsbyk12[theavgx][merger_statsbyk12[thecount] > 0], marker='s', linestyle='-', color=ptcolor_k12, ms=10.)
#     plt.errorbar(merger_statsbyk12[which_plot], merger_statsbyk12[theavgx], xerr=merger_statsbyk12[thesig], ecolor=ptcolor_k12, linewidth=2, linestyle='None')
    plt.errorbar(merger_statsbyk12[miderr][merger_statsbyk12[thecount] > 0], merger_statsbyk12[theavgx][merger_statsbyk12[thecount] > 0], xerr=merger_statsbyk12[theerr][merger_statsbyk12[thecount] > 0], ecolor=ptcolor_k12, linewidth=2, linestyle='None')

    # binned by gz
    plt.plot(merger_statsbygz[theavgx][merger_statsbygz[thecount] > 0], merger_statsbygz[which_plot][merger_statsbygz[thecount] > 0], marker='o', linestyle='-', color=ptcolor_gz, ms=10.)
#     plt.errorbar(merger_statsbygz[theavgx], merger_statsbygz[which_plot], yerr=merger_statsbygz[thesig], ecolor=ptcolor_gz, linewidth=2, linestyle='None')
    plt.errorbar(merger_statsbygz[theavgx][merger_statsbygz[thecount] > 0], merger_statsbygz[miderr][merger_statsbygz[thecount] > 0], yerr=merger_statsbygz[theerr][merger_statsbygz[thecount] > 0], ecolor=ptcolor_gz, linewidth=2, linestyle='None')



    plt.xlabel("Galaxy Zoo: Merger or Interacting Vote Fraction")
    plt.ylabel("K14: Merger or Interacting Vote Fractions")
    plt.text(0.025, 1.96, '(b)', fontsize=14, va='top')
    cb = plt.colorbar()
    cb.set_label("log(N)")




    ###########################################
    # EDGE-ON
    plt.subplot(223)

    xlimits = (-0.05,1.05)
    ylimits = (-0.05,1.05)

    plt.xlim(xlimits)
    plt.ylim(ylimits)

    # underlying points, binned in 2d
    plt.hexbin(edge_on[0], edge_on[1], gridsize=40, bins='log', cmap='Greys')

    # to plot normal binning
    # binned by k12
    plt.plot(edgeon_statsbyk12[which_plot][edgeon_statsbyk12[thecount] > 0], edgeon_statsbyk12[theavgx][edgeon_statsbyk12[thecount] > 0], marker='s', linestyle='-', color=ptcolor_k12, ms=10.)
#     plt.errorbar(edgeon_statsbyk12[which_plot], edgeon_statsbyk12[theavgx], xerr=edgeon_statsbyk12[thesig], ecolor=ptcolor_k12, linewidth=2, linestyle='None')
    plt.errorbar(edgeon_statsbyk12[miderr][edgeon_statsbyk12[thecount] > 0], edgeon_statsbyk12[theavgx][edgeon_statsbyk12[thecount] > 0], xerr=edgeon_statsbyk12[theerr][edgeon_statsbyk12[thecount] > 0], ecolor=ptcolor_k12, linewidth=2, linestyle='None')

    # binned by gz
    plt.plot(edgeon_statsbygz[theavgx][edgeon_statsbygz[thecount] > 0], edgeon_statsbygz[which_plot][edgeon_statsbygz[thecount] > 0], marker='o', linestyle='-', color=ptcolor_gz, ms=10.)
#     plt.errorbar(edgeon_statsbygz[theavgx], edgeon_statsbygz[which_plot], yerr=edgeon_statsbygz[thesig], ecolor=ptcolor_gz, linewidth=2, linestyle='None')
    plt.errorbar(edgeon_statsbygz[theavgx][edgeon_statsbygz[thecount] > 0], edgeon_statsbygz[miderr][edgeon_statsbygz[thecount] > 0], yerr=edgeon_statsbygz[theerr][edgeon_statsbygz[thecount] > 0], ecolor=ptcolor_gz, linewidth=2, linestyle='None')




    plt.xlabel("Galaxy Zoo: Edge-On Vote Fraction")
    plt.ylabel("K14: Edge-On Vote Fraction")
    plt.text(0.025, 0.98, '(c)', fontsize=14, va='top')
    cb = plt.colorbar()
    cb.set_label("log(N)")




    ###########################################
    # SPIRAL
    plt.subplot(224)

    xlimits = (-0.05,1.05)
    ylimits = (-0.05,1.05)

    plt.xlim(xlimits)
    plt.ylim(ylimits)

    # underlying points, binned in 2d
    plt.hexbin(spiral[0], spiral[1], gridsize=40, bins='log', cmap='Greys')


    # to plot normal binning
    # binned by k12
    plt.plot(spiral_statsbyk12[which_plot][spiral_statsbyk12[thecount] > 0], spiral_statsbyk12[theavgx][spiral_statsbyk12[thecount] > 0], marker='s', linestyle='-', color=ptcolor_k12, ms=10.)
#     plt.errorbar(spiral_statsbyk12[which_plot], spiral_statsbyk12[theavgx], xerr=spiral_statsbyk12[thesig], ecolor=ptcolor_k12, linewidth=2, linestyle='None')
    plt.errorbar(spiral_statsbyk12[miderr][spiral_statsbyk12[thecount] > 0], spiral_statsbyk12[theavgx][spiral_statsbyk12[thecount] > 0], xerr=spiral_statsbyk12[theerr][spiral_statsbyk12[thecount] > 0], ecolor=ptcolor_k12, linewidth=2, linestyle='None')

    # binned by gz
    plt.plot(spiral_statsbygz[theavgx][spiral_statsbygz[thecount] > 0], spiral_statsbygz[which_plot][spiral_statsbygz[thecount] > 0], marker='o', linestyle='-', color=ptcolor_gz, ms=10.)
#     plt.errorbar(spiral_statsbygz[theavgx], spiral_statsbygz[which_plot], yerr=spiral_statsbygz[thesig], ecolor=ptcolor_gz, linewidth=2, linestyle='None')
    plt.errorbar(spiral_statsbygz[theavgx][spiral_statsbygz[thecount] > 0], spiral_statsbygz[miderr][spiral_statsbygz[thecount] > 0], yerr=spiral_statsbygz[theerr][spiral_statsbygz[thecount] > 0], ecolor=ptcolor_gz, linewidth=2, linestyle='None')



    plt.xlabel("Galaxy Zoo: Spiral Feature Vote Fraction")
    plt.ylabel("K14: Spiral Feature Vote Fraction")
    plt.text(0.025, 0.98, '(d)', fontsize=14, va='top')
    cb = plt.colorbar()
    cb.set_label("log(N)")


    plt.tight_layout()
    plt.show()


    plt.close()
    plt.cla()
    plt.clf()





def plot_class_visukmc():
    #matched_class_file = '/Users/vrooje/Astro/GalaxyZoo/GZ_CANDELS/GZcandels_reduction/gzcandels_candelsteam_match_onlygooddata.csv'
    #matched_class_file = '/Users/vrooje/Astro/Zooniverse/gz_reduction_sandbox/data/gz_candelsteam_visclass_compare_all_radecmatch_gds_uds.csv'
    matched_class_file = '/Users/vrooje/Astro/Zooniverse/gz_reduction_sandbox/data/gz_candelsteam_ukmc_visclass_compare_all_radecmatch_gds_uds.csv'
    matched_class_all = pd.read_csv(matched_class_file,low_memory=False)

    # select based on surface brightness etc
    #For all comparisons below we have compared the subset of sources in CANDELS above the surface brightness limit mu_SB < 24.2 which have visual classifications from both teams, which have not been deemed “unclassifiable” by the CANDELS team, and which have not been rejected as stars or artifacts by more than 50% of classifiers for either project.

    to_keep = (matched_class_all.SB_AUTO < 24.5) & (matched_class_all.f_PS < 0.5) & (matched_class_all.t00_smooth_or_featured_a2_artifact_weighted_frac < 0.5) & (matched_class_all.UNw<0.35) & (matched_class_all.Qw>0.65)

    matched_class = matched_class_all[to_keep]

    # note we've already removed likely artifacts from the file above, otherwise we'd add that as a requirement below
    is_smooth = matched_class.t00_smooth_or_featured_a0_smooth_weighted_frac >= 0.3
    is_featured = matched_class.t00_smooth_or_featured_a1_features_weighted_frac >= 0.3
    is_clumpy = matched_class.t02_clumpy_appearance_a0_yes_weighted_frac >= 0.7 # this is really really conservative but matches S14
    is_edge_on = matched_class.t09_disk_edge_on_a0_yes_weighted_frac >= 0.5
    not_clumpy = np.invert(is_clumpy)
    not_edge_on = np.invert(is_edge_on)

    is_featured_clumpy                = is_featured & is_clumpy
    is_featured_not_clumpy            = is_featured & not_clumpy
    is_featured_not_clumpy_edgeon     = is_featured_not_clumpy & is_edge_on
    is_featured_not_clumpy_not_edgeon = is_featured_not_clumpy & not_edge_on
    potential_spiral                  = is_featured_not_clumpy_not_edgeon & (matched_class.t12_spiral_pattern_weight > 10.)
    is_disky_ok                       = matched_class.DSw > 0.65
    is_featured_disky                 = is_featured & is_disky_ok
    is_spiral_disky                   = potential_spiral & is_disky_ok
    is_edgeon_disky                   = is_featured_not_clumpy & is_disky_ok





    candels_disky    = matched_class.Dv
    candels_int      = matched_class.MIv2
    candels_edgeon   = matched_class.EOw
    candels_barred   = matched_class.Bw[is_featured_not_clumpy_not_edgeon]
    candels_spiral   = matched_class.SSw
    candels_disturb  = matched_class.IPw

    merger_or_interacting = np.array([matched_class.t16_merging_tidal_debris_a0_merging_weighted_frac + matched_class.t16_merging_tidal_debris_a1_tidal_debris_weighted_frac + matched_class.t16_merging_tidal_debris_a2_both_weighted_frac, candels_int, candels_int + np.random.uniform(-0.0125,0.0125,len(candels_int))])

    merger_disturbed = np.array([matched_class.t16_merging_tidal_debris_a0_merging_weighted_frac + matched_class.t16_merging_tidal_debris_a1_tidal_debris_weighted_frac + matched_class.t16_merging_tidal_debris_a2_both_weighted_frac, candels_disturb, candels_disturb + np.random.uniform(-0.0125,0.0125,len(candels_disturb))])

    disky_featured = np.array([matched_class.t00_smooth_or_featured_a1_features_weighted_frac[is_disky_ok], candels_disky[is_disky_ok], candels_disky[is_disky_ok] + np.random.uniform(-0.0125,0.0125,len(candels_disky[is_disky_ok]))])

    disky_spiral = np.array([matched_class.t12_spiral_pattern_a0_yes_weighted_frac[is_spiral_disky], candels_disky[is_spiral_disky], candels_disky[is_spiral_disky] +  np.random.uniform(-0.0125,0.0125,len(candels_disky[is_spiral_disky]))])

    edgeon = np.array([matched_class.t09_disk_edge_on_a0_yes_weighted_frac[is_featured_not_clumpy], candels_edgeon[is_featured_not_clumpy], candels_edgeon[is_featured_not_clumpy] + np.random.uniform(-0.0125,0.0125,len(candels_edgeon[is_featured_not_clumpy]))])

    disky_edgeon = np.array([matched_class.t09_disk_edge_on_a0_yes_weighted_frac[is_edgeon_disky], candels_disky[is_edgeon_disky], candels_disky[is_edgeon_disky] + np.random.uniform(-0.0125,0.0125,len(candels_disky[is_edgeon_disky]))])

    barred = np.array([matched_class.t11_bar_feature_a0_yes_weighted_frac[is_featured_not_clumpy_not_edgeon], candels_barred, candels_barred + np.random.uniform(-0.0125,0.0125,len(candels_barred))])

    spiral = np.array([matched_class.t12_spiral_pattern_a0_yes_weighted_frac[potential_spiral], candels_spiral[potential_spiral], candels_spiral[potential_spiral] + np.random.uniform(-0.0125,0.0125,len(candels_spiral[potential_spiral]))])

    # define the eventual return data
    theavg   = 0
    thesig   = 1
    themed   = 2
    the5pct  = 3 #2 sigma (kinda), low
    the16pct = 4 #1 sigma, low
    the25pct = 5 #lower IQR
    the75pct = 6 #upper IQR
    the84pct = 7 #1 sigma, hi
    the95pct = 8 #2 sigma, hi
    miderr   = 9 #midpt btwn lo&hi 1-sigma
    theerr   = 10#half dist btwn lo&hi 1-sigma
    thecount = 11
    theavgx  = 12


    # Note the binning does < for the high bin so set the boundaries slightly higher
    gz_minmax      = [0., 1.001]
    feat_minmax    = [0., 2.501]
    merger_minmax  = [0., 2.001]
    edgeon_minmax  = [0., 1.001]
    spiral_minmax  = [0., 1.001]





    #####################   Normal Binning     ##########################################

    disky_featured_statsbygz  = binned_stats( disky_featured[0],        disky_featured[1], gz_minmax, 10.)
    disky_spiral_statsbygz = binned_stats(      disky_spiral[0],          disky_spiral[1], gz_minmax, 10.)
    disky_edgeon_statsbygz = binned_stats(      disky_edgeon[0],          disky_edgeon[1], gz_minmax, 7.)
    merger_statsbygz    = binned_stats(merger_or_interacting[0], merger_or_interacting[1], gz_minmax, 10.)
    disturb_statsbygz   = binned_stats(     merger_disturbed[0],      merger_disturbed[1], gz_minmax, 10.)
    edgeon_statsbygz    = binned_stats(               edgeon[0],                edgeon[1], gz_minmax, 7.)
    spiral_statsbygz    = binned_stats(               spiral[0],                spiral[1], gz_minmax, 7.)

    disky_featured_statsbyk12 = binned_stats( disky_featured[1],        disky_featured[0], gz_minmax,   10.)
    disky_spiral_statsbyk12 = binned_stats(     disky_spiral[1],          disky_spiral[0], gz_minmax,   10.)
    disky_edgeon_statsbyk12 = binned_stats(     disky_edgeon[1],          disky_edgeon[0], gz_minmax,   10.)
    merger_statsbyk12   = binned_stats(merger_or_interacting[1], merger_or_interacting[0], gz_minmax, 10.)
    disturb_statsbyk12  = binned_stats(     merger_disturbed[1],      merger_disturbed[0], gz_minmax, 10.)
    edgeon_statsbyk12   = binned_stats(               edgeon[1],                edgeon[0], edgeon_minmax, 7.)
    spiral_statsbyk12   = binned_stats(               spiral[1],                spiral[0], spiral_minmax, 7.)

    #####################################################################################



    disky_featured_r, disky_featured_p = pearsonr(np.array(disky_featured[0]),                          np.array(disky_featured[1]))
    disky_spiral_r,     disky_spiral_p = pearsonr(np.array(disky_spiral[0]),                              np.array(disky_spiral[1]))
    disky_edgeon_r,     disky_edgeon_p = pearsonr(np.array(disky_edgeon[0]),                              np.array(disky_edgeon[1]))
    merger_r,     merger_p             = pearsonr(np.array(merger_or_interacting[0]), np.array(merger_or_interacting[1]))
    merger_disturbed_r, merger_disturbed_p = pearsonr(np.array(merger_disturbed[0]), np.array(merger_disturbed[1]))
    edgeon_r,   edgeon_p               = pearsonr(np.array(edgeon[0]),                            np.array(edgeon[1]))
    spiral_r,     spiral_p             = pearsonr(np.array(spiral[0]),                              np.array(spiral[1]))





    #####################################################################
    #####################################################################
    #####################################################################
    #####################################################################
    #####################################################################

    ptcolor_k12 = '#d30202' #slightly darker than fire-engine red

    ptcolor_gz = '#007aa3' #slightly darker than teal blue

    #which_plot = themed
    which_plot = theavg



    xlimits = (-0.025, 1.025)
    ylimits = xlimits

    fig = plt.figure(figsize=(11, 8))
    gs = gridspec.GridSpec(2, 2)
    gs.update(hspace=0.25, wspace=0.3)



    # GZ spiral vs CANDELS disky
    ax1 = fig.add_subplot(gs[0,0])
    # it should be ax1 not plt below but if I do that I can't get the colorbar to work
    plt.hexbin(disky_spiral[0], disky_spiral[1], gridsize=40, bins='log', cmap='Greys')

    # to plot normal binning
    # binned by k12
    plt.plot(disky_spiral_statsbyk12[which_plot][disky_spiral_statsbyk12[thecount] > 0], disky_spiral_statsbyk12[theavgx][disky_spiral_statsbyk12[thecount] > 0], marker='s', linestyle='-', color=ptcolor_k12, ms=10.)
    #plt.errorbar(featured_statsbyk12[which_plot], featured_statsbyk12[theavgx], xerr=featured_statsbyk12[thesig], ecolor=ptcolor_k12, linewidth=2, linestyle='None')
    plt.errorbar(disky_spiral_statsbyk12[miderr][disky_spiral_statsbyk12[thecount] > 0], disky_spiral_statsbyk12[theavgx][disky_spiral_statsbyk12[thecount] > 0], xerr=disky_spiral_statsbyk12[theerr][disky_spiral_statsbyk12[thecount] > 0], ecolor=ptcolor_k12, linewidth=2, linestyle='None')

    # binned by gz
    plt.plot(disky_spiral_statsbygz[theavgx][disky_spiral_statsbygz[thecount] > 0], disky_spiral_statsbygz[which_plot][disky_spiral_statsbygz[thecount] > 0], marker='o', linestyle='-', color=ptcolor_gz, ms=10.)
    #plt.errorbar(featured_statsbygz[theavgx], featured_statsbygz[which_plot], yerr=featured_statsbygz[thesig], ecolor=ptcolor_gz, linewidth=2, linestyle='None')
    plt.errorbar(disky_spiral_statsbygz[theavgx][disky_spiral_statsbygz[thecount] > 0], disky_spiral_statsbygz[miderr][disky_spiral_statsbygz[thecount] > 0], yerr=disky_spiral_statsbygz[theerr][disky_spiral_statsbygz[thecount] > 0], ecolor=ptcolor_gz, linewidth=2, linestyle='None')

    ax1.set_xlim(xlimits)
    ax1.set_ylim(ylimits)
    ax1.set_xlabel("Galaxy Zoo: Spiral Vote Fraction")
    ax1.set_ylabel("CANDELS Diskiness Value")
    cb1 = plt.colorbar()
    cb1.set_label("log(N)")



    # GZ Edge-on vs CANDELS disky
    ax2 = fig.add_subplot(gs[0,1])
    # it should be ax1 not plt below but if I do that I can't get the colorbar to work
    plt.hexbin(disky_edgeon[0], disky_edgeon[1], gridsize=40, bins='log', cmap='Greys')
    # to plot normal binning
    # binned by k12
    plt.plot(disky_edgeon_statsbyk12[which_plot][disky_edgeon_statsbyk12[thecount] > 0], disky_edgeon_statsbyk12[theavgx][disky_edgeon_statsbyk12[thecount] > 0], marker='s', linestyle='-', color=ptcolor_k12, ms=10.)
    #plt.errorbar(featured_statsbyk12[which_plot], featured_statsbyk12[theavgx], xerr=featured_statsbyk12[thesig], ecolor=ptcolor_k12, linewidth=2, linestyle='None')
    plt.errorbar(disky_edgeon_statsbyk12[miderr][disky_edgeon_statsbyk12[thecount] > 0], disky_edgeon_statsbyk12[theavgx][disky_edgeon_statsbyk12[thecount] > 0], xerr=disky_edgeon_statsbyk12[theerr][disky_edgeon_statsbyk12[thecount] > 0], ecolor=ptcolor_k12, linewidth=2, linestyle='None')

    # binned by gz
    plt.plot(disky_edgeon_statsbygz[theavgx][disky_edgeon_statsbygz[thecount] > 0], disky_edgeon_statsbygz[which_plot][disky_edgeon_statsbygz[thecount] > 0], marker='o', linestyle='-', color=ptcolor_gz, ms=10.)
    #plt.errorbar(featured_statsbygz[theavgx], featured_statsbygz[which_plot], yerr=featured_statsbygz[thesig], ecolor=ptcolor_gz, linewidth=2, linestyle='None')
    plt.errorbar(disky_edgeon_statsbygz[theavgx][disky_edgeon_statsbygz[thecount] > 0], disky_edgeon_statsbygz[miderr][disky_edgeon_statsbygz[thecount] > 0], yerr=disky_edgeon_statsbygz[theerr][disky_edgeon_statsbygz[thecount] > 0], ecolor=ptcolor_gz, linewidth=2, linestyle='None')
    ax2.set_xlim(xlimits)
    ax2.set_ylim(ylimits)
    ax2.set_xlabel("Galaxy Zoo: Edge-on Vote Fraction")
    ax2.set_ylabel("CANDELS Diskiness Value")
    cb2 = plt.colorbar()
    cb2.set_label("log(N)")


    # GZ merger vs weighted CANDELS merger
    ax3 = fig.add_subplot(gs[1,0])
    # it should be ax1 not plt below but if I do that I can't get the colorbar to work
    plt.hexbin(merger_or_interacting[0], merger_or_interacting[1], gridsize=40, bins='log', cmap='Greys')
    # to plot normal binning
    # binned by k12
    plt.plot(merger_statsbyk12[which_plot][merger_statsbyk12[thecount] > 0], merger_statsbyk12[theavgx][merger_statsbyk12[thecount] > 0], marker='s', linestyle='-', color=ptcolor_k12, ms=10.)
    #plt.errorbar(featured_statsbyk12[which_plot], featured_statsbyk12[theavgx], xerr=featured_statsbyk12[thesig], ecolor=ptcolor_k12, linewidth=2, linestyle='None')
    plt.errorbar(merger_statsbyk12[miderr][merger_statsbyk12[thecount] > 0], merger_statsbyk12[theavgx][merger_statsbyk12[thecount] > 0], xerr=merger_statsbyk12[theerr][merger_statsbyk12[thecount] > 0], ecolor=ptcolor_k12, linewidth=2, linestyle='None')

    # binned by gz
    plt.plot(merger_statsbygz[theavgx][merger_statsbygz[thecount] > 0], merger_statsbygz[which_plot][merger_statsbygz[thecount] > 0], marker='o', linestyle='-', color=ptcolor_gz, ms=10.)
    #plt.errorbar(featured_statsbygz[theavgx], featured_statsbygz[which_plot], yerr=featured_statsbygz[thesig], ecolor=ptcolor_gz, linewidth=2, linestyle='None')
    plt.errorbar(merger_statsbygz[theavgx][merger_statsbygz[thecount] > 0], merger_statsbygz[miderr][merger_statsbygz[thecount] > 0], yerr=merger_statsbygz[theerr][merger_statsbygz[thecount] > 0], ecolor=ptcolor_gz, linewidth=2, linestyle='None')
    ax3.set_xlim(xlimits)
    ax3.set_ylim(ylimits)
    ax3.set_xlabel("Galaxy Zoo: Merger or Interacting Vote Fraction")
    ax3.set_ylabel("CANDELS Merger or Interacting Value")
    cb3 = plt.colorbar()
    cb3.set_label("log(N)")

    ax4 = fig.add_subplot(gs[1,1])
    # it should be ax1 not plt below but if I do that I can't get the colorbar to work
    plt.hexbin(merger_disturbed[0], merger_disturbed[1], gridsize=40, bins='log', cmap='Greys')
    # to plot normal binning
    # binned by k12
    plt.plot(disturb_statsbyk12[which_plot][disturb_statsbyk12[thecount] > 0], disturb_statsbyk12[theavgx][disturb_statsbyk12[thecount] > 0], marker='s', linestyle='-', color=ptcolor_k12, ms=10.)
    #plt.errorbar(featured_statsbyk12[which_plot], featured_statsbyk12[theavgx], xerr=featured_statsbyk12[thesig], ecolor=ptcolor_k12, linewidth=2, linestyle='None')
    plt.errorbar(disturb_statsbyk12[miderr][disturb_statsbyk12[thecount] > 0], disturb_statsbyk12[theavgx][disturb_statsbyk12[thecount] > 0], xerr=disturb_statsbyk12[theerr][disturb_statsbyk12[thecount] > 0], ecolor=ptcolor_k12, linewidth=2, linestyle='None')

    # binned by gz
    plt.plot(disturb_statsbygz[theavgx][disturb_statsbygz[thecount] > 0], disturb_statsbygz[which_plot][disturb_statsbygz[thecount] > 0], marker='o', linestyle='-', color=ptcolor_gz, ms=10.)
    #plt.errorbar(featured_statsbygz[theavgx], featured_statsbygz[which_plot], yerr=featured_statsbygz[thesig], ecolor=ptcolor_gz, linewidth=2, linestyle='None')
    plt.errorbar(disturb_statsbygz[theavgx][disturb_statsbygz[thecount] > 0], disturb_statsbygz[miderr][disturb_statsbygz[thecount] > 0], yerr=disturb_statsbygz[theerr][disturb_statsbygz[thecount] > 0], ecolor=ptcolor_gz, linewidth=2, linestyle='None')
    ax4.set_xlim(xlimits)
    ax4.set_ylim(ylimits)
    ax4.set_xlabel("Galaxy Zoo: Merger or Interacting Vote Fraction")
    ax4.set_ylabel("CANDELS Irregular/Peculiar Weight")
    cb4 = plt.colorbar()
    cb4.set_label("log(N)")



    #plt.show()
    plt.savefig('weighted_visclass_gz_candels.png', facecolor='None', edgecolor='None')
    plt.savefig('weighted_visclass_gz_candels.eps', facecolor='None', edgecolor='None')
    plt.close('All')
    plt.cla()
    plt.clf()



    ############################################################################    ############################################################################    ##############################   SECOND PLOT   #############################    ############################################################################    ############################################################################



    xlimits = (-0.025, 1.025)
    ylimits = xlimits

    fig = plt.figure(figsize=(5, 4))
    gs = gridspec.GridSpec(1, 1)
    #gs.update(hspace=0.25, wspace=0.001)



    # GZ featured vs CANDELS disky
    ax1 = fig.add_subplot(gs[0,0])
    # it should be ax1 not plt below but if I do that I can't get the colorbar to work
    plt.hexbin(disky_featured[0], disky_featured[1], gridsize=40, bins='log', cmap='Greys')

    # to plot normal binning
    # binned by k12
    plt.plot(disky_featured_statsbyk12[which_plot][disky_featured_statsbyk12[thecount] > 0], disky_featured_statsbyk12[theavgx][disky_featured_statsbyk12[thecount] > 0], marker='s', linestyle='-', color=ptcolor_k12, ms=10.)
    #plt.errorbar(featured_statsbyk12[which_plot], featured_statsbyk12[theavgx], xerr=featured_statsbyk12[thesig], ecolor=ptcolor_k12, linewidth=2, linestyle='None')
    plt.errorbar(disky_featured_statsbyk12[miderr][disky_featured_statsbyk12[thecount] > 0], disky_featured_statsbyk12[theavgx][disky_featured_statsbyk12[thecount] > 0], xerr=disky_featured_statsbyk12[theerr][disky_featured_statsbyk12[thecount] > 0], ecolor=ptcolor_k12, linewidth=2, linestyle='None')

    # binned by gz
    plt.plot(disky_featured_statsbygz[theavgx][disky_featured_statsbygz[thecount] > 0], disky_featured_statsbygz[which_plot][disky_featured_statsbygz[thecount] > 0], marker='o', linestyle='-', color=ptcolor_gz, ms=10.)
    #plt.errorbar(featured_statsbygz[theavgx], featured_statsbygz[which_plot], yerr=featured_statsbygz[thesig], ecolor=ptcolor_gz, linewidth=2, linestyle='None')
    plt.errorbar(disky_featured_statsbygz[theavgx][disky_featured_statsbygz[thecount] > 0], disky_featured_statsbygz[miderr][disky_featured_statsbygz[thecount] > 0], yerr=disky_featured_statsbygz[theerr][disky_featured_statsbygz[thecount] > 0], ecolor=ptcolor_gz, linewidth=2, linestyle='None')

    ax1.set_xlim(xlimits)
    ax1.set_ylim(ylimits)
    ax1.set_xlabel("Galaxy Zoo: Featured Vote Fraction")
    ax1.set_ylabel("CANDELS Diskiness Value")
    cb1 = plt.colorbar()
    cb1.set_label("log(N)")



    #plt.show()
    plt.tight_layout()
    plt.savefig('weighted_visclass_gz_candels_diskvsfeat.png', facecolor='None', edgecolor='None')
    plt.savefig('weighted_visclass_gz_candels_diskvsfeat.eps', facecolor='None', edgecolor='None')
    plt.close('All')
    plt.cla()
    plt.clf()












def binned_stats(x, y, minmax, nbins=10., running=False, sampling=10.):
    # x,y are the datasets in x and y
    #   x is what you're binning by, y is what you're computing binned stats on
    # minmax is: [xmin, xmax]
    # nbins is number of bins (e.g. 10)
    # running=False if you want strictly nbins non-overlapping bins
    #        =True if you want nbins to determine binsize but to compute nbins*sampling points via running (overlapping) binning

    if running:
        nbins_tot = nbins*sampling
    else:
        nbins_tot = nbins

    # define the eventual return data
    theavg   = 0
    thesig   = 1
    themed   = 2
    the5pct  = 3 #2 sigma (kinda), low
    the16pct = 4 #1 sigma, low
    the25pct = 5 #lower IQR
    the75pct = 6 #upper IQR
    the84pct = 7 #1 sigma, hi
    the95pct = 8 #2 sigma, hi
    miderr   = 9 #midpt btwn lo&hi 1-sigma
    theerr   = 10#half dist btwn lo&hi 1-sigma
    thecount = 11
    theavgx  = 12

    the_empty = np.arange(0.,1.,1./nbins_tot)*0.0
    the_stats = np.array([the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy()])
    #the_stats = np.array([the_stats_x, the_stats_y])


    # make this easier to read
    xmin = float(minmax[0])
    xmax = float(minmax[1])

    # this uses nbins regardless of whether running=True or False
    x_binwidth = (xmax - xmin)/float(nbins)

    # if running=True this will be much smaller
    x_step = (xmax - xmin)/float(nbins_tot)

    x_bins  = np.arange(xmin, xmax, x_binwidth) + x_binwidth/2.

    x_binlo = x_bins - x_binwidth/2.

    x_binhi = x_bins + x_binwidth/2.

    for i, bin in enumerate(the_empty):

        # get boundaries for this bin
        binlo = x_binlo[0] + (i*x_step)
        binhi = x_binhi[0] + (i*x_step)

        inbin = (x >= binlo) & (x < binhi)

        if sum(inbin) > 0:
            this_data = y[inbin]

            the_stats[theavgx][i]  =       np.mean(x[inbin])

            the_stats[theavg][i]   =       np.mean(this_data)
            the_stats[thesig][i]   =        np.std(this_data)
            the_stats[themed][i]   =     np.median(this_data)
            the_stats[the5pct][i]  = np.percentile(this_data, 5)
            the_stats[the16pct][i] = np.percentile(this_data, 16)
            the_stats[the25pct][i] = np.percentile(this_data, 25)
            the_stats[the75pct][i] = np.percentile(this_data, 75)
            the_stats[the84pct][i] = np.percentile(this_data, 84)
            the_stats[the95pct][i] = np.percentile(this_data, 95)
            the_stats[thecount][i] =           len(this_data)
            the_stats[miderr][i]   = 0.5*(the_stats[the16pct][i]+the_stats[the84pct][i])
            the_stats[theerr][i]   = 0.5*(the_stats[the84pct][i] - the_stats[the16pct][i])

        else:
            the_stats[theavgx][i]  = 0.5*(binlo + binhi)

            the_stats[theavg][i]   = 999.
            the_stats[thesig][i]   = 999.
            the_stats[themed][i]   = 999.
            the_stats[the5pct][i]  = 999.
            the_stats[the16pct][i] = 999.
            the_stats[the25pct][i] = 999.
            the_stats[the75pct][i] = 999.
            the_stats[the84pct][i] = 999.
            the_stats[the95pct][i] = 999.
            the_stats[thecount][i] = 0
            the_stats[miderr][i]   = 999.
            the_stats[theerr][i]   = 999.



    return the_stats













def diagbins_stats(x, y, minmax, nbins=10., running=False, sampling=10.):
    # x,y are the datasets in x and y
    # minmax is: [[xmin, xmax], [ymin, ymax]]
    # nbins is number of bins (e.g. 10)
    # running=False if you want strictly nbins non-overlapping bins
    #        =True if you want nbins to determine binsize but to compute nbins*sampling points via running (overlapping) binning

    if running:
        nbins_tot = nbins*sampling
    else:
        nbins_tot = nbins

    # define the eventual return data
    theavg   = 0
    thesig   = 1
    themed   = 2
    the5pct  = 3 #2 sigma (kinda), low
    the16pct = 4 #1 sigma, low
    the25pct = 5 #lower IQR
    the75pct = 6 #upper IQR
    the84pct = 7 #1 sigma, hi
    the95pct = 8 #2 sigma, hi
    miderr   = 9 #midpt btwn lo&hi 1-sigma
    theerr   = 10#half dist btwn lo&hi 1-sigma
    thecount = 11

    the_empty = np.arange(0.,1.,1./nbins_tot)*0.0
    the_stats_x = np.array([the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy()])
    the_stats_y = np.array([the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy(), the_empty.copy()])
    #the_stats = np.array([the_stats_x, the_stats_y])


    # make this easier to read
    xmin = float(minmax[0][0])
    xmax = float(minmax[0][1])
    ymin = float(minmax[1][0])
    ymax = float(minmax[1][1])


    # the slope is -45 degrees on the plot but that depends on the values along each axis
    # we can use (xmin, ymax) and (xmax, ymin) for the slope -- note all the slopes are the same
    slope  = (ymax-ymin)/(xmin-xmax)
    print("Slope: %.3f\n" % slope)

    # define low and high bins in x and y
    # the boundaries we want are to make nbins stripes within the range (xmin, ymin) to (xmax, ymax)
    # so figure out what those x and y intercepts are given the slope of the lines

    # point-slope form: y = m(x - x1) + y1
    # gives the following for y-intercept and x-intercept
    yintmin = ymin - slope*xmin
    xintmin = -1.*yintmin/slope

    yintmax = ymax - slope*xmax
    xintmax = -1.*yintmax/slope

    # this uses nbins regardless of whether running=True or False
    x_binwidth = (xintmax - xintmin)/float(nbins)
    y_binwidth = (yintmax - yintmin)/float(nbins)

    # if running=True this will be much smaller
    x_step = (xintmax - xintmin)/float(nbins_tot)
    y_step = (yintmax - yintmin)/float(nbins_tot)

    x_bins  = np.arange(xintmin, xintmax, x_binwidth) + x_binwidth/2.
    y_bins  = np.arange(yintmin, yintmax, y_binwidth) + y_binwidth/2.

    x_binlo = x_bins - x_binwidth/2.
    y_binlo = y_bins - y_binwidth/2.

    x_binhi = x_bins + x_binwidth/2.
    y_binhi = y_bins + y_binwidth/2.

    # For each bin, we want to pick points that are above a minimum line and below a maximum parallel line
    # so we are going to need the slope of the lines and points on those lines to get the equations
    # the points are just (x_binlo[i], y_binlo[i]) and similar for the high bin

    #lines = [slope, y_binlo, y_binhi]

    for i, bin in enumerate(the_empty):

        # get y-intercepts for this bin
        yint_lo = y_binlo[0] + (i*y_step)
        yint_hi = y_binhi[0] + (i*y_step)
        lines = [slope, yint_lo, yint_hi]
        inbin = iswithinlines_arr(np.array([x, y]), lines)

        if sum(inbin) > 0:
            this_x = x[inbin]
            this_y = y[inbin]

            the_stats_x[theavg][i]   =       np.mean(this_x)
            the_stats_x[thesig][i]   =        np.std(this_x)
            the_stats_x[themed][i]   =     np.median(this_x)
            the_stats_x[the5pct][i]  = np.percentile(this_x, 5)
            the_stats_x[the16pct][i] = np.percentile(this_x, 16)
            the_stats_x[the25pct][i] = np.percentile(this_x, 25)
            the_stats_x[the75pct][i] = np.percentile(this_x, 75)
            the_stats_x[the84pct][i] = np.percentile(this_x, 84)
            the_stats_x[the95pct][i] = np.percentile(this_x, 95)
            the_stats_x[thecount][i] =           len(this_x)
            the_stats_x[miderr][i]   = 0.5*(the_stats_x[the16pct][i]+the_stats_x[the84pct][i])
            the_stats_x[theerr][i]   = the_stats_x[the84pct][i] - the_stats_x[the16pct][i]

            the_stats_y[theavg][i]   =       np.mean(this_y)
            the_stats_y[thesig][i]   =        np.std(this_y)
            the_stats_y[themed][i]   =     np.median(this_y)
            the_stats_y[the5pct][i]  = np.percentile(this_y, 5)
            the_stats_y[the16pct][i] = np.percentile(this_y, 16)
            the_stats_y[the25pct][i] = np.percentile(this_y, 25)
            the_stats_y[the75pct][i] = np.percentile(this_y, 75)
            the_stats_y[the84pct][i] = np.percentile(this_y, 84)
            the_stats_y[the95pct][i] = np.percentile(this_y, 95)
            the_stats_y[thecount][i] =           len(this_y)
            the_stats_y[miderr][i]   = 0.5*(the_stats_y[the16pct][i]+the_stats_y[the84pct][i])
            the_stats_y[theerr][i]   = the_stats_y[the84pct][i] - the_stats_y[the16pct][i]
        else:
            the_stats_x[theavg][i]   = 999.
            the_stats_x[thesig][i]   = 999.
            the_stats_x[themed][i]   = 999.
            the_stats_x[the5pct][i]  = 999.
            the_stats_x[the16pct][i] = 999.
            the_stats_x[the25pct][i] = 999.
            the_stats_x[the75pct][i] = 999.
            the_stats_x[the84pct][i] = 999.
            the_stats_x[the95pct][i] = 999.
            the_stats_x[thecount][i] = 0
            the_stats_x[miderr][i]   = 999.
            the_stats_x[theerr][i]   = 999.

            the_stats_y[theavg][i]   = 999.
            the_stats_y[thesig][i]   = 999.
            the_stats_y[themed][i]   = 999.
            the_stats_y[the5pct][i]  = 999.
            the_stats_y[the16pct][i] = 999.
            the_stats_y[the25pct][i] = 999.
            the_stats_y[the75pct][i] = 999.
            the_stats_y[the84pct][i] = 999.
            the_stats_y[the95pct][i] = 999.
            the_stats_y[thecount][i] = 0
            the_stats_y[miderr][i]   = 999.
            the_stats_y[theerr][i]   = 999.


    return the_stats_x, the_stats_y







def iswithinlines_arr(points, lines):

    # points is np.array([x, y]) where x and y are points in a dataset
    # lines is [m, b_lo, b_hi] where each b is an array of i intercepts
    # i is the index you want to test to see whether the points are line_lo < points < line_hi
    # please make point and lines floats, okthxbai

    m  = lines[0]
    b_lo = lines[1]
    b_hi = lines[2]

    # point-slope form: y = m(x - x1) + y1 but pts_* gives y intercepts so x1 = 0 always
    this_ylo = m*(points[0]) + b_lo
    this_yhi = m*(points[0]) + b_hi

    return ((points[1] >= this_ylo) & (points[1] < this_yhi))



def iswithinlines_pt(point, lines):

    # point is [x, y]
    # lines is [m, b_lo, b_hi] where each b is an array of i intercepts
    # i is the index you want to test to see whether the point is line_lo < point < line_hi
    # please make point and lines floats, okthxbai

    m  = lines[0]
    b_lo = lines[1]
    b_hi = lines[2]

    # point-slope form: y = m(x - x1) + y1 but pts_* gives y intercepts so x1 = 0 always
    this_ylo = m*(point[0]) + b_lo
    this_yhi = m*(point[0]) + b_hi

    if (point[1] >= this_ylo) & (point[1] < this_yhi):
        return True
    else:
        return False




def testlines():
    plt.clf()
    plt.cla()

    xx = [0.01, 0.02, 0.11, 0.15, 0.41, 0.45, 0.55]
    yy = [0.51, 0.94, 0.34, 0.61, 0.00, 0.75, 0.22]
    plt_line_x = np.arange(0.,1.,0.01)
    #plt_linelo_y = slope*(plt_line_x - pts_lo[4][0]) + pts_lo[4][1]
    #plt_linehi_y = slope*(plt_line_x - pts_hi[4][0]) + pts_hi[4][1]
    plt_linelo_y = slope*(plt_line_x) + y_binlo[2]
    plt_linehi_y = slope*(plt_line_x) + y_binhi[2]
    xlimits = (0,1.0)
    ylimits = (0,2.0)
    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.plot(plt_line_x, plt_linelo_y, linewidth=2, color='black')
    plt.plot(plt_line_x, plt_linehi_y, linewidth=2, color='blue')
    plt.plot(xx, yy, marker='o', linestyle='None')
    plt.show()




plot_class_viscandels()
plot_class_visukmc()
