import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker
import numpy as np
#import pandas as pd
import astropy
from astropy.table import Table
from scipy import stats, interpolate, special
from scipy.stats import gaussian_kde


def get_cplot(counts, xbins, ybins):
    ct = counts.transpose()
    extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()]
    return ct, extent


infile = 'candels_restframe_phot_weighted_classifications.fits'
fsmooth   = 't00_smooth_or_featured_a0_smooth_weighted_frac'
ffeatured = 't00_smooth_or_featured_a1_features_weighted_frac'
fartifact = 't00_smooth_or_featured_a2_artifact_weighted_frac'

print("Reading file %s ...\n" % infile)
gz_all = Table.read(infile)
print("    ... done.")

VJ_min = -1.7
VJ_max = 3.0
UV_min = -0.5
UV_max = 3.99
z_max = 1.5

zmins = [0.0, 0.5, 1.0]
zmaxs = [0.5, 1.0, z_max]
whichplot = [0, 1, 2]

threshold = 0.02 # 1 - threshold is the %ile to plot a contour at
threshold_col = 0.01 # We will plot the outer <--- fraction of points
bins = 60
bins_all = 60

okdata = np.invert(np.isnan(gz_all['z_best'])) & np.invert(np.isnan(gz_all['UX_rest'])) & np.invert(np.isnan(gz_all['V_rest'])) & np.invert(np.isnan(gz_all['J_rest']))

gz = gz_all[okdata]

gz['UmV'] = gz['UX_rest'] - gz['V_rest']
gz['VmJ'] = gz['V_rest']  - gz['J_rest']

not_artifact = gz[ffeatured] < 0.5
in_zlim = gz['z_best'] <= z_max
UVJlims = (gz['UmV'] > UV_min) & (gz['UmV'] < UV_max) & (gz['VmJ'] > VJ_min) & (gz['VmJ'] < VJ_max) & in_zlim & not_artifact

smooth_wt_sp = np.ones_like((np.array(gz['UmV']),np.array(gz['UmV']),np.array(gz['UmV'])))
smooth_wt_cl = np.ones_like((np.array(gz['UmV']),np.array(gz['UmV']),np.array(gz['UmV'])))
clumpy_wt    = np.ones_like((np.array(gz['UmV']),np.array(gz['UmV']),np.array(gz['UmV'])))
spiral_wt    = np.ones_like((np.array(gz['UmV']),np.array(gz['UmV']),np.array(gz['UmV'])))


colall = '#AAAAAA'

colcl = '#006e35'
colsp = '#30a0ca'
colft = "#0022CC"
colsm = "#ac0e30"

thresh_multi = [0.02, 0.1, 0.2, 0.4, 0.6, 0.75, 0.9, 0.95]

UVlim = (UV_min, UV_max)
VJlim = (VJ_min, VJ_max)

''' From Williams et al. (2009), ApJ 691, 1879, https://ui.adsabs.harvard.edu/#abs/2009ApJ...691.1879W/abstract :

Diagonal criteria are:
(U−V) > 0.88*(V−J)+0.69 for z < 0.5
(U−V) > 0.88*(V−J)+0.59 for 0.5 < z < 1
(U−V) > 0.88*(V−J)+0.49 for 1 < z < 2

And these are cut off by a horizontal line at (U-V) = 1.3 and
a vertical line at (V-J)=1.6 at all redshifts. These together with their
plot limits make the selection box with the corner cut off. To quote them:

"Additional criteria of U − V > 1.3 and V − J < 1.6 are applied to the quiescent galaxies at all redshifts to prevent contamination from unobscured and dusty star-forming galaxies, respectively. The samples of star-forming galaxies are then defined by everything falling outside this box (but within the color range plotted in Figure 9, such that the very small number of extreme color outliers are not included in either sample)."

So I need to construct a series of line segments that's:
(VJ_min, 1.3), (VJ_left_z, 1.3), (1.6, UV_right_z), (1.6, UV_max)
for each z value.

UmV = 0.88*VmJ+0.69
1.3 = 0.88 VmJ + 0.69
1.3 - 0.69 = 0.88 VmJ
(1.3 - 0.69)/0.88 = VmJ  <-- leftmost VmJ of z=0 diagonal

UmV = 0.88*1.6+0.69 <-- UmV at rightmost VmJ of z=0 diagonal

I will calculate this in the z loop using the changing coefficients defined below (from Williams et al. 2009, equation 4).

'''

passivebox_coeff = [0.69, 0.59, 0.49]

kwargs = dict(fontsize=14, ha='right', va='baseline')

which_plot = 'spiral'
#which_plot = 'featured'
#which_plot = 'clumpy'

nrow = 1
ncol = 3
fig, axs = plt.subplots(nrow, ncol, figsize=(12, 4))
for i, ax in enumerate(fig.axes):
    #ax.set_ylabel(str(i))

    VmJ_passive = [VJ_min, (1.3 - passivebox_coeff[i])/0.88, 1.6, 1.6]
    UmV_passive = [1.3, 1.3, 0.88*1.6+passivebox_coeff[i], UV_max]

    in_zlim = (gz['z_best'] > zmins[i]) & (gz['z_best'] <= zmaxs[i])

    # for contour purposes
    clean_featured = (gz['clean_featured']) & in_zlim
    clean_spiral   = (gz['clean_spiral'])   & in_zlim
    clean_clumpy   = (gz['clean_clumpy'])   & in_zlim
    UVJlims = (gz['UmV'] > UV_min) & (gz['UmV'] < UV_max) & (gz['VmJ'] > VJ_min) & (gz['VmJ'] < VJ_max) & in_zlim & not_artifact

    if which_plot == 'featured':
        magmax = max(gz['V_rest'][clean_featured])
        magmin = min(gz['V_rest'][clean_featured])
    elif which_plot == 'spiral':
        magmax = max(gz['V_rest'][clean_spiral])
        magmin = min(gz['V_rest'][clean_spiral])
    elif which_plot == 'clumpy':
        magmax = max(gz['V_rest'][clean_clumpy])
        magmin = min(gz['V_rest'][clean_clumpy])

    in_maglim = (gz['V_rest'] >= magmin) & (gz['V_rest'] <= magmax)
    clean_smooth   = (gz['clean_smooth'])   & in_zlim & in_maglim


    # get contours from KDE kernels
    # all data
    all_col = np.vstack([np.array(gz['VmJ'][UVJlims]), np.array(gz['UmV'][UVJlims])])
    kde_allc = gaussian_kde(all_col)
    #kde_allc = gaussian_kde(all_col, bw_method=kde_allc.scotts_factor()/2.)
    z_allc = kde_allc(all_col)
    xc = np.ma.masked_where(z_allc > threshold_col, np.array(gz['VmJ'][UVJlims]))
    yc = np.ma.masked_where(z_allc > threshold_col, np.array(gz['UmV'][UVJlims]))

    # Smooth - UVJ colors
    cleansmooth_colpts = np.vstack([np.array(gz['VmJ'][clean_smooth]), np.array(gz['UmV'][clean_smooth])])
    kde_scl = gaussian_kde(cleansmooth_colpts)
    #kde_scl = gaussian_kde(cleansmooth_colpts, bw_method=kde_s08.scotts_factor()/2.)
    z_scl = kde_scl(cleansmooth_colpts)
    # mask points above density threshold
    x_scl = np.ma.masked_where(z_scl > threshold, np.array(gz['VmJ'][clean_smooth]))
    y_scl = np.ma.masked_where(z_scl > threshold, np.array(gz['UmV'][clean_smooth]))

    # Featured - UVJ colors
    cleanfeatured_colpts = np.vstack([np.array(gz['VmJ'][clean_featured]), np.array(gz['UmV'][clean_featured])])
    kde_fcl = gaussian_kde(cleanfeatured_colpts)
    #kde_fcl = gaussian_kde(cleanfeatured_colpts, bw_method=kde_s08.scotts_factor()/2.)
    z_fcl = kde_fcl(cleanfeatured_colpts)
    # mask points above density threshold
    x_fcl = np.ma.masked_where(z_fcl > threshold, np.array(gz['VmJ'][clean_featured]))
    y_fcl = np.ma.masked_where(z_fcl > threshold, np.array(gz['UmV'][clean_featured]))

    # Spiral - UVJ colors
    cleanspiral_colpts = np.vstack([np.array(gz['VmJ'][clean_spiral]), np.array(gz['UmV'][clean_spiral])])
    kde_spcl = gaussian_kde(cleanspiral_colpts)
    #kde_spcl = gaussian_kde(cleanspiral_colpts, bw_method=kde_s08.scotts_factor()/2.)
    z_spcl = kde_spcl(cleanspiral_colpts)
    # mask points above density threshold
    x_spcl = np.ma.masked_where(z_spcl > threshold, np.array(gz['VmJ'][clean_spiral]))
    y_spcl = np.ma.masked_where(z_spcl > threshold, np.array(gz['UmV'][clean_spiral]))

    # Clumpy - UVJ colors
    cleanclumpy_colpts = np.vstack([np.array(gz['VmJ'][clean_clumpy]), np.array(gz['UmV'][clean_clumpy])])
    kde_clcl = gaussian_kde(cleanclumpy_colpts)
    #kde_clcl = gaussian_kde(cleanclumpy_colpts, bw_method=kde_s08.scotts_factor()/2.)
    z_clcl = kde_clcl(cleanclumpy_colpts)
    # mask points above density threshold
    x_clcl = np.ma.masked_where(z_clcl > threshold, np.array(gz['VmJ'][clean_clumpy]))
    y_clcl = np.ma.masked_where(z_clcl > threshold, np.array(gz['UmV'][clean_clumpy]))




    ax.set_xlim(VJlim)
    ax.set_ylim(UVlim)

    xmin, xmax = VJ_min, VJ_max
    ymin, ymax = UV_min, UV_max
    #xmin = -.2

    # prepare grid for density map
    xedges = np.linspace(xmin, xmax, bins)
    yedges = np.linspace(ymin, ymax, bins)
    xxc, yyc = np.meshgrid(xedges, yedges)
    gridpoints = np.array([xxc.ravel(), yyc.ravel()])

    # compute density maps
    zz_allc = np.reshape(kde_allc(gridpoints), xxc.shape)
    zz_fcl  = np.reshape(kde_fcl(gridpoints), xxc.shape)
    zz_scl  = np.reshape(kde_scl(gridpoints), xxc.shape)
    zz_spcl = np.reshape(kde_spcl(gridpoints), xxc.shape)
    zz_clcl = np.reshape(kde_clcl(gridpoints), xxc.shape)

    imc = ax.imshow(zz_allc, cmap='Greys', interpolation='nearest', origin='lower', extent=[xmin, xmax, ymin, ymax], aspect='auto')
    ax.scatter(xc, yc, c=colall, marker='.', edgecolor='None')

    ax.contour(xxc, yyc, zz_allc, levels=[threshold_col], colors=colall, linestyles='solid')

    csm = ax.contour(xxc, yyc, zz_scl, levels=thresh_multi, colors=colsm, linestyles='solid')
    csm.collections[0].set_label('${\\rm Smooth}$')

    ax.plot(VmJ_passive, UmV_passive, marker='None', linestyle='dashed', color='black')

    #csf = ax.contour(xxc, yyc, zz_fcl, levels=thresh_multi, colors=colft, linestyles='dashed')
    #csf.collections[0].set_label('${\\rm Featured}$')

    #csp = ax.contour(xxc, yyc, zz_spcl, levels=thresh_multi, colors=colsp, linestyles='dashed')
    #csp.collections[0].set_label('${\\rm Spiral}$')

    #ccl = ax.contour(xxc, yyc, zz_clcl, levels=thresh_multi, colors=colcl, linestyles='dashed')
    #csp.collections[0].set_label('${\\rm Clumpy}$')

    #ax.plot(gz['VmJ'][clean_smooth], gz['UmV'][clean_smooth], marker='o',  color=colsm, ms=4., linestyle='None', label = '${\\rm Smooth}$')

    if which_plot == 'featured':
        ax.plot(gz['VmJ'][clean_featured], gz['UmV'][clean_featured], marker='o',  color=colft, ms=4., linestyle='None', label = '${\\rm Featured}$', alpha=0.8)
    elif which_plot == 'spiral':
        ax.plot(gz['VmJ'][clean_spiral], gz['UmV'][clean_spiral], marker='o',  alpha=0.8, color=colsp, ms=4., linestyle='None', label = '${\\rm Spiral}$')
    elif which_plot == 'clumpy':
        ax.plot(gz['VmJ'][clean_clumpy], gz['UmV'][clean_clumpy], marker='s',  alpha=0.8, color=colcl, ms=4., linestyle='None', label = '${\\rm Clumpy}$')

    ax.set_xlabel('${\\rm Rest-frame\\ } (V-J)$')
    # all the \\ are forced spaces because Python doesn't quite do LaTeX spacing right
    zlabel = '$%.1f\\ <\\ z\\ \leq\\ %.1f$' % (zmins[i], zmaxs[i])
    if i == 0:
        ax.set_ylabel('${\\rm Rest-frame\\ } (U-V)$')
        zlabel = '$z\\ \leq\\ %.1f$' % (zmaxs[i])


    ax.text(2.75, 3.5, zlabel, **kwargs)

    ax.legend(loc='upper left', frameon=True)




#####################################################
#####################################################
#####################################################
#####################################################
#####################################################






plt.tight_layout()

fout = 'UVJ_smooth_%s_contours_zbins_2' % which_plot
plt.savefig('%s.png' % fout, facecolor='None', edgecolor='None')
plt.savefig('%s.eps' % fout, facecolor='None', edgecolor='None')
plt.close()
plt.cla()
plt.clf()

#booya
