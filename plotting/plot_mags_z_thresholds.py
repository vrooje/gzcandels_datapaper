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




infile = 'candels_restframe_phot_weighted_classifications.fits'
fsmooth   = 't00_smooth_or_featured_a0_smooth_weighted_frac'
ffeatured = 't00_smooth_or_featured_a1_features_weighted_frac'
fartifact = 't00_smooth_or_featured_a2_artifact_weighted_frac'

print("Reading file %s ...\n" % infile)
gz_all = Table.read(infile)
print("    ... done.")

threshold = 0.02 # 1 - threshold is the %ile to plot a contour at
threshold_all = 0.02 # We will plot the outer <--- fraction of points
threshold_col = 0.01 # We will plot the outer <--- fraction of points
bins = 60
bins_all = 60

okdata = np.invert(np.isnan(gz_all['z_best'])) & np.invert(np.isnan(gz_all['UX_rest'])) & np.invert(np.isnan(gz_all['V_rest'])) & np.invert(np.isnan(gz_all['J_rest']))

gz = gz_all[okdata]


not_artifact = gz[ffeatured] < 0.5
Vzlims = (gz['z_best'] > 0.0) & (gz['z_best'] < 4.1) & (gz['V_rest'] > -25.5) & (gz['V_rest'] < -9.5) & not_artifact


# so that we can plot points without overplotting duplicates
smooth04 = (gz[fsmooth] >= 0.4) & (gz[fsmooth] < 0.5)
smooth05 = (gz[fsmooth] >= 0.5) & (gz[fsmooth] < 0.6)
smooth06 = (gz[fsmooth] >= 0.6) & (gz[fsmooth] < 0.7)
smooth07 = (gz[fsmooth] >= 0.7) & (gz[fsmooth] < 0.8)
smooth08 = (gz[fsmooth] >= 0.8)

# for contour purposes
smooth04a = (gz[fsmooth] >= 0.4)
smooth05a = (gz[fsmooth] >= 0.5)
smooth06a = (gz[fsmooth] >= 0.6)
smooth07a = (gz[fsmooth] >= 0.7)
smooth08a = smooth08

featured04 = (gz[ffeatured] >= 0.4) & (gz[ffeatured] < 0.5)
featured05 = (gz[ffeatured] >= 0.5) & (gz[ffeatured] < 0.6)
featured06 = (gz[ffeatured] >= 0.6) & (gz[ffeatured] < 0.7)
featured07 = (gz[ffeatured] >= 0.7)

featured04a = (gz[ffeatured] >= 0.4)
featured05a = (gz[ffeatured] >= 0.5)
featured06a = (gz[ffeatured] >= 0.6)
featured07a = featured07



# get contours from KDE kernels
# all data
all_pts = np.vstack([np.array(gz['z_best'][Vzlims]), np.array(gz['V_rest'][Vzlims])])
kde_all = gaussian_kde(all_pts)
kde_all = gaussian_kde(all_pts, bw_method=kde_all.scotts_factor()/2.)
z_all = kde_all(all_pts)
x = np.ma.masked_where(z_all > threshold_all, np.array(gz['z_best'][Vzlims]))
y = np.ma.masked_where(z_all > threshold_all, np.array(gz['V_rest'][Vzlims]))


#smooth - Vz
smooth_pts04 = np.vstack([np.array(gz['z_best'][smooth04a]), np.array(gz['V_rest'][smooth04a])])
smooth_pts05 = np.vstack([np.array(gz['z_best'][smooth05a]), np.array(gz['V_rest'][smooth05a])])
smooth_pts06 = np.vstack([np.array(gz['z_best'][smooth06a]), np.array(gz['V_rest'][smooth06a])])
smooth_pts07 = np.vstack([np.array(gz['z_best'][smooth07a]), np.array(gz['V_rest'][smooth07a])])
smooth_pts08 = np.vstack([np.array(gz['z_best'][smooth08a]), np.array(gz['V_rest'][smooth08a])])
kde_s04 = gaussian_kde(smooth_pts04, bw_method=kde_s04.scotts_factor()/2.)
kde_s05 = gaussian_kde(smooth_pts05, bw_method=kde_s05.scotts_factor()/2.)
kde_s06 = gaussian_kde(smooth_pts06, bw_method=kde_s06.scotts_factor()/2.)
kde_s07 = gaussian_kde(smooth_pts07, bw_method=kde_s07.scotts_factor()/2.)
kde_s08 = gaussian_kde(smooth_pts08, bw_method=kde_s08.scotts_factor()/2.)
z_s04 = kde_s04(smooth_pts04)
z_s05 = kde_s05(smooth_pts05)
z_s06 = kde_s06(smooth_pts06)
z_s07 = kde_s07(smooth_pts07)
z_s08 = kde_s08(smooth_pts08)
# mask points above density threshold
x_s04 = np.ma.masked_where(z_s04 > threshold, np.array(gz['z_best'][smooth04a]))
y_s04 = np.ma.masked_where(z_s04 > threshold, np.array(gz['V_rest'][smooth04a]))
x_s05 = np.ma.masked_where(z_s05 > threshold, np.array(gz['z_best'][smooth05a]))
y_s05 = np.ma.masked_where(z_s05 > threshold, np.array(gz['V_rest'][smooth05a]))
x_s06 = np.ma.masked_where(z_s06 > threshold, np.array(gz['z_best'][smooth06a]))
y_s06 = np.ma.masked_where(z_s06 > threshold, np.array(gz['V_rest'][smooth06a]))
x_s07 = np.ma.masked_where(z_s07 > threshold, np.array(gz['z_best'][smooth07a]))
y_s07 = np.ma.masked_where(z_s07 > threshold, np.array(gz['V_rest'][smooth07a]))
x_s08 = np.ma.masked_where(z_s08 > threshold, np.array(gz['z_best'][smooth08a]))
y_s08 = np.ma.masked_where(z_s08 > threshold, np.array(gz['V_rest'][smooth08a]))


# featured - Vz
featured_ptf04 = np.vstack([np.array(gz['z_best'][featured04a]), np.array(gz['V_rest'][featured04a])])
featured_ptf05 = np.vstack([np.array(gz['z_best'][featured05a]), np.array(gz['V_rest'][featured05a])])
featured_ptf06 = np.vstack([np.array(gz['z_best'][featured06a]), np.array(gz['V_rest'][featured06a])])
featured_ptf07 = np.vstack([np.array(gz['z_best'][featured07a]), np.array(gz['V_rest'][featured07a])])
kde_f04 = gaussian_kde(featured_ptf04, bw_method=kde_f04.scotts_factor()*.8)
kde_f05 = gaussian_kde(featured_ptf05, bw_method=kde_f05.scotts_factor()*.8)
kde_f06 = gaussian_kde(featured_ptf06, bw_method=kde_f06.scotts_factor()*.8)
kde_f07 = gaussian_kde(featured_ptf07, bw_method=kde_f07.scotts_factor()*.8)
z_f04 = kde_f04(featured_ptf04)
z_f05 = kde_f05(featured_ptf05)
z_f06 = kde_f06(featured_ptf06)
z_f07 = kde_f07(featured_ptf07)
# mask points above density threshold
x_f04 = np.ma.masked_where(z_f04 > threshold, np.array(gz['z_best'][featured04a]))
y_f04 = np.ma.masked_where(z_f04 > threshold, np.array(gz['V_rest'][featured04a]))
x_f05 = np.ma.masked_where(z_f05 > threshold, np.array(gz['z_best'][featured05a]))
y_f05 = np.ma.masked_where(z_f05 > threshold, np.array(gz['V_rest'][featured05a]))
x_f06 = np.ma.masked_where(z_f06 > threshold, np.array(gz['z_best'][featured06a]))
y_f06 = np.ma.masked_where(z_f06 > threshold, np.array(gz['V_rest'][featured06a]))
x_f07 = np.ma.masked_where(z_f07 > threshold, np.array(gz['z_best'][featured07a]))
y_f07 = np.ma.masked_where(z_f07 > threshold, np.array(gz['V_rest'][featured07a]))




colall = '#AAAAAA'

col04 = '#006e35'
col05 = '#4455CC'
col06 = '#30a0ca'
#col07 = '#00ccaE'
col07 = "#ac0e30"

colclean = "#ac0e30"

sty04 = 'dashed'
sty05 = 'dotted'
sty06 = 'dashdot'
sty07 = 'solid'

styclean = 'solid'


fig = plt.figure(figsize=(10, 4))
gs = gridspec.GridSpec(1,2)
#gs.update(hspace=0.25, wspace=0.001)

zaxis = (0.0, 4.)
Vaxis = (-24.2, -10.)

ax1 = fig.add_subplot(gs[0,0])
ax1.set_xlim(zaxis)
ax1.set_ylim(Vaxis)
ax1.invert_yaxis()

# it should be ax1 not plt below but if I do that I can't get the colorbar to work
#plt.hexbin(gz['z_best'][Vzlims], gz['V_rest'][Vzlims], gridsize=25, bins='log', cmap='Greys', label='_nolegend_')

# plot unmasked points
ax1.scatter(x, y, c=colall, marker='.', edgecolor='None')
# get bounds from axes
# this is a bit silly as we've already defined them above, but just in case
# you need this for some other purpose later you'll maybe find this in a search
xmin, xmax = ax1.get_xlim()
ymin, ymax = ax1.get_ylim()
#xmin = -.2

# prepare grid for density map
xedges = np.linspace(xmin, xmax, bins)
yedges = np.linspace(ymin, ymax, bins)
xx, yy = np.meshgrid(xedges, yedges)
gridpoints = np.array([xx.ravel(), yy.ravel()])

# compute density maps
zz = np.reshape(kde_all(gridpoints), xx.shape)

zz_s04 = np.reshape(kde_s04(gridpoints), xx.shape)
zz_s05 = np.reshape(kde_s05(gridpoints), xx.shape)
zz_s06 = np.reshape(kde_s06(gridpoints), xx.shape)
zz_s07 = np.reshape(kde_s07(gridpoints), xx.shape)
zz_s08 = np.reshape(kde_s08(gridpoints), xx.shape)

# plot density map
im1 = ax1.imshow(zz, cmap='Greys', interpolation='nearest', origin='lower', extent=[xmin, xmax, ymin, ymax], aspect='auto')
ax1.contour(xx, yy, zz, levels=[threshold_all], colors=colall, linestyles='solid', label = '_nolegend_')


# plot threshold contour
#ax1.contour(xx, yy, zz_s04, levels=[threshold], colors=col04, linestyles=sty04, label = '$f_{\\rm smooth} \\geq 0.4$', lineweights=2)
cs04 = ax1.contour(xx, yy, zz_s04, levels=[threshold], colors=col04, linestyles=sty04, label = '$f_{\\rm smooth} \\geq 0.4$', lineweights=2)
cs05 = ax1.contour(xx, yy, zz_s05, levels=[threshold], colors=col05, linestyles=sty05, label = '$f_{\\rm smooth} \\geq 0.5$', lineweights=2)
cs06 = ax1.contour(xx, yy, zz_s06, levels=[threshold], colors=col06, linestyles=sty06, label = '$f_{\\rm smooth} \\geq 0.6$', lineweights=2)
cs07 = ax1.contour(xx, yy, zz_s07, levels=[threshold], colors=col07, linestyles=sty07, label = '$f_{\\rm smooth} \\geq 0.7$', lineweights=2)
#cs08 = ax1.contour(xx, yy, zz_s08, levels=[threshold], colors=colclean, linestyles=styclean, label = '$f_{\\rm smooth} \\geq 0.8$', lineweights=3)

cslabels = ['$f_{\\rm smooth} \\geq 0.4$', '$f_{\\rm smooth} \\geq 0.5$', '$f_{\\rm smooth} \\geq 0.6$', '$f_{\\rm smooth} \\geq 0.7$']
cs04.collections[0].set_label(cslabels[0])
cs05.collections[0].set_label(cslabels[1])
cs06.collections[0].set_label(cslabels[2])
cs07.collections[0].set_label(cslabels[3])


ax1.set_xlabel('Redshift $z$')
ax1.set_ylabel('Rest-frame $V$ absolute magnitude')
ax1.legend(loc='lower right', frameon=True)

cb1 = plt.colorbar(im1)
cb1.set_label("log(N)")




ax2 = fig.add_subplot(gs[0,1])
plt.xlim(zaxis)
plt.ylim(Vaxis)
ax2.invert_yaxis()

# it should be ax1 not plt below but if I do that I can't get the colorbar to work
#plt.hexbin(gz['z_best'][Vzlims], gz['V_rest'][Vzlims], gridsize=25, bins='log', cmap='Greys', label='_nolegend_')
# plot density map
im2 = ax2.imshow(zz, cmap='Greys', interpolation='nearest', origin='lower', extent=[xmin, xmax, ymin, ymax], aspect='auto')
ax2.contour(xx, yy, zz, levels=[threshold_all], colors=colall, linestyles='solid', label = '_nolegend_')

# plot unmasked points
ax2.scatter(x, y, c='#AAAAAA', marker='.', edgecolor='None')


xmin, xmax = ax2.get_xlim()
ymin, ymax = ax2.get_ylim()
#xmin = -.2

# prepare grid for density map
xedges = np.linspace(xmin, xmax, bins)
yedges = np.linspace(ymin, ymax, bins)
xx, yy = np.meshgrid(xedges, yedges)
gridpoints = np.array([xx.ravel(), yy.ravel()])

# compute density maps
zz_f04 = np.reshape(kde_f04(gridpoints), xx.shape)
zz_f05 = np.reshape(kde_f05(gridpoints), xx.shape)
zz_f06 = np.reshape(kde_f06(gridpoints), xx.shape)
zz_f07 = np.reshape(kde_f07(gridpoints), xx.shape)

# plot density map
#im = ax1.imshow(zz, cmap='CMRmap_r', interpolation='nearest', origin='lower', extent=[xmin, xmax, ymin, ymax])

# plot threshold contour
cf04 = ax2.contour(xx, yy, zz_f04, levels=[threshold], colors=col04, linestyles=sty04, label = '$f_{\\rm featured} \\geq 0.4$', lineweights=2)
cf05 = ax2.contour(xx, yy, zz_f05, levels=[threshold], colors=col05, linestyles=sty05, label = '$f_{\\rm featured} \\geq 0.5$', lineweights=2)
cf06 = ax2.contour(xx, yy, zz_f06, levels=[threshold], colors=col06, linestyles=sty06, label = '$f_{\\rm featured} \\geq 0.6$', lineweights=2)
cf07 = ax2.contour(xx, yy, zz_f07, levels=[threshold], colors=col07, linestyles=sty07, label = '$f_{\\rm featured} \\geq 0.7$', lineweights=3)

cflabels = ['$f_{\\rm featured} \\geq 0.4$', '$f_{\\rm featured} \\geq 0.5$', '$f_{\\rm featured} \\geq 0.6$', '$f_{\\rm featured} \\geq 0.7$']

cf04.collections[0].set_label(cflabels[0])
cf05.collections[0].set_label(cflabels[1])
cf06.collections[0].set_label(cflabels[2])
cf07.collections[0].set_label(cflabels[3])



ax2.set_xlabel('Redshift $z$')
#ax2.ylabel('Rest-frame $V$ absolute magnitude')
ax2.legend(loc='lower right', frameon=True)

cb2 = plt.colorbar(im2)
cb2.set_label("log(N)")

plt.tight_layout()


fout = 'V_z_thresholds_smooth_featured'
plt.savefig('%s.png' % fout, facecolor='None', edgecolor='None')
plt.savefig('%s.eps' % fout, facecolor='None', edgecolor='None')
plt.close()
plt.cla()
plt.clf()





#####################################################
#####################################################
#####################################################
#####################################################
#####################################################
#booya
