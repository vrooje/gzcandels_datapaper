import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from smfigure import SMfigure
#import datetime
# from mpl_toolkits.mplot3d import Axes3D
# import scipy.optimize
# import functools
# from scipy.interpolate import interp1d

from collections import Counter
#from pymongo import MongoClient

try:
    from astropy.io import fits as pyfits
    from astropy.io.fits import Column
    from astropy.io import ascii
except ImportError:
    import pyfits
    from pyfits import Column

from survey_tasks_responses import get_question_answer_mapping, get_weird_question
from survey_tasks_responses_wt import get_question_answer_mapping_wt, get_label_wt
from survey_tasks_responses_dc import get_question_answer_mapping_dc

which_survey = 'candels'

q_a_map    = get_question_answer_mapping(which_survey)
q_a_map_wt = get_question_answer_mapping_wt(which_survey)
q_a_map_wt_dc = get_question_answer_mapping_dc(which_survey)
label_map  = get_label_wt(which_survey)
weird_question = get_weird_question(which_survey)

questions = [which_survey+'-%i' % j for j in np.arange(len(q_a_map))]
if (len(weird_question) > 0):
    questions.remove(weird_question)

vote_fractions_in = '/Users/vrooje/Astro/Zooniverse/gz_reduction_sandbox/data/candels_weighted_seeded_collated_05_withmetadata_nodup_nobots_depthcorr_allcorr.csv'


def get_this(vote_fractions, this_q, this_a):
    this_count = q_a_map_wt[this_q]['count']
    this_ans   = q_a_map_wt[this_q][this_a]
    this_label = 'p_{'+label_map[this_q][this_a]+'}'
    oktoplot = vote_fractions[this_count] > 10.
    # this is an array with identical values of 1/N_tot
    make_frac_hist = np.ones_like(np.array(vote_fractions[this_ans][oktoplot]))/float(len(vote_fractions[this_ans][oktoplot]))

    return this_count, this_ans, this_label, oktoplot, make_frac_hist



def plot_ans_hist(vote_fractions, this_q, this_a):
    this_count, this_ans, this_label, oktoplot, make_frac_hist = get_this(vote_fractions, this_q, this_a)

    fig = plt.figure(figsize=(4, 4))
    ax1 = fig.add_subplot(1,1,1)
    xlimits = (-0.025, 1.025)
    ax1.hist(np.array(vote_fractions[this_ans][oktoplot]), bins=20, weights=make_frac_hist, histtype='step', color='black', label=this_label)
    ax1.set_xlim(xlimits)
    #ax1.set_xlabel(this_label)
    the_ylim = ax1.get_ylim()
    the_ylabelpos = the_ylim[0] + 0.05*(the_ylim[1]-the_ylim[0])
    ax1.text(0.2, the_ylabelpos, this_label, color='blue')
    ax1.yaxis.tick_right()
    #ax1.set_ylabel('N/N_{tot}')
    #ax1.legend(frameon=False)
    plt.tight_layout()

    plt.savefig(this_ans+'.png', facecolor='None', edgecolor='None')
    plt.savefig(this_ans+'.eps', facecolor='None', edgecolor='None')

    plt.cla()
    plt.clf()
    plt.close()





# ugh. but speed is needed.
def quick_and_dirty(vote_fractions, normalised, w1, h1):
    # smooth featured artifact clumpy edge-on spiral merger
    these_q = 'candels-0 candels-0 candels-0 candels-2 candels-9 candels-12 candels-16'.split()
    these_a = 'a-0 a-1 a-2 a-0 a-0 a-0 a-0'.split()

    fig = plt.figure(figsize=(w1, h1))
    #fig = SMfigure()
    gs = gridspec.GridSpec(7, 1)
    gs.update(hspace=0.00000, right=0.5)
    #fig.subplots_adjust(hspace=0.0, wspace=0.00)
    xlimits = (-0.025, 1.025)

    i=0
    this_count, this_ans, this_label, oktoplot, make_frac_hist = get_this(vote_fractions, these_q[i], these_a[i])

    ax0 = fig.add_subplot(gs[i,0])
    if normalised == True:
        ax0.hist(np.array(vote_fractions[this_ans][oktoplot]), bins=20, weights=make_frac_hist, histtype='step', color='black', label=this_label)
    else:
        ax0.hist(np.array(vote_fractions[this_ans][oktoplot]), bins=20, histtype='step', color='black', label=this_label)

    ax0.set_xlim(xlimits)
    the_ylim = ax0.get_ylim()
    the_ylabelpos = the_ylim[0] + 0.05*(the_ylim[1]-the_ylim[0])
    ax0.text(0.2, the_ylabelpos, this_label, color='blue')
    ax0.yaxis.tick_right()
    yticks = [int(q) for q in plt.gca().get_yticks().tolist()] # get list of ticks as ints
    yticks[0] = ''                           # set first tick to empty string
    ax0.set_yticklabels(yticks)              # set the labels

    # no x-axis labels
    ax0.xaxis.set_major_formatter(plt.NullFormatter())


    i=1
    this_count, this_ans, this_label, oktoplot, make_frac_hist = get_this(vote_fractions, these_q[i], these_a[i])

    ax1 = fig.add_subplot(gs[i,0])
    if normalised == True:
        ax1.hist(np.array(vote_fractions[this_ans][oktoplot]), bins=20, weights=make_frac_hist, histtype='step', color='black', label=this_label)
    else:
        ax1.hist(np.array(vote_fractions[this_ans][oktoplot]), bins=20, histtype='step', color='black', label=this_label)
    ax1.set_xlim(xlimits)
    the_ylim = ax1.get_ylim()
    the_ylabelpos = the_ylim[0] + 0.05*(the_ylim[1]-the_ylim[0])
    ax1.text(0.2, the_ylabelpos, this_label, color='blue')
    ax1.yaxis.tick_right()
    yticks = [int(q) for q in plt.gca().get_yticks().tolist()] # get list of ticks as ints
    yticks[0] = ''                           # set first tick to empty string
    yticks[-1] = ''                          # set last tick to empty string
    ax1.set_yticklabels(yticks)              # set the labels
    # no x-axis labels
    ax1.xaxis.set_major_formatter(plt.NullFormatter())


    i=2
    this_count, this_ans, this_label, oktoplot, make_frac_hist = get_this(vote_fractions, these_q[i], these_a[i])

    ax2 = fig.add_subplot(gs[i,0])
    if normalised == True:
        ax2.hist(np.array(vote_fractions[this_ans][oktoplot]), bins=20, weights=make_frac_hist, histtype='step', color='black', label=this_label)
    else:
        ax2.hist(np.array(vote_fractions[this_ans][oktoplot]), bins=20, histtype='step', color='black', label=this_label)
    ax2.set_xlim(xlimits)
    the_ylim = ax2.get_ylim()
    the_ylabelpos = the_ylim[0] + 0.05*(the_ylim[1]-the_ylim[0])
    ax2.text(0.2, the_ylabelpos, this_label, color='blue')
    ax2.yaxis.tick_right()
    yticks = [int(q) for q in plt.gca().get_yticks().tolist()] # get list of ticks as ints
    yticks[0] = ''                           # set first tick to empty string
    yticks[-1] = ''                          # set last tick to empty string
    ax2.set_yticklabels(yticks)              # set the labels
    # no x-axis labels
    ax2.xaxis.set_major_formatter(plt.NullFormatter())


    i=3
    this_count, this_ans, this_label, oktoplot, make_frac_hist = get_this(vote_fractions, these_q[i], these_a[i])

    ax3 = fig.add_subplot(gs[i,0])
    if normalised == True:
        ax3.hist(np.array(vote_fractions[this_ans][oktoplot]), bins=20, weights=make_frac_hist, histtype='step', color='black', label=this_label)
    else:
        ax3.hist(np.array(vote_fractions[this_ans][oktoplot]), bins=20, histtype='step', color='black', label=this_label)
    ax3.set_xlim(xlimits)
    the_ylim = ax3.get_ylim()
    the_ylabelpos = the_ylim[0] + 0.05*(the_ylim[1]-the_ylim[0])
    ax3.text(0.2, the_ylabelpos, this_label, color='blue')
    ax3.yaxis.tick_right()
    yticks = [int(q) for q in plt.gca().get_yticks().tolist()] # get list of ticks as ints
    yticks[0] = ''                           # set first tick to empty string
    yticks[-1] = ''                          # set last tick to empty string
    ax3.set_yticklabels(yticks)              # set the labels
    # no x-axis labels
    ax3.xaxis.set_major_formatter(plt.NullFormatter())
    ax3.set_ylabel('Number of Galaxies')
    ax3.yaxis.set_label_position("right")


    i=4
    this_count, this_ans, this_label, oktoplot, make_frac_hist = get_this(vote_fractions, these_q[i], these_a[i])

    ax4 = fig.add_subplot(gs[i,0])
    if normalised == True:
        ax4.hist(np.array(vote_fractions[this_ans][oktoplot]), bins=20, weights=make_frac_hist, histtype='step', color='black', label=this_label)
    else:
        ax4.hist(np.array(vote_fractions[this_ans][oktoplot]), bins=20, histtype='step', color='black', label=this_label)
    ax4.set_xlim(xlimits)
    the_ylim = ax4.get_ylim()
    the_ylabelpos = the_ylim[0] + 0.05*(the_ylim[1]-the_ylim[0])
    ax4.text(0.2, the_ylabelpos, this_label, color='blue')
    ax4.yaxis.tick_right()
    yticks = [int(q) for q in plt.gca().get_yticks().tolist()] # get list of ticks as ints
    yticks[0] = ''                           # set first tick to empty string
    yticks[-1] = ''                          # set last tick to empty string
    ax4.set_yticklabels(yticks)              # set the labels
    # no x-axis labels
    ax4.xaxis.set_major_formatter(plt.NullFormatter())


    i=5
    this_count, this_ans, this_label, oktoplot, make_frac_hist = get_this(vote_fractions, these_q[i], these_a[i])

    ax5 = fig.add_subplot(gs[i,0])
    if normalised == True:
        ax5.hist(np.array(vote_fractions[this_ans][oktoplot]), bins=20, weights=make_frac_hist, histtype='step', color='black', label=this_label)
    else:
        ax5.hist(np.array(vote_fractions[this_ans][oktoplot]), bins=20, histtype='step', color='black', label=this_label)
    ax5.set_xlim(xlimits)
    the_ylim = ax5.get_ylim()
    the_ylabelpos = the_ylim[0] + 0.05*(the_ylim[1]-the_ylim[0])
    ax5.text(0.2, the_ylabelpos, this_label, color='blue')
    ax5.yaxis.tick_right()
    yticks = [int(q) for q in plt.gca().get_yticks().tolist()] # get list of ticks as ints
    yticks[0] = ''                           # set first tick to empty string
    yticks[-1] = ''                          # set last tick to empty string
    ax5.set_yticklabels(yticks)              # set the labels
    # no x-axis labels
    ax5.xaxis.set_major_formatter(plt.NullFormatter())


    i=6
    this_count, this_ans, this_label, oktoplot, make_frac_hist = get_this(vote_fractions, these_q[i], these_a[i])

    ax6 = fig.add_subplot(gs[i,0])
    if normalised == True:
        ax6.hist(np.array(vote_fractions[this_ans][oktoplot]), bins=20, weights=make_frac_hist, histtype='step', color='black', label=this_label)
    else:
        ax6.hist(np.array(vote_fractions[this_ans][oktoplot]), bins=20, histtype='step', color='black', label=this_label)
    ax6.set_xlim(xlimits)
    the_ylim = ax6.get_ylim()
    the_ylabelpos = the_ylim[0] + 0.05*(the_ylim[1]-the_ylim[0])
    ax6.text(0.2, the_ylabelpos, this_label, color='blue')
    ax6.yaxis.tick_right()
    yticks = [int(q) for q in plt.gca().get_yticks().tolist()] # get list of ticks as ints
    yticks[-1] = ''                          # set last tick to empty string
    ax6.set_yticklabels(yticks)              # set the labels
    # we want this one to have x-axis labels
    ax6.set_xlabel('Vote Fraction')




    #plt.tight_layout()

    plt.savefig('stacked_p_hists.png', facecolor='None', edgecolor='None')
    plt.savefig('stacked_p_hists.eps', facecolor='None', edgecolor='None')

    plt.cla()
    plt.clf()
    plt.close()







vote_fractions = pd.read_csv(vote_fractions_in, low_memory=False)

#this_q   = 'candels-0'
#this_a   = 'a-0'
#this_ans = q_a_map_wt[this_q][this_a]

quick_and_dirty(vote_fractions, False, 5, 16)


#done
