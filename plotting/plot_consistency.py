import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize
import functools
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D

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
from survey_tasks_responses_wt import get_question_answer_mapping_wt
from survey_tasks_responses_dc import get_question_answer_mapping_dc

which_survey = 'candels'

q_a_map    = get_question_answer_mapping(which_survey)
q_a_map_wt = get_question_answer_mapping_wt(which_survey)
q_a_map_wt_dc = get_question_answer_mapping_dc(which_survey)
weird_question = get_weird_question(which_survey)

questions = [which_survey+'-%i' % j for j in np.arange(len(q_a_map))]
if (len(weird_question) > 0):
    questions.remove(weird_question)


vote_fractions_out = '/Users/vrooje/Astro/Zooniverse/gz_reduction_sandbox/data/candels_weighted_seeded_collated_05_withmetadata_nodup_nobots_depthcorr.csv'

galdiff_file = '/Users/vrooje/Astro/Zooniverse/gz_reduction_sandbox/data/gal_diff_goods_depth_fracdiff.csv'







def lorenzcurve(nclass_user):
    # calculate lorenz curve of classifiers' classifications
    # Note: Lorenz was a person and not the same person as Lorentz
    # He wasn't even a physicist.
    nclass_user_sort = nclass_user.copy()
    nclass_user_sort.sort()
    nclass_user_lorenz = nclass_user_sort.copy()
    for thisuser, this_nclass in enumerate(nclass_user_sort):
        if thisuser > 0:
            nclass_user_lorenz[thisuser] += nclass_user_lorenz[thisuser-1]
    lorenz_curve = nclass_user_lorenz / max(nclass_user_lorenz)
    lorenz_user = len(lorenz_curve)

    plt.figure(1)
    fig = plt.gcf()
    fig.set_size_inches(8,3)
    plt.subplot(121)

    # plot histogram of classifications by subject
    qbins = np.arange(11,83) - 0.5
    plt.hist(vote_fractions.num_classifications, bins=qbins, histtype='step', color='black')
    plt.xlabel("Number of Classifications Per Subject")
    plt.ylabel("Number of Subjects")
    plt.yscale('log', nonposy='clip')
    #plt.show()
    #plt.clf()
    #plt.cla()

    plt.subplot(122)

    ylimits = (0,1)
    xlimits = (1, lorenz_user)

    plt.ylim(ylimits)
    plt.xlim(xlimits)
    plt.plot(lorenz_curve, color='black')
    plt.plot(xlimits, ylimits, '--', color='black')
    plt.xlabel("Volunteer number (sorted by number of classifications)")
    plt.ylabel("Cumulative number of total project classifications")

    fig.savefig('classifications_users_basicinfo.eps',dpi=300)
    #plt.show()







def consistency_plot:
    from survey_tasks_responses import get_question_answer_mapping, get_weird_question
    from survey_tasks_responses_wt import get_question_answer_mapping_wt
    from consensus_weight import weight_from_consistency

    #vote_fractions_filename = 'data/candels_unseeded_avgweighted_03_collated_2015-03-01.csv'
    #classifications_filename = 'data/2015-03-01_galaxy_zoo_classifications_CANDELSonly_unseeded_avgweighted_03.csv'
    #classifications_filebase = 'data/2015-03-01_galaxy_zoo_classifications_CANDELSonly_unseeded_avgweighted_0'

    #vote_fractions_filename = 'data/candels_avgweighted_unseeded_collated_05_wdup.csv'
    #classifications_filename = 'data/candels_avgweighted_unseeded_classifications_05_wdup.csv'
    #classifications_filebase = 'data/candels_avgweighted_unseeded_classifications_0'

    #vote_fractions_filename = 'data/candels_weighted_unseeded_collated_05_wdup.csv'
    #classifications_filename = 'data/candels_weighted_unseeded_classifications_05_wdup.csv'
    #classifications_filebase = 'data/candels_weighted_unseeded_classifications_0'
    #suffix = '_wdup.csv'

    vote_fractions_filename = 'data/candels_weighted_seeded_collated_05_wdup_nobots.csv'
    classifications_filename = 'data/candels_weighted_seeded_classifications_05_wdup_nobots.csv'
    classifications_filebase = 'data/candels_weighted_seeded_classifications_0'
    suffix = '_wdup_nobots.csv'

    print('Reading training file %s ...' % vote_fractions_filename)
    vote_fractions_in = pd.read_csv(vote_fractions_filename)
    vote_fractions_subjects = vote_fractions_in.subject_id.unique()
    vote_fractions = vote_fractions_in.set_index('subject_id')

    print('Reading classifications file %s ...' % classifications_filename)
    classifications_in = pd.read_csv(classifications_filename, low_memory=False)
    classifications = classifications_in.set_index('id')
    all_subjects = classifications.subject_id.unique()
    n_classifications = len(classifications)

    # we'll use this to count unweighted classifications (seems faster than value_counts somehow)
    classifications['unweight'] = (classifications.weight > -1).astype(int)
    classifications['unweight'] *= 0
    classifications['unweight'] += 1


    if 'user_name' in classifications.columns:
        usercol = 'user_name'
    elif 'login' in classifications.columns:
        usercol = 'login'
    elif 'user_id' in classifications.columns:
        usercol = 'user_id'
    elif 'user' in classifications.columns:
        usercol = 'user'
    elif 'zooniverse_user_id' in classifications.columns:
        usercol = 'zooniverse_user_id'


    all_users = classifications[usercol].unique()
    not_registered = np.core.defchararray.startswith(all_users.astype(str), 'not-logged-in-')
    is_registered  = np.invert(not_registered)
    n_users = len(all_users)

    by_user = classifications.groupby(classifications[usercol])
    # get raw classification numbers
    nclass_user = by_user.unweight.count()


    # values in these columns should be the same for every classification the user makes
    # and .first() times out as slightly faster than .mean()
    user_params = by_user['avg_weight avg_consistency weight_from_avg_consistency'.split()].first()

    # now read previous iterations
    classfile = classifications_filebase + '1' + suffix
    print('Reading %s' % classfile)
    classifications_this = pd.read_csv(classfile)

    by_user_this = classifications_this.groupby(classifications_this[usercol])
    user_params_01 = by_user_this['avg_weight avg_consistency weight_from_avg_consistency'.split()].first()

    classfile = classifications_filebase + '2' + suffix
    print('Reading %s' % classfile)
    classifications_this = pd.read_csv(classfile)

    by_user_this = classifications_this.groupby(classifications_this[usercol])
    user_params_02 = by_user_this['avg_weight avg_consistency weight_from_avg_consistency'.split()].first()

    classfile = classifications_filebase + '3' + suffix
    print('Reading %s' % classfile)
    classifications_this = pd.read_csv(classfile)

    by_user_this = classifications_this.groupby(classifications_this[usercol])
    user_params_03 = by_user_this['avg_weight avg_consistency weight_from_avg_consistency'.split()].first()

    classfile = classifications_filebase + '4' + suffix
    print('Reading %s' % classfile)
    classifications_this = pd.read_csv(classfile)

    by_user_this = classifications_this.groupby(classifications_this[usercol])
    user_params_04 = by_user_this['avg_weight avg_consistency weight_from_avg_consistency'.split()].first()


    wt = 'avg_consistency'


    dwt_0201 = user_params_02[wt] - user_params_01[wt]
    dwt_0302 = user_params_03[wt] - user_params_02[wt]
    dwt_0403 = user_params_04[wt] - user_params_03[wt]
    dwt_0504 = user_params[wt]    - user_params_04[wt]

    pwt_0201 = dwt_0201/user_params_01[wt]
    pwt_0302 = dwt_0302/user_params_02[wt]
    pwt_0403 = dwt_0403/user_params_03[wt]
    pwt_0504 = dwt_0504/user_params_04[wt]

    mean_diff = [np.mean(dwt_0201), np.mean(dwt_0302), np.mean(dwt_0403), np.mean(dwt_0504)]
    low_diff  = [np.percentile(dwt_0201, 1),  np.percentile(dwt_0302, 1),  np.percentile(dwt_0403, 1),  np.percentile(dwt_0504, 1)]
    high_diff = [np.percentile(dwt_0201, 99), np.percentile(dwt_0302, 99), np.percentile(dwt_0403, 99), np.percentile(dwt_0504, 99)]
    min_diff  = [min(dwt_0201), min(dwt_0302), min(dwt_0403), min(dwt_0504)]
    max_diff  = [max(dwt_0201), max(dwt_0302), max(dwt_0403), max(dwt_0504)]

    mean_pct = [np.mean(pwt_0201), np.mean(pwt_0302), np.mean(pwt_0403), np.mean(pwt_0504)]
    low_pct  = [np.percentile(pwt_0201, 1),  np.percentile(pwt_0302, 1),  np.percentile(pwt_0403, 1),  np.percentile(pwt_0504, 1)]
    high_pct = [np.percentile(pwt_0201, 99), np.percentile(pwt_0302, 99), np.percentile(pwt_0403, 99), np.percentile(pwt_0504, 99)]
    min_pct  = [min(pwt_0201), min(pwt_0302), min(pwt_0403), min(pwt_0504)]
    max_pct  = [max(pwt_0201), max(pwt_0302), max(pwt_0403), max(pwt_0504)]


    #oh, the memory
    classifications_this = 0
    by_user_this = 0

    #plt.clf()
    #plt.cla()




    fig = plt.figure(figsize=(5.5, 5.5))
    ax1 = fig.add_subplot(111)
    #fig.set_size_inches(8,3)
    qbins = np.arange(0,1000)/1000.

    subx1 = 0.74
    subx2 = 0.88
    suby1 = 7e4
    suby2 = 1e5

    xctr = 0.5*(subx1+subx2)
    yctr = 0.5*(suby1+suby2)
    xsize = subx2-subx1
    ysize = suby2-suby1

#    rect = plt.Rectangle([xctr, yctr], xsize, ysize, facecolor='black', edgecolor='none', alpha=0.8)
    rect = plt.Rectangle([subx1, suby1], xsize, ysize, facecolor='#faf6cf', edgecolor='none')
    ax1.add_patch(rect)

    #plt.subplot(121)

    #fig.set_aspect('equal')
    xlimits = (0.275,0.9999)
    ylimits = (3,1.1*len(user_params))
    ax1.set_ylim(ylimits)
    ax1.set_xlim(xlimits)
    ax1.set_yscale('log', nonposy='clip')
    plt.ylim(ylimits)
    plt.xlim(xlimits)
    plt.hist(user_params_01[wt], bins=qbins, histtype='step', cumulative=True, color='black', linestyle='dotted', label='1st Iteration')
    plt.hist(user_params_02[wt], bins=qbins, histtype='step', cumulative=True, color='black', linestyle='dashed', label='2nd')
    #plt.hist(user_params_03[wt], bins=qbins, histtype='step', cumulative=True, color='blue', linestyle='dotted', linewidth=2, label='3rd')
    #plt.hist(user_params_04[wt], bins=qbins, histtype='step', cumulative=True, color='green', linestyle='dashed', linewidth=2, label='4th')
    plt.hist(user_params[wt],    bins=qbins, histtype='step', cumulative=True, color='black', linestyle='solid', label='3rd-5th')
    plt.yscale('log', nonposy='clip')
    plt.xlabel("User Consistency", fontsize=16)
    plt.ylabel("Cumulative User Count", fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=14)
    plt.legend(loc='lower right', frameon=None, fontsize=16) #loc=2
    #fig.savefig('consistencies_iterations.eps',dpi=300)

    xline1 = [0.61, 0.74]
    yline1 = [10425., 1.0e5]

    xline2 = [0.98, 0.88]
    yline2 = [10425., 1.0e5]

    ax1.add_line(Line2D(xline1, yline1, linewidth=0.8, color='#888888', linestyle='dashdot'))
    ax1.add_line(Line2D(xline2, yline2, linewidth=0.8, color='#888888', linestyle='dashdot'))


    # this is another inset axes over the main axes
    a = plt.axes([0.5223, 0.375, .393, .393], axisbg='#faf6cf')
    plt.hist(user_params_01[wt], bins=qbins, histtype='step', cumulative=True, color='black', linestyle='dotted', label='1st Iteration')
    plt.hist(user_params_02[wt], bins=qbins, histtype='step', cumulative=True, color='black', linestyle='dashed', label='2nd')
    #plt.hist(user_params_03[wt], bins=qbins, histtype='step', cumulative=True, color='blue', linestyle='dotted', linewidth=2, label='3rd')
    #plt.hist(user_params_04[wt], bins=qbins, histtype='step', cumulative=True, color='green', linestyle='dashed', linewidth=2, label='4th')
    plt.hist(user_params[wt],    bins=qbins, histtype='step', cumulative=True, color='black', linestyle='solid', label='3rd-5th')
    plt.yscale('log', nonposy='clip')
    plt.xlim(subx1, subx2)
    plt.ylim(suby1,suby2)
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.savefig('consistencies.png', facecolor='None', edgecolor='None')
    plt.savefig('consistencies.eps', facecolor='None', edgecolor='None')

    #plt.show()
    plt.close()

    # plt.subplot(122)
    # wt = 'weight_from_avg_consistency'
    #
    # plt.ylim(ylimits)
    # plt.xlim(xlimits)
    # plt.hist(user_params_01[wt], bins=qbins, histtype='step', cumulative=True, color='black', linestyle='dotted', label='1st Iteration')
    # plt.hist(user_params_02[wt], bins=qbins, histtype='step', cumulative=True, color='black', linestyle='dashed', label='2nd Iteration')
    # plt.hist(user_params[wt],    bins=qbins, histtype='step', cumulative=True, color='black', linestyle='solid',  label='3rd Iteration')
    # plt.yscale('log', nonposy='clip')
    #
    # plt.show()
    #plt.clf()
    #plt.cla()

#done
