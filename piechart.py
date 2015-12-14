from matplotlib import pyplot as plt
import numpy as np
from collections import Counter
import re
import pandas as pd

"""

piechart
=========

Generate pie chart of the plurality classifications for the CANDELS data.

Kyle Willett (UMN) - 10 Dec 2015

"""

def survey_dict():

    # Information about the specific group settings in the project

    d = {u'candels':        {'name':u'CANDELS','retire_limit':80},
        u'candels_2epoch':  {'name':u'CANDELS 2-epoch','retire_limit':80},
        u'decals':          {'name':u'DECaLS','retire_limit':40},
        u'ferengi':         {'name':u'FERENGI','retire_limit':40},
        u'goods_full':      {'name':u'GOODS full-depth','retire_limit':40},
        u'illustris':       {'name':u'Illustris','retire_limit':40},
        u'sloan_singleband':{'name':u'SDSS single-band','retire_limit':40},
        u'ukidss':          {'name':u'UKIDSS','retire_limit':40}}
        #u'sloan':           {'name':u'SDSS DR8','retire_limit':60},        # Memory error - can't collate the full classification file in pandas

    return d
    
def is_number(s):

    # Is a string a representation of a number?

    try:
        int(s)
        return True
    except ValueError:
        return False

def plurality(datarow,survey='candels',check_threshold = 0.50):

    """ Determine the plurality for the consensus GZ2 classification of a
        galaxy's morphology.
    
    Parameters
    ----------
    datarow : astropy.io.fits.fitsrec.FITS_record
        Iterated element (row) of a final
        GZ2 table, containing all debiased probabilities
        and vote counts
    
    survey : string indicating the survey group that defines
        the workflow/decision tree. Default is 'decals'. 
        Possible options should be:
            
            'candels'
            'candels_2epoch'
            'decals'
            'ferengi'
            'goods_full'
            'illustris'
            'sloan',
            'sloan_singleband'
            'ukidss'

    check_threshold: float indicating the threshold plurality level for
        checkbox questions. If no questions meet this, don't select any answer.
    
    Returns
    -------
    task_eval: array [N]
        1 if question was answered by the plurality path through the tree; 0 if not
    
    task_ans: array [N]
        Each answer gives the index of most common answer
            regardless of if it was in the plurality path or not.
    
    Notes
    -------
    
    """
    
    weights = datarow
    
    if survey in ('candels','candels_2epoch'):
        
        d = { 0:{'idx':0 ,'len':3},       # 'Shape', 'Is the galaxy simply smooth and rounded, with no sign of a disk?', ->
              1:{'idx':3 ,'len':3},       # 'Round', 'How rounded is it?', leadsTo: 'Is there anything odd?', ->
              2:{'idx':6 ,'len':2},       # 'Clumps', 'Does the galaxy have a mostly clumpy appearance?', ->
              3:{'idx':8 ,'len':6},       # 'Clumps', 'How many clumps are there?', leadsTo: 'Do the clumps appear in a straight line, a chain, or a cluster?', ->
              4:{'idx':14,'len':4},       # 'Clumps', 'Do the clumps appear in a straight line, a chain, or a cluster?', leadsTo: 'Is there one clump which is clearly brighter than the others?', ->
              5:{'idx':18,'len':2},       # 'Clumps', 'Is there one clump which is clearly brighter than the others?', ->
              6:{'idx':20,'len':2},       # 'Clumps', 'Is the brightest clump central to the galaxy?', ->
              7:{'idx':22,'len':2},       # 'Symmetry', 'Does the galaxy appear symmetrical?', leadsTo: 'Do the clumps appear to be embedded within a larger object?', ->
              8:{'idx':24,'len':2},       # 'Clumps', 'Do the clumps appear to be embedded within a larger object?', leadsTo: 'Is there anything odd?', ->
              9:{'idx':26,'len':2},       # 'Disk', 'Could this be a disk viewed edge-on?', ->
             10:{'idx':28,'len':2},       # 'Bulge', 'Does the galaxy have a bulge at its center?', leadsTo: 'Is there anything odd?', ->
             11:{'idx':30,'len':2},       # 'Bar', 'Is there any sign of a bar feature through the centre of the galaxy?', leadsTo: 'Is there any sign of a spiral arm pattern?', ->
             12:{'idx':32,'len':2},       # 'Spiral', 'Is there any sign of a spiral arm pattern?', ->
             13:{'idx':34,'len':3},       # 'Spiral', 'How tightly wound do the spiral arms appear?', leadsTo: 'How many spiral arms are there?', ->
             14:{'idx':37,'len':6},       # 'Spiral', 'How many spiral arms are there?', leadsTo: 'How prominent is the central bulge, compared with the rest of the galaxy?', ->
             15:{'idx':43,'len':3},       # 'Bulge', 'How prominent is the central bulge, compared with the rest of the galaxy?', leadsTo: 'Is there anything odd?', ->
             16:{'idx':46,'len':4}}       #  Merging/tidal debris

        task_eval = [0]*len(d)
        task_ans  = [0]*len(d)
        
        # Top-level: smooth/features/artifact
        task_eval[0] = 1
        
        if weights[d[0]['idx']:d[0]['idx']+d[0]['len']].argmax() < 2:

            # Smooth galaxies
            if weights[d[0]['idx']:d[0]['idx']+d[0]['len']].argmax() == 0:
                # Roundness
                task_eval[1] = 1

            # Features/disk galaxies
            if weights[d[0]['idx']:d[0]['idx']+d[0]['len']].argmax() == 1:
                task_eval[2] = 1

                # Clumpy question
                if weights[d[2]['idx']] > weights[d[2]['idx']+1]:

                    # Clumpy galaxies
                    task_eval[3] = 1
                    if weights[d[3]['idx']:d[3]['idx'] + d[3]['len']].argmax() > 0:
                        # Multiple clumps
                        if weights[d[3]['idx']:d[3]['idx'] + d[3]['len']].argmax() > 1:
                            # One bright clump
                            task_eval[4] = 1
                        task_eval[5] = 1
                        if weights[d[5]['idx']] > weights[d[5]['idx']+1]:
                            # Bright clump symmetrical
                            task_eval[6] = 1
                    if weights[d[6]['idx']] > weights[d[6]['idx']+1]:
                        task_eval[7] = 1
                        task_eval[8] = 1

                else:
                    # Disk galaxies
                    task_eval[9] = 1
                    # Edge-on disks
                    if weights[d[9]['idx']] > weights[d[9]['idx']+1]:
                        # Bulge shape
                        task_eval[10] = 1
                    # Not edge-on disks
                    else:
                        task_eval[11] = 1
                        task_eval[12] = 1
                        if weights[d[12]['idx']] > weights[d[12]['idx']+1]:
                            # Spirals
                            task_eval[13] = 1
                            task_eval[14] = 1
                        task_eval[15] = 1
            
            # Merging/tidal debris
            task_eval[16] = 1

    # Assign the plurality task numbers

    for i,t in enumerate(task_ans):
        try:
            task_ans[i]  = weights[d[i]['idx']:d[i]['idx'] + d[i]['len']].argmax() + d[i]['idx']
        except ValueError:
            print len(weights),len(task_ans)

    return task_eval,task_ans

def morphology_distribution(survey='candels'):

    # What's the plurality distribution of morphologies?

    # Get weights
    try:
        collation_file = "../gz_reduction_sandbox/data/candels_weighted_seeded_collated_05_wdup_nobots.csv"
        collated = pd.read_csv(collation_file)
    except IOError:
        print "Collation file for {0:} does not exist. Aborting.".format(survey)
        return None

    columns = collated.columns

    fraccols,colnames = [],[]
    for c in columns:
        if c[-13:] == 'weighted_frac':
            fraccols.append(c)
        if c[0] == 't' and is_number(c[1:3]):
            colnames.append(c[:3])

    collist = list(set(colnames))
    collist.sort()

    # Plot distribution of vote fractions for each task

    ntasks = len(collist)
    ncols = 4 if ntasks > 9 else int(np.sqrt(ntasks))
    nrows = int(ntasks / ncols) if ntasks % ncols == 0 else int(ntasks / ncols) + 1

    sd = survey_dict()[survey]
    survey_name = sd['name']

    def f7(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))] 

    tasklabels = f7([re.split("[ax][0-9]",f)[0][4:-1] for f in fraccols])
    labels = [re.split("[ax][0-9]",f[4:-14])[-1][1:] for f in fraccols]

    # Make pie charts of the plurality votes

    votearr = np.array(collated[fraccols])
    class_arr,task_arr,task_ans = [],[],[]
    for v in votearr:
        e,a = plurality(v,survey) 
        task_arr.append(e)
        task_ans.append(a)

    task_arr = np.array(task_arr)
    task_ans = np.array(task_ans)

    fig,axarr = plt.subplots(nrows=nrows,ncols=ncols,figsize=(15,12))

    colors=[u'#377EB8', u'#E41A1C', u'#4DAF4A', u'#984EA3', u'#FF7F00',u'#A6761D',u'#1B9E77']

    n = (task_arr.shape)[1]
    for i in range(n):
        ax = axarr.ravel()[i]
        c = Counter(task_ans[:,i][task_arr[:,i] == True])
        pv,pl = [],[]
        for k in c:
            pv.append(c[k])
            pl.append(labels[k])
        ax.pie(pv,labels=pl,colors=colors,autopct='%1.0f%%')
        title = '{0:} - t{1:02} {2:}'.format(survey_name,i,tasklabels[i]) if i == 0 else 't{0:02} {1:}'.format(i,tasklabels[i])
        ax.set_title(title)
        ax.set_aspect('equal')

    # Remove empty axes from subplots
    if axarr.size > ntasks:
        for i in range(axarr.size - ntasks):
            ax = axarr.ravel()[axarr.size-(i+1)]
            ax.set_axis_off()

    fig.set_tight_layout(True)
    plt.savefig('pie_{0:}.eps'.format(survey))
    plt.close()

    return None

if __name__ == "__main__":

    morphology_distribution()
