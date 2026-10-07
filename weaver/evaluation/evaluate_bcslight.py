# Do model evaluation for jet flavour tagging

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

thisdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(thisdir)

from tools import read_file
from plot_roc_multi import plot_scores_multi
from plot_roc_multi import plot_roc_multi
try:
    from plot_ternary import plot_ternary
    do_plot_ternary = True
except:
    do_plot_ternary = False


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfiles', required=True, nargs='+')
    args = parser.parse_args()

    # hard coded settings
    treename = 'Events'
    signal_categories = {
        'b': {
            'label_branch': 'recojet_match_isB',
            'score_branch': 'score_recojet_match_isB',
            'color': 'red',
            'label': r'$b$-jets'
        },
        'bbar': {
            'label_branch': 'recojet_match_isBbar',
            'score_branch': 'score_recojet_match_isBbar',
            'color': 'orange',
            'label': r'$\bar{b}$-jets'
        },
        'c': {
            'label_branch': 'recojet_match_isC',
            'score_branch': 'score_recojet_match_isC',
            'color': 'purple',
            'label': r'$c$-jets'
        },
        'cbar': {
            'label_branch': 'recojet_match_isCbar',
            'score_branch': 'score_recojet_match_isCbar',
            'color': 'brown',
            'label': r'$\bar{c}$-jets'
        },
        's': {
            'label_branch': 'recojet_match_isS',
            'score_branch': 'score_recojet_match_isS',
            'color': 'blue',
            'label': r'$s$-jets'
        },
        'sbar': {
            'label_branch': 'recojet_match_isSbar',
            'score_branch': 'score_recojet_match_isSbar',
            'color': 'cyan',
            'label': r'$\bar{s}$-jets'
        }
    }
    background_categories = {
        'u': {
            'label_branch': 'recojet_match_isU',
            'score_branch': 'score_recojet_match_isU',
            'color': 'green',
            'label': r'$u$-jets'
        },
        'ubar': {
            'label_branch': 'recojet_match_isUbar',
            'score_branch': 'score_recojet_match_isUbar',
            'color': 'lightgreen',
            'label': r'$\bar{u}$-jets'
        },
        'd': {
            'label_branch': 'recojet_match_isD',
            'score_branch': 'score_recojet_match_isD',
            'color': 'magenta',
            'label': r'$d$-jets'
        },
        'dbar': {
            'label_branch': 'recojet_match_isDbar',
            'score_branch': 'score_recojet_match_isDbar',
            'color': 'pink',
            'label': r'$\bar{d}$-jets'
        }
    }
    all_categories = {**signal_categories, **background_categories}

    # find all branches to read
    branches_to_read = (
        [cat['label_branch'] for cat in all_categories.values()]
        + [cat['score_branch'] for cat in all_categories.values()]
    )

    # loop over input files
    for inputfile in args.inputfiles:
        print(f'Running on input file {inputfile}...')

        # load events
        events = read_file(
                   inputfile,
                   treename = treename,
                   branches = branches_to_read)
        keys = list(events.keys())
        nevents = len(events[list(events.keys())[0]])
        print('Read events file with following properties:')
        print(f'  - Keys: {keys}.')
        print(f'  - Number of events: {nevents}.')

        # define output directory
        outputdir = inputfile.replace('.root', '_plots')

        # plot the score distributions and ROC curves
        print(f'    Plotting ROC...')
        plot_scores_multi(
            events,
            all_categories,
            outputdir = outputdir)
        plot_roc_multi(
            events,
            signal_categories,
            background_categories,
            outputdir = outputdir)
        plt.close()
