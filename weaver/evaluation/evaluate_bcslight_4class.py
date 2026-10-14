# Do model evaluation for jet flavour tagging

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

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
    parser.add_argument('--compare', action='store_true',
                        help='Compare 10-class (charge, first file) and 4-class (no charge, second file) in one plot')
    args = parser.parse_args()

    # hard coded settings
    treename = 'Events'
    signal_categories = {
        'b': {
            'label_branch': 'recojet_match_isB | recojet_match_isBbar',
            'score_branch': '(score_recojet_match_isB + score_recojet_match_isBbar)/(score_recojet_match_isB + score_recojet_match_isBbar + score_recojet_match_isC + score_recojet_match_isCbar + score_recojet_match_isS + score_recojet_match_isSbar + score_recojet_match_isU + score_recojet_match_isUbar + score_recojet_match_isD + score_recojet_match_isDbar)',
            'color': 'red',
            'label': r'$b$-jets'
        },
        'c': {
            'label_branch': 'recojet_match_isC | recojet_match_isCbar',
            'score_branch': '(score_recojet_match_isC + score_recojet_match_isCbar)/(score_recojet_match_isB + score_recojet_match_isBbar + score_recojet_match_isC + score_recojet_match_isCbar + score_recojet_match_isS + score_recojet_match_isSbar + score_recojet_match_isU + score_recojet_match_isUbar + score_recojet_match_isD + score_recojet_match_isDbar)',
            'color': 'purple',
            'label': r'$c$-jets'
        },
        's': {
            'label_branch': 'recojet_match_isS | recojet_match_isSbar',
            'score_branch': '(score_recojet_match_isS + score_recojet_match_isSbar)/(score_recojet_match_isB + score_recojet_match_isBbar + score_recojet_match_isC + score_recojet_match_isCbar + score_recojet_match_isS + score_recojet_match_isSbar + score_recojet_match_isU + score_recojet_match_isUbar + score_recojet_match_isD + score_recojet_match_isDbar)',
            'color': 'blue',
            'label': r'$s$-jets'
        },
    }
    background_categories = {
        'u': {
            'label_branch': 'recojet_match_isU | recojet_match_isUbar | recojet_match_isD | recojet_match_isDbar',
            'score_branch': '(score_recojet_match_isU + score_recojet_match_isUbar + score_recojet_match_isD + score_recojet_match_isDbar)/(score_recojet_match_isB + score_recojet_match_isBbar + score_recojet_match_isC + score_recojet_match_isCbar + score_recojet_match_isS + score_recojet_match_isSbar + score_recojet_match_isU + score_recojet_match_isUbar + score_recojet_match_isD + score_recojet_match_isDbar)',
            'color': 'green',
            'label': r'$ud$-jets'
        }
    }
    all_categories = {**signal_categories, **background_categories}

    # categories of the 4-class (no charge) model, same keys as above
    nocharge_categories = {
        'b': {'label_branch': 'recojet_isB', 'score_branch': 'score_recojet_isB'},
        'c': {'label_branch': 'recojet_isC', 'score_branch': 'score_recojet_isC'},
        's': {'label_branch': 'recojet_isS', 'score_branch': 'score_recojet_isS'},
        'u': {'label_branch': 'recojet_isUDG', 'score_branch': 'score_recojet_isUDG'},
    }

    # find all branches to read
    branches_to_read = (
        [cat['label_branch'] for cat in all_categories.values()]
        + [cat['score_branch'] for cat in all_categories.values()]
    )

    # compare charge (solid) and no charge (dashed) results in one plot
    if args.compare:
        charge_file, nocharge_file = args.inputfiles
        events = {
            '-': read_file(charge_file, treename=treename, branches=branches_to_read),
            '--': read_file(nocharge_file, treename=treename,
                    branches=[cat['label_branch'] for cat in nocharge_categories.values()]
                    + [cat['score_branch'] for cat in nocharge_categories.values()]),
        }
        categories = {'-': all_categories, '--': nocharge_categories}
        names = list(all_categories.keys())
        pairs = [(s, b) for sidx, s in enumerate(names) for b in names[sidx+1:]]

        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = plt.get_cmap('cool', len(pairs))
        for linestyle, model in [('-', 'charge'), ('--', 'no charge')]:
            for cidx, (sig, bkg) in enumerate(pairs):
                cats = categories[linestyle]
                scores = events[linestyle][cats[sig]['score_branch']]
                scores_sig = scores[events[linestyle][cats[sig]['label_branch']].astype(bool)]
                scores_bkg = scores[events[linestyle][cats[bkg]['label_branch']].astype(bool)]
                this_scores = np.concatenate((scores_sig, scores_bkg))
                this_labels = np.concatenate((np.ones(len(scores_sig)), np.zeros(len(scores_bkg))))
                fpr, tpr, _ = roc_curve(this_labels, this_scores)
                auc = roc_auc_score(this_labels, this_scores)
                label = all_categories[sig]['label'] + ' vs. ' + all_categories[bkg]['label']
                label += f' ({model}, AUC: {auc:.3f})'
                ax.plot(fpr, tpr, color=cmap(cidx), linewidth=2, linestyle=linestyle, label=label)

        # add reference
        ax.scatter(0.00216, 0.1957, s=35, color=cmap(0), edgecolors='red', label='$b$-jets vs. $c$-jets (ALEPH)')
        ax.scatter(0.00043, 0.1957, s=35, color=cmap(1), edgecolors='red', label='$b$-jets vs. $uds$-jets (ALEPH)')
        ax.set_xlabel('Background pass-through', fontsize=15)
        ax.set_ylabel('Signal efficiency', fontsize=15)
        ax.grid(which='both')
        ax.set_ylim((-0.05, 1.1))
        ax.legend(fontsize=7)
        outputdir = charge_file.replace('.root', '_plots')
        if not os.path.exists(outputdir): os.makedirs(outputdir)
        fig.tight_layout()
        fig.savefig(os.path.join(outputdir, 'roc_compare.png'))
        ax.set_xscale('log')
        ax.set_xlim((1e-5, 1))
        fig.tight_layout()
        fig.savefig(os.path.join(outputdir, 'roc_compare_log.png'))
        print(f'Saved comparison plots in {outputdir}.')
        sys.exit()

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
