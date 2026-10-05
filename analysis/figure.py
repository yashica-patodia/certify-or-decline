"""Plot the paper's Figure 1 from the released numerical evidence.

python -m pip install '.[figures]'
python -m analysis.figure --out figures/figure1.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from experiments.cluster_intervals import wilson


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':7})
    import matplotlib.pyplot as plt
    root = Path(__file__).resolve().parents[1]
    pub = root/'experiments/public_runs'
    read = lambda name: json.loads((pub/name).read_text())
    values = read('selective_prediction_value.json')
    curves = {k:read(f'self_consistency_k{k}_curve.json') for k in [8,30]}
    arms = json.loads((root/'analysis/arms.json').read_text())
    fig,axes = plt.subplots(1,2,figsize=(5.5,2.15))
    for benchmark,color,marker in [('GPQA','#1f77b4','o'),('AIME-2025','#e66c00','s'),('MATH-500','#00997a','^')]:
        rows = [r for r in values if r['benchmark']==benchmark]
        axes[0].scatter([r['no_verification_precision'] for r in rows],
                        [r['precision_gain'] for r in rows],c=color,marker=marker,s=20,label=benchmark)
    axes[0].set(xlabel='Initial-answer accuracy (%)',ylabel='Precision gain (points)',
                xlim=(40,101),ylim=(-3,53),title='(a) Gain versus initial-answer accuracy')
    axes[0].legend(frameon=False,fontsize=6)
    curve = curves[30]['by_denominator']['all_k']['points']
    axes[1].plot([r['coverage'] for r in curve],[r['precision'] for r in curve],
                 'o-',color='#7f7f7f',markersize=2.5,linewidth=.8,label='SC k=30, all draws')
    points=[]
    for name in arms['GPQA strong']:
        rows = read(name);selected = [r for r in rows if r['verified'] and not r['error']]
        points.append((100*len(selected)/len(rows),100*sum(r['key_match'] for r in selected)/len(selected)))
    axes[1].scatter(*zip(*points),marker='D',c='#da6b00',s=20,label='Verifier runs',zorder=4)
    point=next(r for r in curves[8]['by_denominator']['all_k']['points'] if r['coverage']==35)
    lo,hi=wilson(point['n_correct'],point['n_answered'])
    axes[1].errorbar([35],[point['precision']],yerr=[[point['precision']-lo],[hi-point['precision']]],
                     marker='s',color='#1f77b4',markersize=4,linewidth=.8,label='SC k=8, all draws')
    point=curves[30]['by_denominator']['parsed']['points'][0]
    axes[1].scatter([point['coverage']],[point['precision']],facecolors='none',edgecolors='#00997a',s=30,
                    label='SC k=30, answering draws',zorder=4)
    axes[1].set(xlabel='Coverage (%)',ylabel='Precision (%)',xlim=(18,47),ylim=(65,101),
                title='(b) Verifier and self-consistency')
    axes[1].legend(frameon=False,fontsize=5.5,loc='lower left')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.2,linewidth=.5)
    fig.tight_layout(pad=.6)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out,bbox_inches='tight')
    print(f'Wrote {args.out}; verifier points: {points}')


if __name__ == '__main__':main()
