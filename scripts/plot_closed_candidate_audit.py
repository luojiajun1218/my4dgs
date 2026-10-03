"""Plot preselected candidate loss/gradient evidence, without selecting nodes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    rows = json.loads((args.run / 'rows.json').read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for column, split in enumerate(('train', 'validation')):
        selected = [row for row in rows if row['split'] == split]
        axis = axes[0, column]
        for level in range(3):
            group = [row for row in selected if row['level'] == level]
            axis.scatter([row['loss_change'] * 1e6 for row in group],
                         [row['gate_share_minus_keep_gradient'] * 1e6 for row in group],
                         label=f'Level{level}', alpha=.7, s=25)
        axis.axvline(0, color='gray', linewidth=.8)
        axis.axhline(0, color='gray', linewidth=.8)
        axis.set_xlabel('Actual SHARE minus KEEP image loss (x1e6)')
        axis.set_ylabel('Gate SHARE minus KEEP gradient (x1e6)')
        axis.set_title(f'{split}: lower right = worse candidate rewarded')
        axis.legend()
        cosine = [row['motion_gradient_cosine'] for row in selected
                  if row['motion_gradient_cosine'] is not None]
        axis = axes[1, column]
        axis.hist(cosine, bins=[-.999 - .001 + i * .1 for i in range(21)], color='steelblue', alpha=.8)
        axis.axvline(0, color='gray', linewidth=.8)
        axis.set_xlabel('True candidate / hard-forward proxy motion gradient cosine')
        axis.set_ylabel('Preselected node-view pairs')
        axis.set_title(f'{sum(value < 0 for value in cosine)}/{len(cosine)} update directions oppose')
    fig.suptitle('Fixed frame5:24 image-blind KEEP nodes,4 train and4 held-out cameras')
    fig.tight_layout()
    fig.savefig(args.run / 'candidate_gradient_audit.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
