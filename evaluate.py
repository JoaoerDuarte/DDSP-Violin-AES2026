"""Parameter recovery on the test split: the two result tables of the paper.

    python evaluate.py [prefix]        # every finished run under runs/, or those starting with prefix

Runs without the DDSP-Violin source (the baseline) are skipped.

Brightness: partial Spearman correlation of α with bow force and with bow speed, per string,
controlling for the other two bowing parameters and the semitone.

Bow position: the node order predicted for each frame. Accuracy for each node order k = 2..5 and
their mean (Mean/k), the false-positive rate FP (non-node samples or frames predicted as a node) and
the macro F1 over the five classes, at sample level (majority vote over the loud frames) and at frame
level (every loud frame). The continuous-β ablation is mapped to a node order by |β - 1/k| < 0.1/k.

Runs <config>, <config>_v2, ... are seeds of one config, reported as mean ± standard deviation.
Writes output/evaluate/results.txt and results.csv (one row per run). Predictions are cached per run
in output/evaluate/cache; FORCE=1 recomputes them.
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr

from ddsp_torch.model import DDSP
from ddsp_torch.model_utils import NODE_ORDERS
from ddsp_torch.train_util import LOUD_DB
from preprocess import data_dirs

RUNS = Path('runs')
OUT = Path('output/evaluate')
STRINGS = ('G', 'D', 'A', 'E')
NODES = [k for k in NODE_ORDERS if k > 0]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_run(run):
    """Model (in eval mode) and config of a finished run."""
    config = yaml.safe_load((RUNS / run / 'config.yaml').read_text())
    model = DDSP(**config['model']).to(DEVICE)
    model.load_state_dict(torch.load(RUNS / run / 'final_state.pth', map_location=DEVICE, weights_only=True))
    return model.eval(), config


def load_test_set(config):
    """Preprocessed test arrays and the manifest row of each segment."""
    _, folder = data_dirs(config, 'test')
    arrays = {name: np.load(Path(folder) / f'{name}.npy')
              for name in ('pitches', 'loudness', 'signals', 'harmonic_amps', 'filenames')}
    manifest_dir = Path(config.get('data', {}).get('manifest_dir', 'dataset/manifests'))
    manifest = pd.concat([pd.read_csv(manifest_dir / f'test_k{k}.csv') for k in NODE_ORDERS])
    manifest = manifest.set_index('filename').loc[arrays.pop('filenames')].reset_index()
    return arrays, manifest


def predict(run):
    """Per test sample: median α over the loud frames, majority node order and the number of loud
    frames predicted as each class, with the sample's bowing controls (cached)."""
    cache = OUT / 'cache' / f'{run}.npz'
    if cache.exists() and not os.environ.get('FORCE'):
        return dict(np.load(cache))
    model, config = load_run(run)
    arrays, manifest = load_test_set(config)
    n = len(manifest)
    alpha, k_pred = np.full(n, np.nan), np.full(n, np.nan)
    frame_counts = np.zeros((n, len(NODE_ORDERS)), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, n, 16):
            batch = slice(start, min(start + 16, n))
            x = {name: torch.tensor(arrays[name][batch], dtype=torch.float32, device=DEVICE) for name in arrays}
            diag = model(x['pitches'].unsqueeze(-1), x['loudness'].unsqueeze(-1), audio=x['signals'],
                         harmonic_amps=x['harmonic_amps'])['violin_diagnostics']
            alphas = diag['α'].squeeze(-1).cpu().numpy()
            if 'softmax_k_class' in diag:
                frame_classes = np.array(NODE_ORDERS)[diag['softmax_k_class'].cpu().numpy()]
            else:
                beta = diag['β'].squeeze(-1).cpu().numpy()
                frame_classes = np.zeros(beta.shape, dtype=int)
                for k in NODES:
                    frame_classes[np.abs(beta - 1.0 / k) < 0.10 / k] = k
            for b, loud in enumerate(arrays['loudness'][batch] > LOUD_DB):
                if loud.sum() < 10:
                    continue
                i = start + b
                alpha[i] = float(np.median(alphas[b, loud]))
                classes = frame_classes[b, loud]
                frame_counts[i] = [(classes == c).sum() for c in NODE_ORDERS]
                values, counts = np.unique(classes, return_counts=True)
                k_pred[i] = values[np.argmax(counts)]
    prediction = dict(alpha=alpha, k_pred=k_pred, frame_counts=frame_counts,
                      **{c: manifest[c].to_numpy() for c in ('F', 'vb', 'z0', 'semitone', 'k_class')},
                      string=manifest['string'].to_numpy().astype(str))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **prediction)
    return prediction


def partial_spearman(df, x, y, controls):
    """Spearman correlation of x and y after removing their least-squares fit on the controls."""
    df = df.dropna(subset=[x, y] + controls)
    if len(df) < 30:
        return np.nan
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in controls])

    def residual(column):
        values = df[column].to_numpy(float)
        return values - X @ np.linalg.lstsq(X, values, rcond=None)[0]

    return spearmanr(residual(x), residual(y))[0]


def brightness(prediction):
    df = pd.DataFrame({c: prediction[c] for c in ('alpha', 'F', 'vb', 'z0', 'semitone', 'string')})
    cells = {}
    for s in STRINGS:
        rows = df[df.string == s]
        cells[f'{s} f_b'] = partial_spearman(rows, 'F', 'alpha', ['z0', 'vb', 'semitone'])
        cells[f'{s} v_b'] = partial_spearman(rows, 'vb', 'alpha', ['z0', 'F', 'semitone'])
    return cells


def macro_f1(confusion):
    """Unweighted mean of the per-class F1 (%) of a confusion matrix (rows: true class)."""
    f1s = []
    for c in range(len(confusion)):
        tp = confusion[c, c]
        fn, fp = confusion[c].sum() - tp, confusion[:, c].sum() - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1s.append(200.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)
    return float(np.mean(f1s))


def bow_position(counts, true_k):
    """Accuracy per node order, Mean/k, FP and macro F1 (%) from the counts [n, classes] of samples
    or frames predicted as each class."""
    counts = counts.astype(np.float64)
    scored = counts.sum(1) > 0
    confusion = np.stack([counts[scored & (true_k == k)].sum(0) for k in NODE_ORDERS])
    totals = confusion.sum(1)
    accuracy = {k: confusion[i, i] / totals[i] * 100 if totals[i] else np.nan
                for i, k in enumerate(NODE_ORDERS) if k > 0}
    fp = (totals[0] - confusion[0, 0]) / totals[0] * 100 if totals[0] else np.nan
    return accuracy, float(np.nanmean(list(accuracy.values()))), fp, macro_f1(confusion)


def mean_std(values, fmt):
    values = values.dropna()
    if values.empty:
        return '-'
    if len(values) == 1:
        return format(values.iloc[0], fmt)
    return f"{format(values.mean(), fmt)}±{format(values.std(), fmt.lstrip('+'))}"


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else ''
    runs = sorted(d.name for d in RUNS.glob(f'{prefix}*') if (d / 'final_state.pth').exists())
    if not runs:
        sys.exit(f'no finished runs matching runs/{prefix}*')
    rows = []
    for run in runs:
        if yaml.safe_load((RUNS / run / 'config.yaml').read_text())['model'].get('source') is None:
            continue
        prediction = predict(run)
        row = {'run': run, 'config': re.sub(r'_v\d+$', '', run), **brightness(prediction)}
        sample_counts = (prediction['k_pred'][:, None] == np.array(NODE_ORDERS)).astype(np.int64)
        for level, counts in (('sample', sample_counts), ('frame', prediction['frame_counts'])):
            accuracy, mean_k, fp, f1 = bow_position(counts, prediction['k_class'])
            row.update({f'{level} k={k}': value for k, value in accuracy.items()})
            row.update({f'{level} Mean/k': mean_k, f'{level} FP': fp, f'{level} F1': f1})
        rows.append(row)
        print(f'evaluated {run}')
    if not rows:
        sys.exit('no finished runs with the DDSP-Violin source')
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / 'results.csv', index=False)

    bright = [f'{s} {p}' for s in STRINGS for p in ('f_b', 'v_b')]
    bow = [f'k={k}' for k in NODES] + ['Mean/k', 'FP', 'F1']
    lines = ['Brightness: partial Spearman correlation of α with bow force (f_b) and bow speed (v_b), '
             'mean ± std over seeds', '',
             f"{'config':<32} {'n':>2}  " + ' '.join(f'{c:>11}' for c in bright)]
    for config, runs_of_config in df.groupby('config'):
        lines.append(f'{config:<32} {len(runs_of_config):>2}  '
                     + ' '.join(f'{mean_std(runs_of_config[c], "+.2f"):>11}' for c in bright))
    lines += ['', 'Bow position (%): accuracy per node order, Mean/k, false-positive rate (FP) and macro F1, '
              'mean ± std over seeds', '',
              f"{'config':<32} {'level':<6} {'n':>2}  " + ' '.join(f'{c:>11}' for c in bow)]
    for config, runs_of_config in df.groupby('config'):
        for level in ('sample', 'frame'):
            name = config if level == 'sample' else ''
            lines.append(f'{name:<32} {level:<6} {len(runs_of_config):>2}  '
                         + ' '.join(f'{mean_std(runs_of_config[f"{level} {c}"], ".1f"):>11}' for c in bow))
    (OUT / 'results.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
