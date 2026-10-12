"""Image-level preference inference; input counts supplied by the reproducer."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from .pipeline import image_ci


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty_like(p)
    result[order] = adjusted
    return result.tolist()


def summarize(rows):
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = row['comparison'], row['tone'], row['image_id']
        if key in seen:
            raise ValueError('Duplicate image in comparison')
        seen.add(key)
        counts = [int(row[k]) for k in ['win', 'tie', 'loss']]
        if min(counts) < 0 or sum(counts) < 1:
            raise ValueError('Invalid vote counts')
        groups[key[:2]].append(counts)
    if set(groups) != {(c, t) for c in ['AW_vs_Neutral', 'AW_vs_Direct'] for t in ['joy', 'sadness']}:
        raise ValueError('Expected the four pre-specified AW comparisons')
    results = []
    for (comparison, tone), values in sorted(groups.items()):
        counts = np.asarray(values)
        margin = (counts[:, 0] - counts[:, 2]) / counts.sum(1)
        positive, negative = int((margin > 0).sum()), int((margin < 0).sum())
        n = positive + negative
        p = min(1., 2 * sum(math.comb(n, k) for k in range(min(positive, negative) + 1)) / 2**n) if n else 1.
        totals = counts.sum(0)
        results.append({'comparison': comparison, 'tone': tone, 'votes': totals.tolist(),
                        'pooled_shares': (totals / totals.sum()).tolist(),
                        'positive_zero_negative_images': [positive, int((margin == 0).sum()), negative],
                        'margin': image_ci(margin), 'sign_p': p})
    for row, p in zip(results, holm([r['sign_p'] for r in results])):
        row['holm_p'] = p
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with open(args.input, newline='') as handle:
        result = summarize(list(csv.DictReader(handle)))
    with Path(args.output).open('x') as handle:
        json.dump(result, handle, indent=2)


if __name__ == '__main__':
    main()
