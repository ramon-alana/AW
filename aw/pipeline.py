"""Small, explicit CLI for development, validation, and frozen test evaluation.

All generated artifacts are local. No cluster, credentials, or dataset downloads.
Run from the repository root. Separate environments are required per model.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from .common import append_record, digest, load_config, read_jsonl, seed_all, write_json
from .core import direction, choose_window, degeneration


def fingerprint(path):
    from PIL import Image, ImageOps
    with Image.open(path) as src:
        im = ImageOps.exif_transpose(src).convert('RGB')
    pixel = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
    gray = np.asarray(im.convert('L').resize((9, 8), Image.Resampling.LANCZOS))
    perceptual = int(''.join('1' if b else '0' for b in (gray[:, :-1] > gray[:, 1:]).flat), 2)
    return digest(path), pixel, perceptual


def manifest(cfg):
    path = Path(cfg['data']['manifest']).resolve()
    rows = read_jsonl(path)
    seen = set()
    for row in rows:
        if not all(k in row for k in ['image_id', 'image_path', 'partition', 'label']):
            raise ValueError('Manifest requires image_id,image_path,partition,label')
        if row['image_id'] in seen:
            raise ValueError('Duplicate image_id')
        seen.add(row['image_id'])
        if row['partition'] not in cfg['data']['expected_counts']:
            raise ValueError('Unknown partition')
        if row['label'] not in cfg['data']['labels']:
            raise ValueError('Unknown image-emotion label')
        image = Path(row['image_path'])
        row['image_path'] = str(image if image.is_absolute() else path.parent / image)
    if dict(Counter(r['partition'] for r in rows)) != cfg['data']['expected_counts']:
        raise ValueError('Partition sizes differ from the declared protocol')
    return sorted(rows, key=lambda r: r['image_id'])


def audit_manifest(cfg):
    rows = manifest(cfg)
    seen_file, seen_pixel, seen_source, previous = {}, {}, {}, []
    for row in rows:
        file_sha, pixel_sha, phash = fingerprint(row['image_path'])
        part = row['partition']
        # Exact duplicate identities are forbidden even within a partition.
        for seen, key in [(seen_file, file_sha), (seen_pixel, pixel_sha),
                          (seen_source, row.get('source_id', row['image_id']))]:
            if key in seen:
                raise ValueError('Repeated file/pixel/source identity: ' + row['image_id'])
            seen[key] = part
        for old_part, old_hash in previous:
            if old_part != part and (old_hash ^ phash).bit_count() <= cfg['data']['near_duplicate_hamming']:
                raise ValueError('Near duplicate across partitions: ' + row['image_id'])
        previous.append((part, phash))
        if row['image_id'] != pixel_sha[:24]:
            raise ValueError('image_id must be first 24 hex characters of decoded pixel SHA256')
        row.update(image_sha256=file_sha, pixel_sha256=pixel_sha, dhash64=f'{phash:016x}')
    for part, n in [('development', 8), ('validation', 3)]:
        counts = Counter(r['label'] for r in rows if r['partition'] == part)
        if counts != Counter({label: n for label in cfg['data']['labels']}):
            raise ValueError('Development/validation must be stratified 8/3 images per class')
    return rows


def adapter(cfg):
    if cfg['model_key'] == 'qwen':
        from .model import OmniAdapter as Model
    elif cfg['model_key'] == 'phi':
        from .phi_adapter_v1 import PhiAdapter as Model
    elif cfg['model_key'] == 'minicpm':
        from .minicpm_adapter_v1 import MiniCPMAdapter as Model
    else:
        raise ValueError('Unsupported model; no fallback')
    return Model(cfg)


def context(config):
    cfg = load_config(config)
    stamp = {'config_sha256': digest(config), 'manifest_sha256': digest(cfg['data']['manifest'])}
    return cfg, Path(cfg['output']), stamp


def checked(base, stage, stamp, outputs=()):
    folder = base / stage
    receipt = load_config(folder / 'complete.json')
    if receipt['inputs'] != stamp:
        raise ValueError('Configuration/manifest changed; use a new output directory')
    for name in outputs:
        if receipt['outputs'][name] != digest(folder / name):
            raise ValueError('Changed stage output: ' + name)
    return receipt


@contextmanager
def stage(base, name, stamp):
    folder = base / name
    # A pre-existing directory is deliberately not overwritten or auto-resumed.
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / 'started.json', stamp)
    yield folder
    outputs = {p.name: digest(p) for p in folder.iterdir() if p.is_file() and p.name != 'started.json'}
    write_json(folder / 'complete.json', {'inputs': stamp, 'outputs': outputs})


def verify_images(rows, audit):
    indexed = {r['image_id']: r for r in audit}
    for row in rows:
        if digest(row['image_path']) != indexed[row['image_id']]['image_sha256']:
            raise ValueError('Image changed after audit')


def development(cfg, base, stamp):
    checked(base, 'audit', stamp, ['manifest.json'])
    rows = [r for r in manifest(cfg) if r['partition'] != 'test']
    verify_images(rows, load_config(base / 'audit/manifest.json'))
    with stage(base, 'extract', stamp) as out:
        seed_all(cfg['seed'])
        model = adapter(cfg)
        features, metadata = [], []
        for row in rows:
            conditions = ['base', 'neutral', 'joy', 'sadness'] if row['partition'] == 'development' else ['base']
            for condition in conditions:
                states, tokens = model.extract(row['image_path'], model.prompt(None if condition == 'base' else condition))
                features.append(states)
                metadata.append(dict(row, condition=condition, input_tokens=tokens))
                logging.info('Extracted %s %s', row['image_id'], condition)
        np.savez_compressed(out / 'states.npz', states=np.stack(features))
        write_json(out / 'rows.json', metadata)


def fit(cfg, base, stamp):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    checked(base, 'extract', stamp, ['states.npz', 'rows.json'])
    rows = load_config(base / 'extract/rows.json')
    states = np.load(base / 'extract/states.npz')['states']
    def ids(part, condition):
        return [i for i, row in enumerate(rows) if row['partition'] == part and row['condition'] == condition]
    with stage(base, 'fit', stamp) as out:
        neutral = ids('development', 'neutral')
        vectors = {}
        for tone in cfg['directions']['emotions']:
            target = ids('development', tone)
            if [rows[i]['image_id'] for i in neutral] != [rows[i]['image_id'] for i in target]:
                raise ValueError('Unpaired direction inputs')
            vectors[tone] = np.stack([direction(states[neutral, l], states[target, l], cfg['directions']['remove_pcs'])[0]
                                     for l in range(states.shape[1])])
        np.savez_compressed(out / 'directions.npz', **vectors)
        train, val = ids('development', 'base'), ids('validation', 'base')
        y, yv = [rows[i]['label'] for i in train], [rows[i]['label'] for i in val]
        cv = StratifiedKFold(n_splits=cfg['probe']['folds'], shuffle=True, random_state=cfg['seed'])
        scores = []
        for l in range(states.shape[1]):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=cfg['probe']['C'], max_iter=cfg['probe']['max_iter'], random_state=cfg['seed']))
            cv_score = cross_val_score(model, states[train, l], y, cv=cv).mean()
            model.fit(states[train, l], y)
            scores.append({'layer': l + 1, 'development_cv_accuracy': float(cv_score), 'validation_accuracy': float(model.score(states[val, l], yv))})
        write_json(out / 'probe.json', scores)


def random_bands(model, tone, n_layers):
    starts = sorted(range(1, n_layers - 2), key=lambda k: hashlib.sha256(f'AW-RANDOM-BAND-v1|42|{model}|{tone}|{k}'.encode()).hexdigest())[:5]
    return [list(range(s, s + 4)) for s in starts]


def scan_conditions(cfg):
    yield {'name': 'baseline', 'tone': None, 'layers': [], 'prompt': None}
    for tone in cfg['directions']['emotions']:
        yield {'name': tone + '_prompt', 'tone': tone, 'layers': [], 'prompt': tone}
        for layer in range(1, cfg['model']['layers'] + 1):
            yield {'name': f'{tone}_layer_{layer}', 'tone': tone, 'layers': [layer], 'prompt': None}


def test_conditions(cfg, selection):
    result = [{'name': 'baseline', 'tone': None, 'layers': [], 'prompt': None}]
    for tone in cfg['directions']['emotions']:
        window = selection['windows'][tone]
        for name, layers in [('prompt', []), ('probe_best', [selection['probe_layer']]),
                             ('steer_best', [window['best_single']]), ('window', window['layers']),
                             ('window_sum_budget', window['layers']), ('random_direction_window', window['layers'])]:
            result.append({'name': tone + '_' + name, 'tone': tone, 'layers': layers,
                           'prompt': tone if name == 'prompt' else None,
                           'random_direction': name == 'random_direction_window',
                           'budget': 'sum_relative_l2' if name == 'window_sum_budget' else 'per_layer'})
        for k, layers in enumerate(random_bands(cfg['model_key'], tone, cfg['model']['layers'])):
            result.append({'name': f'{tone}_random_band_{k}', 'tone': tone, 'layers': layers, 'prompt': None})
    assert len(result) == 23
    return result


def generate(cfg, base, stamp, mode, shard, shards):
    if not 0 <= shard < shards:
        raise ValueError('Invalid shard')
    checked(base, 'fit', stamp, ['directions.npz'])
    checked(base, 'audit', stamp, ['manifest.json'])
    seeds = [cfg['selection']['seed']] if mode == 'scan' else cfg['generation']['seeds']
    conditions = list(scan_conditions(cfg))
    selection_sha = None
    if mode == 'test':
        checked(base, 'select', stamp, ['selection.json', 'plan.json'])
        conditions = load_config(base / 'select/plan.json')
        selection_sha = digest(base / 'select/selection.json')
    rows = [r for r in manifest(cfg) if r['partition'] == ('validation' if mode == 'scan' else 'test')][shard::shards]
    verify_images(rows, load_config(base / 'audit/manifest.json'))
    with stage(base, f'{mode}/shard_{shard}', stamp) as out:
        import torch, transformers
        write_json(out / 'runtime.json', {'torch': torch.__version__, 'transformers': transformers.__version__,
                                         'gpu': torch.cuda.get_device_name(), 'shard': shard, 'shards': shards})
        model = adapter(cfg)
        vectors = np.load(base / 'fit/directions.npz')
        rng = np.random.default_rng(cfg['seed'])
        random = rng.standard_normal((cfg['model']['layers'], cfg['model']['hidden_size'])).astype('float32')
        random /= np.linalg.norm(random, axis=-1, keepdims=True)
        for row in rows:
            for seed in seeds:
                for c in conditions:
                    v = random if c.get('random_direction') else (vectors[c['tone']] if c['layers'] else None)
                    result = model.generate(row, v, c['layers'], c['prompt'], budget=c.get('budget', 'per_layer'), seed=seed)
                    append_record(out / 'generations.jsonl', dict(result, image_id=row['image_id'], partition=row['partition'],
                                  seed=seed, condition=c['name'], inputs=stamp, selection_sha256=selection_sha))
                    logging.info('%s %s seed%s', row['image_id'], c['name'], seed)


def score(cfg, base, stamp, inputs, tag):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    with stage(base, 'scores/' + tag, stamp) as out:
        tokenizer = AutoTokenizer.from_pretrained(cfg['evaluation']['emotion_path'], local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(cfg['evaluation']['emotion_path'], local_files_only=True, use_safetensors=True).eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        seen = set()
        write_json(out / 'input_hashes.json', {Path(p).as_posix(): digest(p) for p in inputs})
        for path in inputs:
            for row in read_jsonl(path):
                key = (row['image_id'], row['condition'], row['seed'])
                if key in seen or row['inputs'] != stamp:
                    raise ValueError('Duplicate record or changed configuration')
                seen.add(key)
                batch = tokenizer(row['text'], return_tensors='pt', truncation=True, max_length=512).to(device)
                with torch.inference_mode():
                    probs = model(**batch).logits.float().softmax(-1)[0].cpu().tolist()
                scores = {model.config.id2label[i].lower(): float(p) for i, p in enumerate(probs)}
                if not all(t in scores for t in cfg['directions']['emotions']):
                    raise ValueError('Wrong classifier labels')
                append_record(out / 'evaluations.jsonl', dict(row, emotion_scores=scores, degenerate=degeneration(row['text'])))


def matrix(inputs, images, conditions, seeds, stamp, partition, selection_sha=None):
    table = {}
    for path in inputs:
        for row in read_jsonl(path):
            key = (row['image_id'], row['condition'], row['seed'])
            if key in table or row['inputs'] != stamp or row['partition'] != partition:
                raise ValueError('Duplicate, foreign configuration, or wrong partition')
            if selection_sha is not None and row.get('selection_sha256') != selection_sha:
                raise ValueError('Wrong frozen selection')
            scores = row['emotion_scores']
            if not all(np.isfinite(p) and 0 <= p <= 1 for p in scores.values()) or abs(sum(scores.values()) - 1) > 1e-5:
                raise ValueError('Malformed classifier probabilities')
            table[key] = row
    expected = {(i, c, s) for i in images for c in conditions for s in seeds}
    if set(table) != expected:
        raise ValueError(f'Incomplete grid: missing={len(expected - set(table))}, extra={len(set(table) - expected)}')
    return table


def select(cfg, base, stamp, inputs):
    checked(base, 'fit', stamp, ['directions.npz', 'probe.json'])
    images = [r['image_id'] for r in manifest(cfg) if r['partition'] == 'validation']
    seed = cfg['selection']['seed']
    table = matrix(inputs, images, [c['name'] for c in scan_conditions(cfg)], [seed], stamp, 'validation')
    scores = load_config(base / 'fit/probe.json')
    windows, profile = {}, []
    for tone in cfg['directions']['emotions']:
        baseline = np.array([table[i, 'baseline', seed]['emotion_scores'][tone] for i in images])
        gains = []
        for l in range(1, cfg['model']['layers'] + 1):
            gain = np.mean([table[i, f'{tone}_layer_{l}', seed]['emotion_scores'][tone] for i in images]) - baseline.mean()
            gains.append(float(gain))
            profile.append({'tone': tone, 'layer': l, 'gain': float(gain)})
        # Final manuscript rule: no degeneration threshold or hidden fallback.
        choice = choose_window(gains, cfg['selection']['width'])
        if choice['status'] != 'VALIDATION_SELECTED':
            raise ValueError('No finite selected window')
        windows[tone] = dict(choice, best_single=int(np.argmax(gains)) + 1)
    selected = {'windows': windows, 'probe_layer': max(scores, key=lambda r: (r['validation_accuracy'], -r['layer']))['layer']}
    with stage(base, 'select', stamp) as out:
        write_json(out / 'selection.json', selected)
        write_json(out / 'profile.json', profile)
        write_json(out / 'plan.json', test_conditions(cfg, selected))
        write_json(out / 'input_hashes.json', {Path(p).as_posix(): digest(p) for p in inputs})


def image_ci(delta, confidence=.95, seed=42, repeats=10000):
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 1 or len(delta) < 2 or not np.isfinite(delta).all():
        raise ValueError('Need finite image-level paired differences')
    rng = np.random.default_rng(seed)
    # Batches bound memory for 2,000-image test sets.
    samples = np.concatenate([delta[rng.integers(0, len(delta), (min(256, repeats - k), len(delta)))].mean(1)
                              for k in range(0, repeats, 256)])
    lo, hi = np.quantile(samples, [(1 - confidence) / 2, (1 + confidence) / 2])
    return {'delta': float(delta.mean()), 'low': float(lo), 'high': float(hi), 'confidence': confidence, 'n_images': len(delta)}


def analyze(cfg, base, stamp, inputs):
    checked(base, 'select', stamp, ['selection.json', 'plan.json'])
    images = [r['image_id'] for r in manifest(cfg) if r['partition'] == 'test']
    seeds = cfg['generation']['seeds']
    names = [c['name'] for c in load_config(base / 'select/plan.json')]
    table = matrix(inputs, images, names, seeds, stamp, 'test', digest(base / 'select/selection.json'))
    means, contrasts = [], []
    for tone in cfg['directions']['emotions']:
        def vals(name, ss):
            return np.array([np.mean([table[i, name, s]['emotion_scores'][tone] for s in ss]) for i in images])
        for ss in [[s] for s in seeds] + [seeds]:
            label = str(ss[0]) if len(ss) == 1 else 'all'
            for name in names:
                means.append({'tone': tone, 'seed': label, 'condition': name, 'probability': float(vals(name, ss).mean()), 'n_images': len(images)})
            aw = vals(tone + '_window', ss)
            comparisons = [(name, vals(name, ss)) for name in ['baseline'] + [tone + suffix for suffix in ['_prompt', '_probe_best', '_steer_best', '_random_direction_window']]]
            comparisons.append(('mean_five_random_bands', np.mean([vals(f'{tone}_random_band_{k}', ss) for k in range(5)], axis=0)))
            for name, control in comparisons:
                confidence = 1 - .05 / 3 if name == 'mean_five_random_bands' and tone == 'joy' and label == 'all' else .95
                contrasts.append(dict(tone=tone, seed=label, comparison='AW_minus_' + name,
                                      **image_ci(aw - control, confidence, repeats=cfg['evaluation']['bootstrap'])))
            contrasts.append(dict(tone=tone, seed=label, comparison='matched_window_minus_best_single',
                                  **image_ci(vals(tone + '_window_sum_budget', ss) - vals(tone + '_steer_best', ss), repeats=cfg['evaluation']['bootstrap'])))
    with stage(base, 'analysis', stamp) as out:
        for name, rows in [('means.csv', means), ('contrasts.csv', contrasts)]:
            with (out / name).open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['audit', 'extract', 'fit', 'scan', 'score', 'select', 'test', 'analyze'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--inputs', nargs='+')
    parser.add_argument('--tag', default='validation')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    cfg, base, stamp = context(args.config)
    if args.stage == 'audit':
        rows = audit_manifest(cfg)
        with stage(base, 'audit', stamp) as out:
            write_json(out / 'manifest.json', rows)
    elif args.stage == 'extract':
        development(cfg, base, stamp)
    elif args.stage == 'fit':
        fit(cfg, base, stamp)
    elif args.stage in ['scan', 'test']:
        generate(cfg, base, stamp, args.stage, args.shard, args.shards)
    else:
        if not args.inputs:
            parser.error('--inputs is required for score, select and analyze')
        if args.stage == 'score':
            score(cfg, base, stamp, args.inputs, args.tag)
        elif args.stage == 'select':
            select(cfg, base, stamp, args.inputs)
        else:
            analyze(cfg, base, stamp, args.inputs)


if __name__ == '__main__':
    main()
