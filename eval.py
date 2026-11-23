"""Evaluate one checkpoint weight set and generate train/validation sample galleries."""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

from train import (PROJECT, Data, bootstrap_compute, compute_provenance, project_path,
                   read, run_log, setup_runtime, validate, write)

CHECKPOINT_KEYS = {'ema': 'ema', 'raw': 'model'}
DEFAULT_SAMPLE_STEPS = {'euler': 40, 'tweedie': 12}

# Evaluation and sampling

def sampling_options(cfg):
    ev = cfg['evaluation']
    sampler = ev.get('sampler', 'euler')
    if sampler not in DEFAULT_SAMPLE_STEPS:
        raise ValueError('evaluation.sampler must be euler or tweedie')
    steps = ev.get('sample_steps', DEFAULT_SAMPLE_STEPS[sampler])
    if type(steps) is not int or steps < 1:
        raise ValueError('evaluation.sample_steps must be a positive integer')
    scales = dict(cfg_scale=ev.get('cfg_scale', 1.0), pag_scale=ev.get('pag_scale', 0.0))
    for name, value in scales.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'evaluation.{name} must be a finite nonnegative number')
    depth = cfg['model']['depth']
    block = ev.get('pag_block', depth - 1)
    if type(block) is not int or not 0 <= block < depth:
        raise ValueError(f'evaluation.pag_block must be an integer between 0 and {depth - 1}')
    return dict(sampler=sampler, sample_steps=steps, pag_block=block, **scales)


def guided_prediction(model, tokens, times, attributes, *, cfg_scale, pag_scale, pag_block):
    if cfg_scale != 1:
        unconditional = model(tokens, times, attributes.new_zeros(attributes.shape))
        if cfg_scale == 0 and pag_scale == 0:
            return unconditional
    conditional = model(tokens, times, attributes)
    prediction = conditional
    if cfg_scale != 1:
        prediction = unconditional + cfg_scale * (conditional - unconditional)
    if pag_scale:
        perturbed = model(tokens, times, attributes, pag_block=pag_block)
        prediction = prediction + pag_scale * (conditional - perturbed)
    return prediction


def evaluate(models, data, cfg, output, step, *, count, score_weights):
    import torch
    from model import patchify

    with torch.no_grad():
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        ev, m = cfg['evaluation'], cfg['model']
        device = data.device
        records = {}
        for label, model in models.items():
            records[label] = {}
            for clean_time in ev['times']:
                errors = []
                for start in range(0, count, ev['batch_size']):
                    target, attrs = data.evaluation_batch('validation', start, min(start + ev['batch_size'], count))
                    noise = torch.randn((len(target), (m['resolution'] // m['patch'])**2, m['patch']**2 * 3),
                        device=device, generator=torch.Generator(device=device).manual_seed(ev['noise_seed'] + start))
                    times = torch.full((len(target),), clean_time, device=device)
                    prediction = model((1 - clean_time) * noise + clean_time * patchify(2 * target - 1, m['patch']), times, attrs)
                    error = (prediction - target).square().mean((1, 2, 3))
                    if not bool(torch.isfinite(error).all()):
                        raise FloatingPointError('Nonfinite evaluation')
                    errors.extend(error.cpu().tolist())
                records[label][str(clean_time)] = dict(rgb_mse=sum(errors) / count, per_image_rgb_mse=errors)
        score = sum(records[score_weights][str(t)]['rgb_mse'] / scale
                    for t, scale in zip(ev['times'], ev['score_scales'])) / len(ev['times'])
        result = dict(step=step, count=count, image_ids=data.pack['validation']['image_ids'][:count],
                      noise_seed=ev['noise_seed'], noise_batch=ev['batch_size'], metrics=records,
                      score=score, score_weights=score_weights)
        write(output / 'denoising.json', result)
        return result


def sample(model, attributes, cfg, device, *, sampler, sample_steps, **guidance):
    import torch
    from model import patchify, unpatchify

    with torch.no_grad():
        m, ev = cfg['model'], cfg['evaluation']
        token_noise = torch.randn((len(attributes), (m['resolution'] // m['patch'])**2, m['patch']**2 * 3),
            device=device, generator=torch.Generator(device=device).manual_seed(ev['sample_seed']))
        state = unpatchify(token_noise, m['resolution'], m['patch'])
        rng = torch.Generator(device=device).manual_seed(ev['renoise_seed']) if sampler == 'tweedie' else None
        for index in range(sample_steps):
            time = index / sample_steps
            times = torch.full((len(state),), time, device=device)
            prediction = guided_prediction(model, patchify(state, m['patch']), times, attributes, **guidance)
            if index == sample_steps - 1:
                # The final transition reaches t=1: both samplers return the
                # predicted clean image without another model evaluation.
                break
            clean = 2 * prediction - 1
            if sampler == 'tweedie':
                fresh = torch.randn(state.shape, device=device, generator=rng)
                next_time = (index + 1) / sample_steps
                state = next_time * clean + (1 - next_time) * fresh
            else:
                state = state + (clean - state) / ((1 - time) * sample_steps)
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError('Nonfinite generated sample')
        return prediction


def galleries(model, weights, data, cfg, output, sampling):
    import torch

    records = []
    count = cfg['evaluation']['sample_count']
    for split, pack in data.pack.items():
        attrs = pack['attributes'][:count].to(data.device)
        rgb = sample(model, attrs, cfg, data.device, **sampling).cpu()
        path = output / f'{split}-{weights}.png'
        array = rgb.clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
        Image.fromarray(np.concatenate(list(array), axis=1)).save(path)
        torch.save(dict(rgb=rgb, image_ids=pack['image_ids'][:count], attributes=attrs.cpu(),
                        weights=weights, **sampling), path.with_suffix('.pt'))
        records.append(str(path))
    return records


def confirm(cfg, device, output, log, checkpoint_path, sampling):
    import torch
    from model import Denoiser

    weights = cfg['evaluation']['weights']
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    saved_model = state['config']['model']
    saved_data = state['config']['dataset']
    eval_data = cfg['dataset']
    # Historical checkpoints can include extra model metadata.
    if any(saved_model.get(name) != value for name, value in cfg['model'].items()) or (
        dict(saved_data, cache=project_path(saved_data['cache']).resolve())
        != dict(eval_data, cache=project_path(eval_data['cache']).resolve())
    ):
        raise ValueError('Checkpoint architecture/data differs from the evaluation configuration')
    step = state['step']
    model = Denoiser(cfg['model'])
    model.load_state_dict(state[CHECKPOINT_KEYS[weights]])
    del state
    if not all(torch.isfinite(t).all().item() for t in model.state_dict().values()):
        raise FloatingPointError(f'Nonfinite {weights} checkpoint weights')
    model.to(device).eval()
    data = Data(cfg, device, training=False)
    evaluation_dir = output / 'evaluation'
    result = evaluate({weights: model}, data, cfg, evaluation_dir, step,
                      count=cfg['evaluation']['final_count'], score_weights=weights)
    paths = galleries(model, weights, data, cfg, evaluation_dir, sampling)
    begin = cfg['evaluation']['tuning_count']
    heldout = {weights: {}}
    for time_value, record in result['metrics'][weights].items():
        errors = record['per_image_rgb_mse'][begin:]
        heldout[weights][time_value] = sum(errors) / len(errors)
    log.emit('heldout_evaluation', step, images=result['count'] - begin, mse=heldout)
    return dict(passed=True, checkpoint=str(checkpoint_path), heldout_mse=heldout,
                heldout_count=result['count'] - begin, galleries=paths,
                weights=weights, visual_review_pending=True, **sampling)


# Command-line entry point


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'config.json')
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--output', type=Path, help='Fresh evaluation directory, relative to the project')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--weights', choices=CHECKPOINT_KEYS,
                        help='Weights for scoring and sampling; overrides evaluation.weights (default: ema)')
    parser.add_argument('--sampler', choices=DEFAULT_SAMPLE_STEPS,
                        help='Sampling method; overrides evaluation.sampler (default: euler)')
    parser.add_argument('--steps', type=int,
                        help='Override evaluation.sample_steps (defaults: Euler 40, Tweedie 12)')
    parser.add_argument('--cfg-scale', '--cfg', type=float,
                        help='Classifier-free guidance scale for sampling (1: conditional, 0: unconditional with PAG off)')
    parser.add_argument('--pag-scale', '--pag', type=float,
                        help='Perturbed-attention guidance scale for sampling (default: 0, disabled)')
    parser.add_argument('--pag-block', type=int,
                        help='Zero-based PAG block index; overrides evaluation.pag_block (fallback: last block)')
    args = parser.parse_args()
    bootstrap_compute(parser, args.env_file)
    cfg = read(args.config)
    if args.output:
        cfg['output_dir'] = args.output.as_posix()
    ev = cfg['evaluation']
    weights = args.weights or ev.get('weights', 'ema')
    if weights not in CHECKPOINT_KEYS:
        parser.error('evaluation.weights must be ema or raw')
    overrides = dict(sampler=args.sampler, sample_steps=args.steps, cfg_scale=args.cfg_scale,
                     pag_scale=args.pag_scale, pag_block=args.pag_block)
    ev.update({name: value for name, value in overrides.items() if value is not None})
    try:
        sampling = sampling_options(cfg)
    except ValueError as error:
        parser.error(str(error))
    ev.update(weights=weights, **sampling)
    validate(cfg)
    with run_log(cfg, checkpoint=str(args.checkpoint.resolve()), **compute_provenance(cfg)) as (output, log):
        device = setup_runtime(cfg)
        log.emit('compute_start', 0, batch_size=ev['batch_size'], weights=weights, **sampling)
        result = confirm(cfg, device, output, log, args.checkpoint, sampling)
        write(output / 'result.json', result)
        log.emit('run_complete', 0, passed=True)


if __name__ == '__main__':
    main()
