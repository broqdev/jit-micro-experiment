"""Single-device RGB flow matching: training and correctness preflight."""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from dotenv import load_dotenv
import numpy as np


# Configuration and environment

PROJECT = Path(__file__).resolve().parent
COMPILER_CACHE_DEFAULTS = {
    'TORCHINDUCTOR_CACHE_DIR': '.cache/torch/inductor-max-autotune',
    'TRITON_CACHE_DIR': '.cache/torch/triton',
    'TRITON_HOME': '.cache/torch/triton',
}
COMPILER_ENV = (*COMPILER_CACHE_DEFAULTS, 'TORCHINDUCTOR_COMPILE_THREADS')
HF_CACHE_DEFAULTS = {
    'HF_HOME': '.cache/huggingface',
    'HF_HUB_CACHE': '.cache/huggingface/hub',
    'HF_XET_CACHE': '.cache/huggingface/xet',
}


def project_path(value):
    """Interpret relative config paths from the project directory."""
    return PROJECT / Path(value).expanduser()


def bootstrap(env_file=None, *, training=True):
    path = Path(env_file) if env_file is not None else PROJECT / '.env'
    if path.is_file():
        load_dotenv(path, override=True)
    elif env_file is not None:
        raise FileNotFoundError(f'Environment file does not exist: {path}')

    defaults = {**HF_CACHE_DEFAULTS, **COMPILER_CACHE_DEFAULTS} if training else HF_CACHE_DEFAULTS
    for name, default in defaults.items():
        os.environ[name] = str(project_path(os.environ.get(name) or default).resolve())
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    if not training:
        return
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
    for name in COMPILER_CACHE_DEFAULTS:
        cache = Path(os.environ[name])
        root = Path(cache.anchor)
        if cache == root or (os.name == 'nt' and cache.parent == root and cache.name.lower() == 'tc'):
            raise ValueError(f'{name} must name a compiler cache directory, not a shared root')
    if os.name == 'nt' and os.environ['TORCHINDUCTOR_COMPILE_THREADS'] != '1':
        raise ValueError('Set TORCHINDUCTOR_COMPILE_THREADS=1 on Windows')


def bootstrap_compute(parser, env_file):
    bootstrap(env_file)
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('Use one process and one device')
    # Python reads the UTF-8 setting at startup, before Inductor reads templates.
    if os.name == 'nt' and os.environ.get('PYTHONUTF8', '1') == '1' and not sys.flags.utf8_mode:
        raise SystemExit(subprocess.call([sys.executable, '-X', 'utf8', *sys.argv]))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def source_manifest():
    paths = list(PROJECT.glob('*.py'))
    paths.append(PROJECT / 'pyproject.toml')
    return {str(p.relative_to(PROJECT)).replace('\\', '/'): sha(p) for p in sorted(paths)}


def compute_provenance(cfg):
    return dict(sources=source_manifest(), pid=os.getpid(), device=cfg['runtime']['device'],
                environment={name: os.environ[name] for name in COMPILER_ENV})


def validate(cfg):
    model, tr, data = cfg['model'], cfg['training'], cfg['dataset']
    if type(model['depth']) is not int or model['depth'] not in (2, 4, 8, 12, 16):
        raise ValueError('Use a depth of 2, 4, 8, 12 or 16')
    if model['resolution'] % model['patch'] or model['head_dim'] % 4:
        raise ValueError('Resolution must divide into patches; head_dim must divide by four')
    for key in ('resolution', 'patch', 'width', 'heads', 'head_dim', 'attribute_dim', 'head_width'):
        if model[key] <= 0:
            raise ValueError(f'Invalid model.{key}')
    if tr['batch_size'] <= 0 or tr['updates'] <= 0:
        raise ValueError('Batch and updates must be positive')
    if tr['warmup_updates'] < 0:
        raise ValueError('Warmup updates must be nonnegative')
    if not 0 < tr['final_learning_rate'] <= tr['learning_rate']:
        raise ValueError('Learning rates must satisfy 0 < final <= peak')
    if not 0 <= tr['attribute_dropout'] < 1 or tr['sigma_floor'] <= 0:
        raise ValueError('Invalid flow objective configuration')
    if not all(math.isfinite(float(tr[k])) and tr[k] > 0
               for k in ('ema_half_life_updates', 'gradient_clip')):
        raise ValueError('EMA half-life and gradient clipping must be positive and finite')
    ev = cfg['evaluation']
    if not 0 < ev['tuning_count'] < data['validation_count']:
        raise ValueError('Development and confirmation sets must both be nonempty')
    if ev['periodic_count'] != ev['tuning_count'] or ev['final_count'] != data['validation_count']:
        raise ValueError('Evaluation counts do not match the declared partition')
    if any(not 0 <= t < 1 for t in ev['times']):
        raise ValueError('Evaluation times must lie in [0, 1)')
    if len(ev['score_scales']) != len(ev['times']) or any(v <= 0 for v in ev['score_scales']):
        raise ValueError('One positive score normalization is required per time')
    if cfg['runtime']['device'] not in ('cuda', 'cpu'):
        raise ValueError('runtime.device must be cuda or cpu')
    if type(cfg['runtime'].get('training_tf32', False)) is not bool:
        raise ValueError('runtime.training_tf32 must be a boolean')


def learning_rate(cfg, step):
    tr = cfg['training']
    ratio = step / max(1, tr['updates'] - 1)
    value = tr['final_learning_rate'] + .5 * (tr['learning_rate'] - tr['final_learning_rate']) * (1 + math.cos(math.pi * ratio))
    return value * min((step + 1) / max(1, tr['warmup_updates']), 1.)


# Local event logging

class EventLog:
    """Write structured progress to the run directory and stdout."""
    def __init__(self, file):
        self.file = file

    def emit(self, event, step, **values):
        record = dict(event=event, global_step=step, time=time.time(), **values)
        line = json.dumps(record, allow_nan=False)
        self.file.write(line + '\n')
        print(line, flush=True)


@contextmanager
def run_log(cfg, **provenance):
    output = project_path(cfg['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    try:
        with (output / 'metrics.jsonl').open('a', encoding='utf-8', buffering=1) as file:
            write(output / 'resolved-config.json', cfg)
            write(output / 'provenance.json', dict(command=sys.argv, **provenance))
            yield output, EventLog(file)
    except BaseException as error:
        write(output / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
        raise


# Training loss and lazy runtime imports

def weighted_error(prediction, target, time, mask, count, sigma_floor):
    per_image = (prediction - target).square().mean((1, 2, 3))
    # Padded examples carry zero weight; normalize by the valid image count.
    factor = 1 / count.clamp_min(1)
    loss = (per_image / (1 - time).clamp_min(sigma_floor).square() * mask).sum() * factor
    mse = (per_image * mask).sum() * factor
    return torch.stack((loss, mse))


def load_runtime():
    """Load PyTorch and the model after the dotenv bootstrap.

    Configuration and logging helpers can be imported by prepare_data.py
    without importing torch or initializing the training runtime.
    """
    global torch, nn, Denoiser, Objective, patchify, unpatchify
    if 'torch' in globals():
        return
    import torch
    from torch import nn
    from model import Denoiser, patchify, unpatchify

    class Objective(nn.Module):
        def __init__(self, model, floor, fused_attention=False):
            super().__init__()
            self.model, self.floor = model, floor
            self.fused_attention = fused_attention

        def forward(self, xt, time, attributes, target, mask, count):
            prediction = self.model(xt, time, attributes, fused_attention=self.fused_attention)
            return weighted_error(prediction, target, time, mask, count, self.floor)

# Data and optimizer

def compile_static(function, cfg):
    return torch.compile(function, fullgraph=True, dynamic=False, options={
        'max_autotune': cfg['runtime']['compile_mode'] == 'max-autotune',
        'triton.cudagraphs': False, 'layout_optimization': False})


def batch_plan(order, step_in_epoch, batch_size):
    valid = order[step_in_epoch * batch_size:(step_in_epoch + 1) * batch_size]
    count = len(valid)
    if not count:
        raise ValueError('Batch is past the end of the epoch')
    padded = valid.repeat(math.ceil(batch_size / count))[:batch_size]
    mask = torch.arange(batch_size) < count
    return padded, mask, valid


def prepare(clean, attributes, ids, noise, time_noise, dropout_noise, mask, count,
            patch, mean, std, dropout):
    target = clean[ids].float().div(255)
    time = (time_noise * std + mean).sigmoid()
    attrs = attributes[ids] * (dropout_noise >= dropout)[:, None]
    xt = noise * (1 - time[:, None, None]) + patchify(2 * target - 1, patch) * time[:, None, None]
    return xt, time, attrs, target, mask, count


class Data:
    def __init__(self, cfg, device, compiled=False, *, training=True):
        self.cfg, self.device = cfg, device
        self.batch_size = cfg['training']['batch_size']
        self.steps_per_epoch = math.ceil(cfg['dataset']['train_count'] / self.batch_size)
        self.epoch, self.order = -1, None
        root = project_path(cfg['dataset']['cache'])
        resolution, attrs = cfg['model']['resolution'], cfg['model']['attribute_dim']
        self.pack = {}
        for split in ('train', 'validation'):
            images = np.load(root / f'{split}.npy', mmap_mode='r')
            with np.load(root / f'{split}-metadata.npz') as metadata:
                ids, attributes = metadata['ids'].copy(), metadata['attributes'].copy()
            count = cfg['dataset'][split + '_count']
            if images.shape != (count, 3, resolution, resolution) or images.dtype != np.uint8:
                raise ValueError(f'Invalid {split} image tensor')
            if len(ids) != count or len(set(ids.tolist())) != count or attributes.shape != (count, attrs):
                raise ValueError(f'Invalid {split} metadata')
            if not np.isin(attributes, [0, 1]).all():
                raise ValueError('Attributes must be binary')
            self.pack[split] = dict(images=images, image_ids=ids.tolist(),
                                    attributes=torch.from_numpy(attributes).float())
        if set(self.pack['train']['image_ids']) & set(self.pack['validation']['image_ids']):
            raise ValueError('Training and validation IDs overlap')
        if training:
            self.clean = torch.from_numpy(np.array(self.pack['train']['images'], copy=True)).to(device)
            self.attributes = self.pack['train']['attributes'].to(device)
            self.prepare = compile_static(prepare, cfg) if compiled and device.type == 'cuda' else prepare

    def epoch_order(self, epoch):
        seed = self.cfg['training']['seed'] + epoch * 7919
        return torch.randperm(len(self.clean), generator=torch.Generator().manual_seed(seed))

    def batch(self, step):
        tr, model, device = self.cfg['training'], self.cfg['model'], self.device
        epoch, offset = divmod(step, self.steps_per_epoch)
        if epoch != self.epoch:
            self.order = self.epoch_order(epoch)
            self.epoch = epoch
        ids, mask, valid = batch_plan(self.order, offset, self.batch_size)
        rng = torch.Generator(device=device).manual_seed(tr['seed'] + step * 1009)
        noise = torch.randn((self.batch_size, (model['resolution'] // model['patch'])**2,
                             model['patch']**2 * 3), device=device, generator=rng)
        times = torch.randn(self.batch_size, device=device, generator=rng)
        dropout = torch.rand(self.batch_size, device=device, generator=rng)
        batch = self.prepare(self.clean, self.attributes, ids.to(device),
            noise, times, dropout, mask.to(device=device, dtype=torch.float32),
            torch.tensor(float(len(valid)), device=device), model['patch'],
            tr['time_logit_mean'], tr['time_logit_std'], tr['attribute_dropout'])
        return batch, valid

    def evaluation_batch(self, split, start, end):
        pack = self.pack[split]
        # Match the saved evaluation protocol: normalize uint8 on CPU before
        # transfer. GPU division can change BF16 rounding thresholds downstream.
        target = torch.from_numpy(np.array(pack['images'][start:end], copy=True)).float().div_(255).to(self.device)
        return target, pack['attributes'][start:end].to(self.device)


@contextmanager
def training_matmul_precision(enable_tf32):
    if not enable_tf32:
        yield
        return
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


class Engine:
    def __init__(self, cfg, device, compiled=None):
        self.cfg, self.device = cfg, device
        torch.manual_seed(cfg['training']['seed'])
        self.model = Denoiser(cfg['model']).to(device)
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.parameters = list(self.model.parameters())
        tr = cfg['training']
        cuda = device.type == 'cuda'
        self.training_tf32 = cuda and cfg['runtime'].get('training_tf32', False)
        self.lr = torch.tensor(tr['learning_rate'], device=device)
        self.optimizer = torch.optim.AdamW(self.parameters, lr=self.lr, betas=tuple(tr['betas']),
            weight_decay=tr['weight_decay'], fused=cuda, capturable=cuda)
        for parameter in self.parameters:
            parameter.grad = torch.zeros_like(parameter)
            self.optimizer.state[parameter] = dict(step=torch.zeros((), device=device),
                exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))
        self.ema_weight = 1 - 2 ** (-1 / tr['ema_half_life_updates'])
        compiled = cfg['runtime']['compile'] if compiled is None else compiled
        if compiled and not cuda:
            raise ValueError('Use runtime.compile=false for CPU runs')
        self.eager = Objective(self.model, tr['sigma_floor'], cuda and cfg['runtime'].get('fused_attention', False))
        self.objective = compile_static(self.eager, cfg) if compiled else self.eager
        self.replay = bool(cfg['runtime']['cuda_graph'] and compiled)
        self.graph = None

    def compute(self, batch):
        self.optimizer.zero_grad(set_to_none=False)
        with training_matmul_precision(self.training_tf32):
            values = self.objective(*batch)
            values[0].backward()
        norm = nn.utils.clip_grad_norm_(self.parameters, self.cfg['training']['gradient_clip'], foreach=True)
        self.optimizer.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(self.ema.parameters()), self.parameters, self.ema_weight)
        return torch.cat((values.detach(), norm.detach().reshape(1)))

    def capture(self, batch):
        state = self.snapshot()
        self.static_batch = tuple(x.clone() for x in batch)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.compute(self.static_batch)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_result = self.compute(self.static_batch)
        self.restore(state)

    def step(self, batch):
        if self.replay:
            if self.graph is None:
                self.capture(batch)
            for target, source in zip(self.static_batch, batch):
                target.copy_(source)
            self.graph.replay()
            values = self.static_result
        else:
            values = self.compute(batch)
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError(values.cpu().tolist())
        return values.cpu().tolist()

    def snapshot(self):
        def cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {k: cpu(v) for k, v in value.items()}
            if isinstance(value, list):
                return [cpu(v) for v in value]
            if isinstance(value, tuple):
                return tuple(cpu(v) for v in value)
            return value
        return dict(model=cpu(self.model.state_dict()), ema=cpu(self.ema.state_dict()),
                    optimizer=cpu(self.optimizer.state_dict()), lr=self.lr.detach().cpu().clone())

    def restore(self, state):
        self.model.load_state_dict(state['model'], strict=True)
        self.ema.load_state_dict(state['ema'], strict=True)
        self.lr.copy_(state['lr'])
        if self.graph is None:
            self.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
            for group in self.optimizer.param_groups:
                group['lr'] = self.lr
        else:
            for group, saved_group in zip(self.optimizer.param_groups, state['optimizer']['param_groups']):
                for parameter, index in zip(group['params'], saved_group['params']):
                    for key, value in state['optimizer']['state'][index].items():
                        self.optimizer.state[parameter][key].copy_(value)

    def finite(self):
        states = [self.model.state_dict(), self.ema.state_dict(), *self.optimizer.state.values()]
        return all(bool(torch.isfinite(value).all()) for state in states for value in state.values())

    def digest(self):
        digest = hashlib.sha256()
        for model in (self.model, self.ema):
            for tensor in model.state_dict().values():
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


def vector(model):
    return torch.cat([p.detach().float().reshape(-1) for p in model.parameters()])


def relative(actual, reference):
    return float((actual - reference).double().square().mean().sqrt() /
                 reference.double().square().mean().sqrt().clamp_min(1e-12))


# Training, checkpoints and correctness preflight

def signature(cfg, device):
    training = {k: v for k, v in cfg['training'].items() if k not in ('checkpoint_every', 'evaluate_every', 'log_every')}
    return dict(model=cfg['model'], training=training, dataset=cfg['dataset'],
                runtime=cfg['runtime'], sources=source_manifest(),
                torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                caches={k: os.environ[k] for k in ('TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR',
                                                  'TORCHINDUCTOR_COMPILE_THREADS')})


def checkpoint(engine, cfg, device, output, step, visits):
    path = output / f'checkpoint-{step:06d}.pt'
    temp = path.with_suffix('.pt.tmp')
    torch.save(dict(**engine.snapshot(), config=cfg, signature=signature(cfg, device), step=step, visits=visits), temp)
    temp.replace(path)
    receipt = dict(path=str(path), step=step, sha256=sha(path))
    write(output / 'latest-checkpoint.json', receipt)
    return receipt


def validate_resume(state, cfg, device):
    if state.get('signature') != signature(cfg, device):
        raise ValueError('Resume requires identical model, recipe/horizon, data, source, runtime, and caches')
    if not 0 <= state['step'] < cfg['training']['updates']:
        raise ValueError('Resume checkpoint must precede the configured end')


def preflight(cfg, device, log):
    torch.manual_seed(cfg['training']['seed'])
    cpu = Denoiser(cfg['model'])
    m = cfg['model']
    rgb = torch.rand(2, 3, m['resolution'], m['resolution'])
    tokens = patchify(rgb, m['patch'])
    torch.testing.assert_close(unpatchify(tokens, m['resolution'], m['patch']), rgb, rtol=0, atol=0)
    with torch.no_grad():
        cpu.head[-1].weight.normal_(0, .01)
    times, attrs = torch.tensor([.2, .8]), torch.rand(2, m['attribute_dim'])
    prediction = cpu(tokens, times, attrs)
    mask = torch.tensor([1., 0.])
    loss = weighted_error(prediction, rgb, times, mask, torch.tensor(1.), cfg['training']['sigma_floor'])
    reference = weighted_error(prediction[:1], rgb[:1], times[:1], mask[:1], torch.tensor(1.), cfg['training']['sigma_floor'])
    # Slicing changes the CPU reduction shape/strides; permit FP32 roundoff.
    torch.testing.assert_close(loss, reference, rtol=2e-6, atol=2e-8)
    loss[0].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in cpu.parameters())
    assert len(cpu.blocks) == m['depth']
    count = sum(p.numel() for p in cpu.parameters())
    log.emit('cpu_correctness_passed', 0, parameters=count)
    data = Data(cfg, device, compiled=False)
    engine = Engine(cfg, device, compiled=False)
    # Compare production kernels against the eager, unfused training objective.
    engine.eager.fused_attention = False
    engine.training_tf32 = False
    for step in range(3):
        engine.step(data.batch(step)[0])
    assert engine.finite()
    gates = []
    if cfg['runtime']['compile']:
        fixture = engine.snapshot()
        candidate = Engine(cfg, device, compiled=True)
        compiled_data = copy.copy(data)
        compiled_data.prepare = compile_static(prepare, cfg)
        for step in (4, data.steps_per_epoch - 1):
            batch, ids = data.batch(step)
            actual_batch, actual_ids = compiled_data.batch(step)
            assert torch.equal(ids, actual_ids)
            error = max(float((a - b).abs().max()) for a, b in zip(batch, actual_batch))
            assert error <= 2e-6
            engine.restore(fixture)
            candidate.restore(fixture)
            before = vector(engine.model).clone()
            reference_values = engine.step(batch)
            actual_values = candidate.step(actual_batch)
            row = dict(valid_images=len(ids), input_max_abs=error,
                loss_relative=abs(actual_values[0] / reference_values[0] - 1),
                gradient_relative=relative(torch.cat([p.grad.detach().reshape(-1) for p in candidate.parameters]),
                                           torch.cat([p.grad.detach().reshape(-1) for p in engine.parameters])),
                update_relative=relative(vector(candidate.model) - before, vector(engine.model) - before),
                ema_relative=relative(vector(candidate.ema), vector(engine.ema)))
            for key in ('exp_avg', 'exp_avg_sq', 'step'):
                row[key + '_relative'] = relative(
                    torch.cat([s[key].reshape(-1) for s in candidate.optimizer.state.values()]),
                    torch.cat([s[key].reshape(-1) for s in engine.optimizer.state.values()]))
            assert row['loss_relative'] <= .01 and all(value <= .03 for key, value in row.items() if key.endswith('_relative')), row
            assert candidate.finite() and all(float(s['step']) == 4 for s in candidate.optimizer.state.values())
            gates.append(row)
            log.emit('compiled_parity', step, **row)
    return dict(passed=True, parameters=count, cpu_forward_backward_passed=True, compiled_gates=gates,
                config=cfg, signature=signature(cfg, device))


def train(cfg, device, output, log, resume=None):
    from eval import evaluate

    tr = cfg['training']
    engine = Engine(cfg, device)
    data = Data(cfg, device, compiled=cfg['runtime']['compile'])
    start, visits, epoch_ids, startup_epochs = 0, 0, [], []
    if resume:
        state = torch.load(resume, map_location='cpu', weights_only=True)
        validate_resume(state, cfg, device)
        engine.restore(state)
        start, visits = state['step'], state['visits']
        epoch, offset = divmod(start, data.steps_per_epoch)
        epoch_ids = data.epoch_order(epoch)[:offset * tr['batch_size']].tolist()
    started = time.monotonic()
    last_evaluation, last_checkpoint = None, None
    for step in range(start, tr['updates']):
        engine.lr.fill_(learning_rate(cfg, step))
        batch, ids = data.batch(step)
        values = engine.step(batch)
        visits += len(ids)
        epoch_ids.extend(ids.tolist())
        done = step + 1
        if done == start + 1 or done % tr['log_every'] == 0:
            row = dict(updates=tr['updates'], completed_epochs=done // data.steps_per_epoch,
                loss=values[0], rgb_mse=values[1], grad_norm=values[2], lr=float(engine.lr), visits=visits,
                finite=True, elapsed_seconds=time.monotonic() - started)
            write(output / 'progress.json', dict(step=done, **row))
            log.emit('training_progress', done, **row)
        if done % data.steps_per_epoch == 0:
            if not engine.finite():
                raise FloatingPointError('Nonfinite model, optimizer, or EMA at epoch boundary')
            if len(epoch_ids) != len(set(epoch_ids)) or len(epoch_ids) != cfg['dataset']['train_count']:
                raise RuntimeError('Epoch failed exact image coverage check')
            epoch_ids = []
            row = dict(epoch=done // data.steps_per_epoch, images=cfg['dataset']['train_count'],
                       finite=True, loss=values[0], rgb_mse=values[1], visits=visits)
            epoch_record = dict(step=done, **row)
            with (output / 'epochs.jsonl').open('a', encoding='utf-8') as file:
                file.write(json.dumps(epoch_record) + '\n')
            log.emit('epoch_complete', done, **row)
            if len(startup_epochs) < 3:
                startup_epochs.append(epoch_record)
                if len(startup_epochs) == 3:
                    receipt = checkpoint(engine, cfg, device, output, done, visits)
                    write(output / 'startup-verification.json', dict(passed=True, epochs=startup_epochs,
                        model_ema_digest=engine.digest(), checkpoint=receipt,
                        max_allocated_gib=torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0))
        if done % tr['checkpoint_every'] == 0 or done == tr['updates']:
            last_checkpoint = checkpoint(engine, cfg, device, output, done, visits)
        if done % tr['evaluate_every'] == 0 or done == tr['updates']:
            last_evaluation = evaluate({'raw': engine.model, 'ema': engine.ema}, data, cfg,
                                       output / f'evaluation-{done:06d}', done,
                                       count=cfg['evaluation']['periodic_count'], score_weights='ema')
            log.emit('development_evaluation', done, score=last_evaluation['score'])
    if not (engine.finite() and all(float(s['step']) == tr['updates'] for s in engine.optimizer.state.values())):
        raise FloatingPointError('Invalid final optimizer/model state')
    return dict(passed=True, learning_rate=tr['learning_rate'], updates=tr['updates'], visits=visits,
        score=last_evaluation['score'], checkpoint=last_checkpoint, model_ema_digest=engine.digest(),
        elapsed_seconds=time.monotonic() - started, completed_epochs=tr['updates'] // data.steps_per_epoch)


# Shared runtime setup and command-line entry point


def setup_runtime(cfg):
    """Configure the shared training/evaluation runtime after dotenv bootstrap."""
    load_runtime()
    device = torch.device('cuda', 0) if cfg['runtime']['device'] == 'cuda' else torch.device('cpu')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('runtime.device=cuda requires an available CUDA device')
        torch.cuda.set_device(device)
    torch.set_num_threads(cfg['runtime']['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = cfg['runtime']['tf32']
    torch.backends.cudnn.allow_tf32 = cfg['runtime']['tf32']
    torch.backends.cudnn.benchmark = True
    return device

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'preflight'))
    parser.add_argument('--config', type=Path, default=PROJECT / 'config.json')
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--output', type=Path, help='Fresh output directory, relative to the project')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--updates', type=int)
    parser.add_argument('--depth', type=int, choices=(2, 4, 8, 12, 16), help='Override model.depth')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    bootstrap_compute(parser, args.env_file)
    cfg = read(args.config)
    if args.depth is not None and args.depth != cfg['model']['depth']:
        cfg['model']['depth'] = args.depth
        if args.output is None:
            cfg['output_dir'] += f'-depth{args.depth}'
    if args.output:
        cfg['output_dir'] = args.output.as_posix()
    if args.lr is not None:
        cfg['training'].update(learning_rate=args.lr, final_learning_rate=args.lr * .1)
    if args.updates is not None:
        cfg['training']['updates'] = args.updates
    validate(cfg)
    with run_log(cfg, **compute_provenance(cfg)) as (output, log):
        device = setup_runtime(cfg)
        log.emit('compute_start', 0, batch_size=cfg['training']['batch_size'])
        if args.phase == 'preflight':
            result = preflight(cfg, device, log)
        else:
            result = train(cfg, device, output, log, args.resume)
        write(output / 'result.json', result)
        log.emit('run_complete', result.get('updates', 0), passed=True)


if __name__ == '__main__':
    main()
