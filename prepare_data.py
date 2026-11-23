"""Download CelebA-HQ from Hugging Face and prepare RGB256 train/eval arrays."""
import argparse
import importlib.metadata
import os
from pathlib import Path, PurePosixPath
import re
import uuid
import zipfile

import numpy as np
from PIL import Image, ImageOps

from train import PROJECT, bootstrap, project_path, read, run_log, sha, write


ATTRIBUTE_NAMES = (
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
    'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young',
)
FILES = ('train.npy', 'train-metadata.npz', 'validation.npy', 'validation-metadata.npz')


def validate_config(cfg):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or 'RANK' in os.environ:
        raise ValueError('Run prepare_data.py once with python, outside torchrun')
    if cfg['resolution'] != 256:
        raise ValueError('This preparation recipe produces 256x256 images')
    if not re.fullmatch(r'[0-9a-f]{40}', cfg['source']['revision']):
        raise ValueError('Pin source.revision to a full Hugging Face commit hash')
    if not re.fullmatch(r'[0-9a-f]{64}', cfg['source']['sha256']):
        raise ValueError('Set the expected archive SHA256')
    total = cfg['source']['image_count']
    count = cfg['max_images'] if cfg['max_images'] is not None else total
    if not isinstance(count, int) or not 2 <= count <= total:
        raise ValueError('max_images must be between 2 and the source image count')
    if not isinstance(cfg['eval_count'], int) or not 0 < cfg['eval_count'] < count:
        raise ValueError('eval_count must leave both train and validation nonempty')
    if not isinstance(cfg['split_seed'], int) or not 0 <= cfg['split_seed'] < 2**64:
        raise ValueError('split_seed must be an unsigned 64-bit integer')
    data, downloads, output = (project_path(cfg[k]).resolve() for k in ('data_dir', 'download_dir', 'output_dir'))
    if any(a.is_relative_to(b) or b.is_relative_to(a)
           for a, b in ((data, downloads), (data, output), (downloads, output))):
        raise ValueError('Data, downloads and run outputs must be separate directories')
    if any(path == Path(path.anchor) for path in (data, downloads, output)):
        raise ValueError('Data, downloads and run outputs cannot be filesystem roots')


def contract(cfg):
    return dict(format='d2-celebahq-rgb-u8-v1', source=cfg['source'],
                resolution=cfg['resolution'], resample='Pillow.LANCZOS', color='RGB',
                split_seed=cfg['split_seed'], split_algorithm='numpy.PCG64-permutation-eval-first',
                eval_count=cfg['eval_count'], max_images=cfg['max_images'],
                attribute_names=list(ATTRIBUTE_NAMES),
                versions={name: importlib.metadata.version(name) for name in ('numpy', 'Pillow')})


def split_ids(cfg):
    count = cfg['max_images'] if cfg['max_images'] is not None else cfg['source']['image_count']
    # Split original numeric IDs, never ZIP iteration order. Keep randomized
    # validation order so the trainer's first 512 rows form a random dev subset.
    order = np.random.Generator(np.random.PCG64(cfg['split_seed'])).permutation(count)
    boundary = cfg['eval_count']
    return dict(train=order[boundary:].astype(np.int64), validation=order[:boundary].astype(np.int64))


def parse_attributes(text, expected_count):
    lines = [line.split() for line in text.splitlines() if line.strip()]
    if len(lines) < 2 or lines[0] != [str(expected_count)]:
        raise ValueError('Attribute annotation count differs from the image population')
    names = lines[1]
    if len(names) != 40 or set(names) != set(ATTRIBUTE_NAMES):
        raise ValueError('Expected the canonical 40 CelebA attributes')
    columns = [names.index(name) for name in ATTRIBUTE_NAMES]
    attrs = np.zeros((expected_count, 40), dtype=np.uint8)
    seen = set()
    for fields in lines[2:]:
        if len(fields) != 41 or not re.fullmatch(r'\d+\.jpg', fields[0]):
            raise ValueError('Malformed attribute row')
        image_id = int(PurePosixPath(fields[0]).stem)
        values = np.asarray([int(value) for value in fields[1:]], dtype=np.int64)
        if not 0 <= image_id < expected_count or image_id in seen:
            raise ValueError('Duplicate or out-of-range attribute image ID')
        if not np.isin(values, [-1, 1]).all():
            raise ValueError('Expected -1/+1 source attribute labels')
        attrs[image_id] = (values[columns] > 0).astype(np.uint8)
        seen.add(image_id)
    if len(seen) != expected_count:
        raise ValueError('Missing attribute annotations')
    return attrs


def image_members(archive, source):
    members = {}
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if info.is_dir() or path.parent.as_posix() != source['image_directory']:
            continue
        if path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        if not path.stem.isdigit():
            raise ValueError(f'Image filename lacks a numeric ID: {path.name}')
        image_id = int(path.stem)
        if image_id in members:
            raise ValueError(f'Duplicate image ID: {image_id}')
        members[image_id] = info
    if set(members) != set(range(source['image_count'])):
        raise ValueError('Archive must contain each expected CelebA-HQ image ID exactly once')
    return members


def verify_cache(folder, cfg, check_hashes=True):
    manifest = read(folder / 'manifest.json')
    if manifest['contract'] != contract(cfg) or set(manifest['files']) != set(FILES):
        raise ValueError('Existing data cache uses a different preparation recipe')
    expected_splits = split_ids(cfg)
    if manifest['counts'] != {split: len(ids) for split, ids in expected_splits.items()}:
        raise ValueError('Cache manifest counts differ from the deterministic split')
    for split, expected_ids in expected_splits.items():
        images = np.load(folder / f'{split}.npy', mmap_mode='r', allow_pickle=False)
        if images.shape != (len(expected_ids), 3, 256, 256) or images.dtype != np.uint8:
            raise ValueError(f'Invalid {split} image shape/dtype')
        del images
        with np.load(folder / f'{split}-metadata.npz', allow_pickle=False) as meta:
            if not np.array_equal(meta['ids'], expected_ids):
                raise ValueError(f'{split} IDs differ from the deterministic split')
            if meta['attributes'].shape != (len(expected_ids), 40) or not np.isin(meta['attributes'], [0, 1]).all():
                raise ValueError(f'Invalid {split} attributes')
            if meta['attribute_names'].tolist() != list(ATTRIBUTE_NAMES):
                raise ValueError('Attribute column order differs')
    for name in FILES:
        path = folder / name
        if path.stat().st_size != manifest['files'][name]['bytes']:
            raise ValueError(f'Cache file size differs: {name}')
        if check_hashes and sha(path) != manifest['files'][name]['sha256']:
            raise ValueError(f'Cache checksum differs: {name}')
    return manifest


def prepare(cfg, log, local_archive=None):
    target = project_path(cfg['data_dir']).resolve()
    if target.exists():
        if not (target / 'manifest.json').is_file():
            raise FileExistsError(f'Refusing an incomplete/unrelated cache: {target}')
        manifest = verify_cache(target, cfg)
        log.emit('cache_verified', 0, reused=True, counts=manifest['counts'])
        return dict(passed=True, reused=True, data_dir=str(target), counts=manifest['counts'])
    source = cfg['source']
    log.emit('download_start', 0)
    if local_archive:
        archive_path = Path(local_archive)
    else:
        from huggingface_hub import hf_hub_download

        archive_path = Path(hf_hub_download(
            repo_id=source['repo_id'], repo_type='dataset', revision=source['revision'],
            filename=source['filename'], cache_dir=project_path(cfg['download_dir']),
            token=os.environ.get('HF_TOKEN', False)))
    if sha(archive_path) != source['sha256']:
        raise ValueError('Downloaded/local archive SHA256 differs from the pinned source')
    log.emit('download_verified', 0, bytes=archive_path.stat().st_size)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(target.name + '.incomplete-' + uuid.uuid4().hex)
    staging.mkdir()
    write(project_path(cfg['output_dir']) / 'staging.json', dict(path=str(staging), target=str(target)))
    splits = split_ids(cfg)
    processed, dimensions = 0, {}
    with zipfile.ZipFile(archive_path) as archive:
        members = image_members(archive, source)
        attributes = parse_attributes(archive.read(source['attribute_file']).decode('utf-8-sig'), source['image_count'])
        # Read members directly; masks and other archive contents are not extracted.
        for split, ids in splits.items():
            pixels = np.lib.format.open_memmap(staging / f'{split}.npy', mode='w+',
                dtype=np.uint8, shape=(len(ids), 3, 256, 256))
            try:
                for offset, image_id in enumerate(ids):
                    with archive.open(members[int(image_id)]) as member, Image.open(member) as original:
                        image = ImageOps.exif_transpose(original).convert('RGB')
                        if image.width != image.height:
                            raise ValueError(f'Expected a square CelebA-HQ image: {image_id}')
                        shape = f'{image.width}x{image.height}'
                        dimensions[shape] = dimensions.get(shape, 0) + 1
                        resized = image.resize((256, 256), resample=Image.Resampling.LANCZOS)
                        pixels[offset] = np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1)
                    processed += 1
                    if processed % 500 == 0 or offset + 1 == len(ids):
                        log.emit('images_prepared', processed, split=split, split_images=offset + 1,
                                     total_images=sum(len(value) for value in splits.values()))
                pixels.flush()
            finally:
                del pixels
            np.savez(staging / f'{split}-metadata.npz', ids=ids, attributes=attributes[ids],
                     attribute_names=np.asarray(ATTRIBUTE_NAMES))
    manifest = dict(contract=contract(cfg), counts={split: len(ids) for split, ids in splits.items()},
        source_dimensions=dimensions, attribute_names=list(ATTRIBUTE_NAMES),
        files={name: dict(bytes=(staging / name).stat().st_size, sha256=sha(staging / name)) for name in FILES})
    write(staging / 'manifest.json', manifest)
    verify_cache(staging, cfg, check_hashes=False)
    if target.exists() or staging.resolve().parent != target.parent:
        raise FileExistsError(f'Refusing to replace the destination: {target}')
    staging.rename(target)
    log.emit('data_ready', processed, reused=False, counts=manifest['counts'])
    return dict(passed=True, reused=False, data_dir=str(target), counts=manifest['counts'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'prepare_data.json')
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--data-dir', type=Path, help='Destination cache, relative to the project')
    parser.add_argument('--output', type=Path, help='Fresh log directory, relative to the project')
    parser.add_argument('--eval-count', type=int)
    parser.add_argument('--split-seed', type=int)
    parser.add_argument('--max-images', type=int, help='Optional subset of the first N numeric IDs; default all 30,000')
    parser.add_argument('--local-archive', type=Path, help='Reuse an already downloaded ZIP; its SHA256 is still checked')
    args = parser.parse_args()
    bootstrap(args.env_file, training=False)
    cfg = read(args.config)
    for name in ('eval_count', 'split_seed', 'max_images'):
        value = getattr(args, name)
        if value is not None:
            cfg[name] = value
    if args.data_dir:
        cfg['data_dir'] = args.data_dir.as_posix()
    if args.output:
        cfg['output_dir'] = args.output.as_posix()
    validate_config(cfg)
    with run_log(cfg, script_sha256=sha(__file__), training_script_sha256=sha(PROJECT / 'train.py'),
                 local_archive=str(args.local_archive.resolve()) if args.local_archive else None) as (output, log):
        result = prepare(cfg, log, args.local_archive)
        # Explicit snippet: preparation never rewrites an active training config.
        write(output / 'dataset-config.json', dict(dataset=dict(cache=cfg['data_dir'],
            train_count=result['counts']['train'], validation_count=result['counts']['validation'])))
        write(output / 'result.json', result)
        log.emit('run_complete', sum(result['counts'].values()), passed=True)


if __name__ == '__main__':
    main()
