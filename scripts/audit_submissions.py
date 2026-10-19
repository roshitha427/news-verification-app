"""Inventory untrusted submissions without extraction, importing, or model loading.

The video repository ZIP profiles below describe the submitted file versions,
not the full datasets advertised by the upstream projects. Arbitrary ZIPs are
also supported. All archive document contents are treated as data.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import zipfile


VIDEO = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
AUDIO = {'.wav', '.mp3', '.aac', '.flac', '.ogg', '.m4a'}
IMAGE = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tif', '.tiff'}
CHECKPOINT = {'.pkl', '.pickle', '.p', '.pt', '.pth', '.dat', '.joblib', '.ckpt'}
METADATA_READ_LIMIT = 2 * 1024 * 1024

PROFILES = {
    'FakeAVCeleb-main.zip': {
        'expected_sha256': '0cc53b3ef49fc0f313873c98ad0279ffc556eeb5d3946ad850548ccfab56a548',
        'project': 'FakeAVCeleb',
        'source': 'https://github.com/DASH-Lab/FakeAVCeleb',
        'classification': 'source_repository_without_training_media',
        'usable_labeled_video_clips': 0,
        'can_train_video_model': False,
        'reason': 'No video or audio files. Three documentation PNGs are not a labeled corpus.',
        'label_scope': 'Upstream dataset distinguishes real/fake audio and real/fake video; labels are not present for media in this ZIP.',
        'access': 'Upstream README requires a dataset request form; accepted applicants receive a download link.',
        'license': 'README describes code as MIT; data has separate request-form terms. Included LICENSE is empty.',
        'grouping_requirement': 'Keep each identity and original source clip, including all audio/video manipulation variants, within one split.',
    },
    'FaceForensics-master.zip': {
        'expected_sha256': 'f28ece8b7c91a373ac5fb520f165737fa2687bcbe53feafcf0f653063aa9e161',
        'project': 'FaceForensics++',
        'source': 'https://github.com/ondyari/FaceForensics',
        'classification': 'source_repository_with_metadata_without_training_videos',
        'usable_labeled_video_clips': 0,
        'can_train_video_model': False,
        'reason': 'No videos or audio. PNGs/GIFs are icons, example composites, masks, and demonstrations.',
        'label_scope': 'Upstream original_sequences are authentic; manipulated_sequences are face manipulations. This is a media-manipulation task, not news truth.',
        'access': 'Upstream dataset requires an approved request form and a separately supplied downloader.',
        'license': 'Code MIT; dataset is separately covered by FaceForensics Terms of Use.',
        'grouping_requirement': 'Group original YouTube video IDs and connected manipulation source/target IDs before splitting; keep derived frames and quality versions together.',
    },
    'celeb-deepfakeforensics-master.zip': {
        'expected_sha256': 'fc775397a936efdaa857aef597ef22155da3e6bb3e882411421254c74837cf09',
        'project': 'Celeb-DF',
        'source': 'https://github.com/yuezunli/celeb-deepfakeforensics',
        'classification': 'documentation_with_single_unlabeled_demo_video',
        'usable_labeled_video_clips': 0,
        'can_train_video_model': False,
        'reason': 'Only src/demo.mp4 is a video; no Celeb-real, YouTube-real, Celeb-synthesis clips or List_of_testing_videos.txt are supplied.',
        'label_scope': 'Upstream real and synthesized celebrity clips; demo montage has no per-clip label manifest.',
        'access': 'Upstream README requires acceptance through a request form for actual dataset download.',
        'license': 'Data is covered by Terms to Use Celeb-DF in its request form, not a blanket code license.',
        'grouping_requirement': 'Preserve source clip and source/target celebrity identities; do not randomly split extracted frames.',
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def unsafe_member(name: str) -> bool:
    normalized = name.replace('\\', '/')
    return (
        PurePosixPath(normalized).is_absolute()
        or bool(PureWindowsPath(name).drive)
        or '..' in PurePosixPath(normalized).parts
        or ':' in normalized
        or '\x00' in normalized
    )


def bounded_json(archive: zipfile.ZipFile, name: str):
    info = archive.getinfo(name)
    if unsafe_member(name) or info.file_size > METADATA_READ_LIMIT:
        raise ValueError('Metadata member cannot be read safely')
    return json.loads(archive.read(info))


def faceforensics_splits(archive: zipfile.ZipFile) -> dict:
    prefix = 'FaceForensics-master/dataset/'
    expected = [prefix + 'splits/' + name + '.json' for name in ('train', 'val', 'test')]
    if not all(name in archive.namelist() for name in expected):
        return {}
    sets = {}
    result = {'splits': {}}
    for split, name in zip(('train', 'val', 'test'), expected):
        pairs = bounded_json(archive, name)
        if not isinstance(pairs, list) or not all(
            isinstance(pair, list) and len(pair) == 2
            and all(isinstance(value, str) for value in pair) for pair in pairs
        ):
            raise ValueError('Unexpected split metadata shape')
        sets[split] = {item for pair in pairs for item in pair}
        result['splits'][split] = {'pair_count': len(pairs), 'unique_sequence_ids': len(sets[split])}
    combinations = [('train', 'val'), ('train', 'test'), ('val', 'test')]
    result['sequence_overlap'] = {
        a + '/' + b: sorted(sets[a] & sets[b]) for a, b in combinations
    }
    conversion_name = prefix + 'conversion_dict.json'
    if conversion_name in archive.namelist():
        conversion = bounded_json(archive, conversion_name)
        youtube = {
            split: {conversion[key].rsplit(' ', 1)[0] for key in keys}
            for split, keys in sets.items()
        }
        result['youtube_source_overlap'] = {
            a + '/' + b: sorted(youtube[a] & youtube[b]) for a, b in combinations
        }
        result['unique_youtube_sources'] = {split: len(values) for split, values in youtube.items()}
        result['source_video_independent'] = not any(result['youtube_source_overlap'].values())
        result['identity_independence'] = 'Unknown; no person identity mapping was supplied.'
    return result


def audit_archive(path: Path) -> dict:
    digest = sha256_file(path)
    known_profile = PROFILES.get(path.name, {})
    profile = (
        known_profile if known_profile.get('expected_sha256') == digest
        else {'classification': 'unclassified_submission', 'reason': 'No manually reviewed profile matches the archive SHA-256.'}
    )
    result = {
        'path': str(path.resolve()),
        'name': path.name,
        'bytes': path.stat().st_size,
        'sha256': digest,
        'profile': profile,
    }
    with zipfile.ZipFile(path) as archive:
        files = [info for info in archive.infolist() if not info.is_dir()]
        extensions = Counter(Path(info.filename).suffix.lower() for info in files)
        result.update(
            file_count=len(files),
            uncompressed_bytes=sum(info.file_size for info in files),
            extensions=dict(sorted(extensions.items())),
            video_file_count=sum(extensions[e] for e in VIDEO),
            audio_file_count=sum(extensions[e] for e in AUDIO),
            image_file_count=sum(extensions[e] for e in IMAGE),
            checkpoint_file_count=sum(extensions[e] for e in CHECKPOINT),
        )
        result['warnings'] = []
        for info in files:
            if unsafe_member(info.filename):
                result['warnings'].append({'path': info.filename, 'reason': 'unsafe_member_path'})
            if stat.S_ISLNK(info.external_attr >> 16):
                result['warnings'].append({'path': info.filename, 'reason': 'symbolic_link'})
            if info.flag_bits & 1:
                result['warnings'].append({'path': info.filename, 'reason': 'encrypted_member'})
            if info.file_size > 1024 * 1024 and info.file_size / max(1, info.compress_size) > 1000:
                result['warnings'].append({'path': info.filename, 'reason': 'high_expansion_ratio'})
        # Paths are recorded as data and are never used as output paths.
        result['files'] = [
            {'path': info.filename, 'bytes': info.file_size, 'compressed_bytes': info.compress_size}
            for info in files
        ]
        if path.name == 'FaceForensics-master.zip':
            result['included_split_metadata'] = faceforensics_splits(archive)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', nargs='+', type=Path, required=True)
    parser.add_argument('--sidecars', nargs='*', type=Path, default=[])
    parser.add_argument('--output', type=Path, default=Path('data/submission-audit-video.json'))
    args = parser.parse_args()
    result = {
        'schema_version': 1,
        'audited_at': datetime.now(timezone.utc).isoformat(),
        'method': 'Read-only ZIP central-directory inspection, SHA-256, and bounded inert JSON parsing. No extraction or execution; no checkpoint deserialization.',
        'archives': [audit_archive(path) for path in args.archives],
        'sidecars': [
            {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
            for path in args.sidecars
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding='utf-8')
    print(json.dumps({
        'output': str(args.output),
        'archives': [
            {key: item[key] for key in ('name', 'file_count', 'video_file_count', 'audio_file_count', 'image_file_count', 'sha256')}
            for item in result['archives']
        ],
    }, indent=2))


if __name__ == '__main__':
    main()
