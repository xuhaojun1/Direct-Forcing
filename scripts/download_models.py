"""Download upstream model assets into the paths configured for Direct-Forcing."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import load_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/chunkwise_4step.yaml')
    parser.add_argument('--models', nargs='+', choices=['wan', 'dit', 'videomae', 'taew'], default=None)
    parser.add_argument('--list', action='store_true', help='Show the download plan without downloading')
    args = parser.parse_args()
    spec = load_spec(args.config)
    names = args.models or ['wan', 'videomae', 'taew'] + (['dit'] if 'dit' in spec.features.rdm or spec.features.cca == 'dit' else [])
    plans = {
        'wan': dict(repo_id='Wan-AI/Wan2.1-T2V-1.3B', revision='37ec512624d61f7aa208f7ea8140a131f93afc9a'),
        'dit': dict(repo_id='Wan-AI/Wan2.1-T2V-14B-Diffusers', revision='38ec498cb3208fb688890f8cc7e94ede2cbd7f68'),
        'videomae': dict(repo_id='OpenGVLab/VideoMAEv2-Large', revision='9981a9c8f77118c421e5228e1b219468a4b0238d'),
        'taew': dict(url='https://raw.githubusercontent.com/madebyollin/taehv/e743234f3217ab3d1570f65642ab06596d1bd7c5/taew2_1.pth',
                     sha256='d26151e76cdc2c9424bef988de874b33d9a53f30ef3060cd556c429c469c797e'),
    }
    if args.list:
        print(json.dumps({n: dict(plans[n], destination=spec.paths[n]) for n in names}, indent=2))
        return
    from huggingface_hub import HfApi, snapshot_download
    for name in names:
        plan, target = plans[name], Path(spec.paths[name])
        if name != 'taew':
            # Record the immutable upstream commit with the downloaded assets.
            revision = HfApi().model_info(plan['repo_id'], revision=plan['revision']).sha
            target.mkdir(parents=True, exist_ok=True)
            patterns = (['*.json', '*.py', '*.safetensors'] if name == 'videomae' else
                        ['transformer/*.json', 'transformer/*.safetensors'] if name == 'dit' else None)
            snapshot_download(plan['repo_id'], revision=revision, local_dir=target, allow_patterns=patterns)
            (target / 'direct_forcing_source.json').write_text(json.dumps(dict(plan, revision=revision), indent=2) + '\n')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == plan['sha256']:
                continue
            temporary = target.with_suffix('.download')
            try:
                with urllib.request.urlopen(plan['url']) as response, temporary.open('wb') as f:
                    shutil.copyfileobj(response, f)
                if hashlib.sha256(temporary.read_bytes()).hexdigest() != plan['sha256']:
                    raise ValueError('TAEW checkpoint checksum mismatch')
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        print(f'Ready: {name} -> {target}', flush=True)


if __name__ == '__main__':
    main()
