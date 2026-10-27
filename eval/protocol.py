"""Fixed expanded VBench prompt protocol; preserves the reported deduplication."""
import hashlib
import json
from pathlib import Path

PROMPT_HASHES = {
    'all_dimension.txt': 'f9b50654f81ab732b9235d5bb91f0c3a4d152fac19249ce3a3a9185b2d80d8c3',
    'all_dimension_extended.txt': 'c78f1ad32974503efaa8cc935347ace88f3a8b571adde6687bc8425bbf417cf3',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_prompts(original, extended, full_info):
    for path in (original, extended):
        if sha256(path) != PROMPT_HASHES[path.name]:
            raise ValueError(f'Fixed VBench prompt file changed: {path}')
    originals = original.read_text().splitlines()
    expansions = extended.read_text().splitlines()
    metadata = [str(row['prompt_en']).strip() for row in json.loads(full_info.read_text())]
    if originals != metadata or len(originals) != 946 or len(expansions) != 946:
        raise ValueError('VBench originals, expansions and official metadata are not line-aligned')
    seen, prompts, conditioned = set(), [], []
    for prompt, expansion in zip(originals, expansions):
        if not expansion.strip():
            raise ValueError('Empty expanded prompt')
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
            conditioned.append(expansion)
    if len(prompts) != 944:
        raise ValueError('Expected 944 unique VBench prompts')
    expected_filenames(prompts)
    return prompts, conditioned


def expected_filenames(prompts):
    names = [f'{prompt}-{sample}.mp4' for prompt in prompts for sample in range(5)]
    if len(names) != len(set(names)) or any(
        '/' in name or '\x00' in name or len(name.encode()) > 255 for name in names
    ):
        raise ValueError('VBench prompt filenames collide or exceed filesystem limits')
    return names
