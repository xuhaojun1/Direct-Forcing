#!/usr/bin/env python3
"""Run the 16 official VBench dimensions on a standard-named video folder."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "temporal_style",
    "appearance_style",
    "overall_consistency",
]


def standard_metadata(videos_path, full_info, dimensions):
    """Use exact standard MP4 names; export markers are not video suffixes."""
    rows = json.loads(Path(full_info).read_text())
    result = []
    for row in rows:
        if not set(dimensions).intersection(row['dimension']):
            continue
        paths = [Path(videos_path)/f'{row["prompt_en"]}-{index}.mp4' for index in range(5)]
        missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise ValueError(f'Missing standard VBench videos: {missing[:5]}')
        result.append(dict(row, video_list=[str(path) for path in paths]))
    covered = {dim for row in result for dim in row['dimension']}
    if not set(dimensions).issubset(covered):
        raise ValueError(f'No metadata for dimensions: {set(dimensions)-covered}')
    return result


def main() -> None:
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-path", type=Path, required=True)
    parser.add_argument("--full-info", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dimensions", nargs="*", default=DEFAULT_DIMENSIONS)
    parser.add_argument(
        "--local-models",
        action="store_true",
        help="Use/download VBench assets under VBENCH_CACHE_DIR.",
    )
    args = parser.parse_args()

    try:
        from vbench import VBench
        from vbench.distributed import barrier, dist_init, get_rank
    except ImportError as exc:
        raise SystemExit(
            "VBench is not installed. Follow README.md > Evaluation to create "
            "the separate evaluation environment."
        ) from exc

    if not args.videos_path.is_dir():
        raise SystemExit(f"video directory not found: {args.videos_path}")
    if not args.full_info.is_file():
        raise SystemExit(f"VBench metadata not found: {args.full_info}")
    if not args.dimensions:
        raise SystemExit("at least one VBench dimension is required")

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed and not torch.distributed.is_initialized():
        dist_init()
    rank = get_rank()
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        barrier()

    # Upstream infers the extension from os.listdir()[0] and has every rank
    # write the same metadata file. Export markers and concurrent writes make
    # that unreliable. Preserve official prompt/auxiliary data and scoring;
    # commit exact standard filenames once before all ranks start reading.
    class StandardVBench(VBench):
        def build_full_info_json(self, videos_path, name, dimension_list, *args, **kwargs):
            output = Path(self.output_path)/f'{name}_full_info.json'
            if rank == 0:
                rows = standard_metadata(videos_path, self.full_info_dir, dimension_list)
                temporary = output.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(rows, indent=2) + '\n')
                temporary.replace(output)
            if distributed:
                barrier()
            return str(output)

    evaluator = StandardVBench(
        torch.device(args.device), str(args.full_info), str(args.output_dir)
    )
    evaluator.evaluate(
        videos_path=str(args.videos_path),
        name=args.name,
        dimension_list=args.dimensions,
        local=args.local_models,
        mode="vbench_standard",
    )

    if distributed:
        barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
