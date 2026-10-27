"""Check the installation without loading model weights or training data."""
import argparse
import importlib
from importlib.metadata import version
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--imports-only', action='store_true', help='Skip GPU discovery on CPU/login nodes')
    mode.add_argument('--attention-test', action='store_true', help='Run a tiny BF16 FlashAttention forward/backward')
    parser.add_argument('--device', type=int, default=0, help='Visible CUDA device index, default: 0')
    args = parser.parse_args()
    errors = []
    print(f'Python {sys.version.split()[0]}')
    if sys.version_info < (3, 10):
        errors.append('Python >= 3.10 is required; the documented environment uses 3.11.')
    packages = [
        ('torch', 'torch', None), ('torchvision', 'torchvision', None),
        ('diffusers', 'diffusers', '0.37.1'), ('transformers', 'transformers', '4.57.0'),
        ('open_clip_torch', 'open_clip', '3.3.0'), ('flash-attn', 'flash_attn', '2.8.3'),
        ('omegaconf', 'omegaconf', '2.3.1'), ('huggingface_hub', 'huggingface_hub', None),
        ('lmdb', 'lmdb', None), ('safetensors', 'safetensors', None),
        ('scipy', 'scipy', None), ('imageio-ffmpeg', 'imageio_ffmpeg', None),
    ]
    for distribution, module, expected in packages:
        try:
            importlib.import_module(module)
            installed = version(distribution)
            print(f'  {distribution}: {installed}')
            if expected and installed != expected:
                errors.append(f'{distribution} must be {expected} for the default recipe; found {installed}.')
        except Exception as error:
            errors.append(f'{distribution}: {type(error).__name__}: {error}')
    if errors:
        raise SystemExit('Installation check failed:\n- ' + '\n- '.join(errors))
    import torch
    from packaging.version import Version
    if Version(torch.__version__).release[:2] < (2, 6):
        raise SystemExit('PyTorch >= 2.6 is required; see the documented installation pair.')
    if not args.imports_only:
        if not torch.cuda.is_available():
            raise SystemExit('CUDA is unavailable. Check the NVIDIA driver and CUDA PyTorch wheel, or use --imports-only.')
        if not 0 <= args.device < torch.cuda.device_count():
            raise SystemExit(f'--device must be between 0 and {torch.cuda.device_count() - 1}.')
        with torch.cuda.device(args.device):
            info = torch.cuda.get_device_properties(args.device)
            print(f'GPU {args.device}: {info.name}, sm_{info.major}{info.minor}, {info.total_memory / 2**30:.1f} GiB')
            print(f'PyTorch CUDA runtime: {torch.version.cuda}')
            if not torch.cuda.is_bf16_supported():
                raise SystemExit('This recipe requires BF16 support.')
            if args.attention_test:
                from flash_attn import flash_attn_func
                q, k, v = [torch.randn(1, 64, 4, 64, device=f'cuda:{args.device}',
                                      dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
                output = flash_attn_func(q, k, v, causal=True)
                output.float().square().mean().backward()
                torch.cuda.synchronize()
                if not torch.isfinite(output).all() or not all(
                        x.grad is not None and torch.isfinite(x.grad).all() for x in (q, k, v)):
                    raise SystemExit('FlashAttention produced invalid outputs or gradients.')
                print('FlashAttention BF16 forward/backward: passed')
    print('Installation check passed.' if not args.imports_only else 'Dependency imports passed; GPU execution was not tested.')


if __name__ == '__main__':
    main()
