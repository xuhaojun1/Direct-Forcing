# Installation details

The [README installation commands](../README.md#installation) use the following
environment. These versions were used for import, configuration, and numerical
checks.

| Component | Version |
|---|---|
| OS / architecture | Linux x86_64 |
| Python | 3.11 (checked with 3.11.15) |
| PyTorch | 2.12.1+cu130 |
| torchvision | 0.27.1+cu130 |
| CUDA runtime in the PyTorch wheel | 13.0 |
| CUDA toolkit, when compiling extensions | 13.0 (checked with nvcc 13.0.48) |
| FlashAttention | 2.8.3 |
| Diffusers | 0.37.1 |
| Transformers | 4.57.0 |
| OpenCLIP | 3.3.0 |
| OmegaConf | 2.3.1 |

## 1. Driver, runtime, and compiler

These are three different components:

- **NVIDIA driver:** installed on the host. `nvidia-smi` reports it and the maximum
  CUDA version it supports. The documented wheel requires a CUDA-13-compatible driver.
- **PyTorch CUDA runtime:** supplied by the `cu130` PyTorch installation. Check it
  with `python -c "import torch; print(torch.__version__, torch.version.cuda)"`.
- **CUDA toolkit/compiler:** provides `nvcc`, needed when building FlashAttention
  from source. Installing PyTorch alone does not install `nvcc`.

Check the host before installing:

```bash
nvidia-smi
nvcc --version
```

If a CUDA 13.0 toolkit is already installed, activate it through your cluster's
module system or set its actual location. For example:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
```

Alternatively, install the toolkit inside the activated Conda environment:

```bash
conda activate direct-forcing
conda install -c nvidia cuda-toolkit=13.0 -y
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
```

Install the PyTorch / torchvision pair from the README before building
FlashAttention. The toolkit's CUDA major version must match PyTorch's CUDA runtime.
The host also needs a C++ compiler supported by that toolkit.

## 2. FlashAttention

The default recipe explicitly checks for `flash-attn==2.8.3`. The normal install
can use a compatible prebuilt wheel or compile from source:

```bash
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

For a mismatched or unavailable wheel, build in the active environment after
configuring CUDA 13.0 above:

```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 \
  python -m pip install --force-reinstall --no-deps --no-cache-dir \
  --no-binary=flash-attn --no-build-isolation flash-attn==2.8.3
```

`MAX_JOBS` limits parallel compiler jobs and host RAM use. Adjust it to the build
machine. Compile extensions for the GPUs that will run the job; copying a wheel
from a different PyTorch/CUDA installation can produce ABI errors even when its
package version matches.

FlashAttention 2.8.3's source build defaults to CUDA architectures
`80;90;100;120`, which cover A100, H100/H200, B200, and RTX PRO 6000 Blackwell.
With CUDA 13.0, this default supports a build intended for all four families.
An optional `FLASH_ATTN_CUDA_ARCHS` override can shorten compilation for a known
deployment; retain every architecture where the resulting wheel will be used.

On an available GPU, check the actual kernel:

```bash
python scripts/check_environment.py --attention-test --device 0
```

This allocates only small test tensors and checks a BF16 forward and backward.
It does not load Wan, VideoMAE, datasets, or reference caches. It is a kernel check,
not a training memory or throughput benchmark.

## 3. Other GPU or CUDA environments

The default launch uses eight GPUs; `memory.reference_residency=cpu` and
`memory.activation_offload=true` are enabled in the public configs. See the
[configuration guide](configuration.md) to change placement or GPU count.

If a machine cannot use CUDA 13.0, install a matching PyTorch / torchvision pair
from [PyTorch's official installation instructions](https://pytorch.org/get-started/locally/)
and rebuild FlashAttention against that environment. The source requirement is
PyTorch >=2.6, but that lower bound is not a claim that every supported version
has reproduced this release. B200 (`sm_100`) and RTX PRO 6000 Blackwell (`sm_120`)
need PyTorch and attention kernels supporting their respective architectures.

Keep **Diffusers 0.37.1**, **Transformers 4.57.0**, **OpenCLIP 3.3.0**, and
**FlashAttention 2.8.3** for the default recipe. In particular, the DiT feature
extractor checks the Diffusers version and retained model-weight hash.

## 4. Common installation errors

| Symptom | Check / fix |
|---|---|
| `torch.cuda.is_available()` is false | Run on a GPU node; check the driver, visible devices, and that PyTorch is a CUDA build. |
| `nvcc` or `CUDA_HOME` is missing | Install/activate the toolkit described above before compiling FlashAttention. |
| CUDA version mismatch during a build | Match the toolkit selected by `CUDA_HOME` to `torch.version.cuda`. |
| `undefined symbol` while importing FlashAttention | Rebuild it after installing the final PyTorch version; avoid a wheel from a different ABI. |
| `no kernel image is available` / invalid device function | Check support for the GPU's compute capability and rebuild the extension with the correct toolkit. |
| Missing `torchvision` operator | Reinstall the documented torch/torchvision pair from the same CUDA wheel index. |
| Missing models during `train.py --prepare-only` | Follow the [model downloads](../README.md#model-downloads) and data/reference preparation sections; dependency installation does not download these assets. |

Use `python scripts/check_environment.py --imports-only` on CPU/login nodes.
Set proxy variables before package or model downloads when your network requires
them. Keep VBench and its native extensions in a separate environment.
