# Third-party software and model assets

Direct-Forcing contains modified source derived from ViRDM, Self-Forcing,
Causal Forcing, Wan2.1, and the other projects listed below. Original copyright
and license notices are retained where supplied.

[Source provenance](../assets/source_provenance.json) records the source file
and source hash for the adapted runtime. Model weights, training data, and
generated reference caches are not included in this repository.

[TAEHV / TAEW](taehv/) bundles the small upstream autoencoder implementation used
by our TAEW2.1 training decoder and retains its [MIT license](taehv/LICENSE).
[The Direct-Forcing decoder adapter](../model/decoders/taew.py) provides the
interface and latent-gradient propagation. Keeping the upstream implementation
and its license together makes attribution and updates straightforward; no
separate clone is needed. Weights are downloaded separately.

| Component | Source | Use | License |
|---|---|---|---|
| ViRDM | https://github.com/neu-vi/ViRDM | RDM kernel, staged VJP, base training runtime | Apache-2.0 |
| Self-Forcing | https://github.com/guandeh17/Self-Forcing | Causal Wan architecture and rollout; expanded evaluation prompts | Apache-2.0 |
| Causal Forcing | https://github.com/thu-ml/Causal-Forcing | Causal training data format and evaluation recipe | Apache-2.0 |
| Wan2.1 | https://github.com/Wan-Video/Wan2.1 | Generator, text encoder, VAE | Apache-2.0 |
| Diffusers | https://github.com/huggingface/diffusers | Frozen bidirectional DiT | Apache-2.0 |
| TAEHV / TAEW | https://github.com/madebyollin/taehv | Differentiable lightweight decoder | [MIT](taehv/LICENSE) |
| VideoMAE v2 | https://github.com/OpenGVLab/VideoMAEv2 | Frozen video features; downloaded separately | Consult the upstream code and model licenses |
| SigLIP2 / OpenCLIP | https://github.com/mlfoundations/open_clip | Frozen prompt features; downloaded separately | Consult the upstream code and model licenses |
| FlashAttention | https://github.com/Dao-AILab/flash-attention | Attention kernels; installed separately | BSD-3-Clause |
| VBench | https://github.com/Vchitect/VBench | Evaluation; installed separately | Consult the upstream license |

The repository's [Apache-2.0 license](../LICENSE) does not relicense third-party
model weights or datasets. Users obtain those assets from their original providers.
