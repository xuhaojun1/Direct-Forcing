"""Shared data, probes, scheduling, checkpointing, and resume for training."""

from contextlib import contextmanager
from datetime import timedelta
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, StateDictType,
    FullStateDictConfig, FullOptimStateDictConfig)
from omegaconf import OmegaConf
import wandb

from trainer.rdm import RDMTrainer
from utils.clean_data import open_clean_latents, clean_data_identity
from utils.training_schedule import configured_learning_rate, training_batch
from utils.wan_resume import validate_resume_contract, repartition_seed
from utils.wan_initialization import validate_initialization
from trainer.controller import FixedCCAController
from utils.logging_metrics import training_metrics, evaluation_metrics
from utils.transport import exact_ot, squared_cost
from utils.atomic import atomic_save
from utils.early_stop import CheckpointOTStopper, checkpoint_metric


class TrainingRuntime(RDMTrainer):
    @staticmethod
    def _validate_static_config(config):
        RDMTrainer._validate_static_config(config)
        if config.rdm_pairing_mode != 'none':
            raise ValueError('Use CCA for paired supervision')
        if config.get('joint_global_batch') != config.rdm_expected_global_rows:
            raise ValueError('Joint and RDM global batch settings disagree')
        validate_initialization(config)
        configured_learning_rate(1, config, total=int(config.max_steps))
        if config.get('joint_early_stop') is not None:
            stopper = CheckpointOTStopper(config.joint_early_stop)
            if int(config.log_iters) != stopper.policy['checkpoint_every']:
                raise ValueError('Temporal OT stopping requires checkpoints every 25 updates')


    def __init__(self, config):
        super().__init__(config)
        self.rank = dist.get_rank()
        self.run_path = Path(self.output_path)
        self.io_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=7200))
        self.controller = self._make_controller(config)
        self.temporal_early_stop = (CheckpointOTStopper(config.joint_early_stop)
                                   if config.get('joint_early_stop') is not None else None)
        self.clean_data = open_clean_latents(config.clean_data_path)
        self._prepare_training_state(config)

    def _make_controller(self, config):
        return FixedCCAController(0.)

    def _controller_from_state_dict(self, state):
        return FixedCCAController.from_state_dict(state)

    def _extend_contract(self, config):
        pass

    _training_metrics = staticmethod(training_metrics)

    def _select_probe_ids(self, config, sizes):
        selected = random.Random(int(config.probe_split_seed)).sample(range(len(self.dataset)), sum(sizes))
        c, r, _ = sizes
        return selected[:c], selected[c:c+r], selected[c+r:]

    def _prepare_training_state(self, config):
        if len(self.clean_data) != len(self.dataset):
            raise ValueError("Clean GT and prompt tables differ in length")
        sizes = [int(config.control_probe_count), int(config.report_probe_count), int(config.calibration_probe_count)]
        if any(n < 2 or n % self.world_size for n in sizes) or sizes[2] % 2:
            raise ValueError("Probe sizes must divide world size; calibration count must be even")
        self.control_ids, self.report_ids, self.calibration_ids = self._select_probe_ids(config, sizes)
        selected = self.control_ids+self.report_ids+self.calibration_ids
        excluded = set(selected)
        self.training_ids = [i for i in range(len(self.dataset)) if i not in excluded]
        self.pending_batch = []
        self.probe_gt = {}
        data_identity = clean_data_identity(config.clean_data_path)
        cache_contract = dict(encoder=config.rdm_encoder_checkpoint_sha256,
            decoder=config.rdm_taew_checkpoint_sha256, input="normalized clean Wan latent; TAEW; frozen video encoder",
            clean_data=str(Path(config.clean_data_path).resolve()),
            bytes=data_identity.get('gt_lmdb_bytes'), mtime_ns=data_identity.get('gt_lmdb_mtime_ns'),
            compute="bf16", output="float32")
        cache_contract.update(encoder_contract=self.rdm_encoder.contract())
        if self.latent_feature_encoder:
            cache_contract.update(input='normalized clean Wan latent; frozen conditional DiT; raw block8 tokens',
                                  decoder=None)
        cache_key = hashlib.sha256(json.dumps(cache_contract, sort_keys=True).encode()).hexdigest()[:20]
        self.gt_cache = Path(config.feature_cache_root)/cache_key
        self.gt_cache.mkdir(parents=True, exist_ok=True)
        if self.is_main_process:
            (self.gt_cache/"contract.json").write_text(json.dumps(cache_contract, indent=2)+"\n")
        self._event("preparing_gt_probes", count=len(selected), gt_cache=str(self.gt_cache))
        for index in selected[self.rank::self.world_size]:
            self._gt_feature(index)
        dist.barrier(group=self.io_group)
        for index in selected:
            self.probe_gt[index] = self._gt_feature(index)
        calibration = torch.cat([self.probe_gt[i] for i in self.calibration_ids]).double().numpy()
        self.gt_variance = float(np.square(calibration-calibration.mean(0)).sum()/len(calibration))
        costs = squared_cost(calibration, calibration)
        self.probe_sigma2 = float(np.median(costs[np.triu_indices(len(calibration), 1)]))
        half = len(calibration)//2
        self.null_ot = exact_ot(calibration[:half], calibration[half:])[0]
        if self.gt_variance <= 0 or self.probe_sigma2 <= 0:
            raise ValueError("Degenerate frozen GT calibration")
        self.contract = {key: OmegaConf.to_container(config, resolve=True)[key] for key in (
            "rdm_pairing_mode", "max_steps", "lr", "lr_floor", "lr_warmup", "lr_decay_start",
            "seed", "control_every", "probe_split_seed",
            "rdm_expected_global_rows", "rdm_expected_world_size", "smoke_test")}
        self.contract.update(control_ids=self.control_ids, report_ids=self.report_ids,
            calibration_ids=self.calibration_ids, gt_cache_contract=cache_contract,
            gt_variance=self.gt_variance, null_ot=self.null_ot,
            reference_sha256=config.rdm_reference_bundle_sha256)
        if config.get('dataset_id') is not None:
            self.contract['dataset'] = dict(id=config.dataset_id,
                manifest_sha256=config.dataset_manifest_sha256,
                rows=int(config.rdm_expected_reference_rows))
        if config.get('lr_schedule') is not None:
            self.contract.update(lr_schedule=config.lr_schedule,
                lr_cosine_end=int(config.lr_cosine_end))
        self._extend_contract(config)
        from utils.portable import portable_contract
        self.contract = portable_contract(self.contract)
        if self.is_main_process:
            (self.run_path/"training_contract.json").write_text(json.dumps(self.contract, indent=2)+"\n")
        if config.resume_from:
            self._restore(Path(config.resume_from))
        self._event("ready", step=self.step, population=config.rdm_expected_global_rows,
            training_rows=len(self.training_ids), gt_variance=self.gt_variance, gt_null_ot=self.null_ot)

    def _event(self, phase, **values):
        if self.is_main_process:
            record = {"phase": phase, **values}
            print(json.dumps(record, allow_nan=False), flush=True)
            temporary = self.run_path/"status.json.tmp"
            temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
            temporary.replace(self.run_path/"status.json")

    @torch.no_grad()
    def _gt_feature(self, index):
        path = self.gt_cache/f"row{index:04d}.pt"
        if path.exists():
            value = torch.load(path, map_location="cpu", weights_only=True)
        else:
            row = self.clean_data[index]
            if row["caption"] != self.dataset[index]["prompts"]:
                raise ValueError(f"GT caption mismatch at row {index}")
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                latent = row["wan_gt_latents"].unsqueeze(0).transpose(1, 2).to(self.device, self.dtype)
                if self.latent_feature_encoder:
                    embedding = self._get_conditioning([row['caption']])['prompt_embeds']
                    value = self.rdm_encoder(latent.transpose(1, 2), embedding).float().cpu()
                else:
                    decoded = self.taew_decoder.decode_value(latent)
                    value = self.rdm_encoder((decoded.float()*.5+.5).clamp(0, 1)).float().cpu()
            atomic_save(value, path)
        if value.shape != (1, int(self.config.rdm_feature_dim)) or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"Invalid cached GT feature at {path}")
        return value

    def _next_rdm_batch(self):
        if not self.pending_batch:
            batch = training_batch(self.training_ids, step=self.step+1,
                population=int(self.config.rdm_expected_global_rows), seed=int(self.config.seed))
            self.pending_batch = list(reversed(batch[self.rank::self.world_size]))
        index = self.pending_batch.pop()
        return {"idx": torch.tensor([index]), "prompts": [self.dataset[index]["prompts"]]}

    def _evaluate(self, role):
        from pipeline.causal_inference import CausalInferencePipeline
        ids = self.control_ids if role == "control" else self.report_ids
        self._event("evaluation", role=role, step=self.step, count=len(ids))
        self.model.inference_pipeline = None
        self.model.eval()
        owner = self

        class Text(torch.nn.Module):
            def forward(self, text_prompts):
                return owner._get_conditioning(text_prompts)

        pipeline = CausalInferencePipeline(self.config, self.device, generator=self.model.generator,
            text_encoder=Text(), vae=torch.nn.Identity())
        values = []
        with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            for index in ids[self.rank::self.world_size]:
                torch.manual_seed(int(self.config.eval_seed)*100000+index)
                noise = torch.randn(1, 21, 16, 60, 104, device=self.device, dtype=self.dtype)
                prompt = self.dataset[index]["prompts"]
                _, latent = pipeline.inference(noise, [prompt], return_latents=True, return_video=False)
                decoded = self.taew_decoder.decode_value(latent)
                feature = (self.rdm_encoder(latent.transpose(1,2), self._get_conditioning([prompt])['prompt_embeds'])
                    if self.latent_feature_encoder else self.rdm_encoder((decoded.float()*.5+.5).clamp(0, 1))).float()
                values.append(feature)
        local = torch.cat(values)
        parts = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(parts, local)
        if self.is_main_process:
            rank_order = [i for rank in range(self.world_size) for i in ids[rank::self.world_size]]
            x = torch.cat(parts).cpu().numpy()[[rank_order.index(i) for i in ids]]
            y = torch.cat([self.probe_gt[i] for i in ids]).double().numpy()
            text = None if self.joint_text_table is None else self.joint_text_table[ids].cpu().numpy()
            from utils.token_metrics import compare_tokens
            metrics = compare_tokens(x, y, self.rdm_reference, text, self.device)
            target = self.run_path/"evaluation"/f"{role}_step{self.step:06d}.json"
            target.parent.mkdir(exist_ok=True)
            target.write_text(json.dumps(metrics, indent=2, allow_nan=False)+"\n")
            logs = {f"{role}/ot_w2_squared": metrics["empirical_w2_squared"],
                f"{role}/mean_shift_squared": metrics["squared_mean_shift"],
                f"{role}/centered_ot": metrics["centered_empirical_w2_squared"],
                f"{role}/official_mmd": metrics["official_joint_nystrom_mmd"]["raw_biased_mmd2"],
                f"{role}/variance_ratio": metrics["moments"]["variance_trace"] / float(np.square(y-y.mean(0)).sum()/len(y))}
            with (self.run_path/"evaluation.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": self.step, **logs}, allow_nan=False)+"\n")
            if not self.disable_wandb:
                wandb.log(evaluation_metrics(role, logs), step=self.step)
            self._event("evaluation_complete", role=role, step=self.step, **logs)
        dist.barrier(group=self.io_group)
        del pipeline, values, local, parts, noise, latent, decoded, feature
        gc.collect()
        torch.cuda.empty_cache()

    @contextmanager
    def _full_state(self, rank0_only=True):
        with FSDP.state_dict_type(self.model.generator, StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only),
                FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only)):
            yield

    def save(self):
        self._event("checkpoint", step=self.step)
        rng = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
               "numpy": np.random.get_state(), "python": random.getstate()}
        rng_states = [None]*self.world_size
        dist.all_gather_object(rng_states, rng, group=self.io_group)
        with self._full_state():
            model_state = self.model.generator.state_dict()
            optimizer_state = FSDP.optim_state_dict(self.model.generator, self.generator_optimizer)
        torch.cuda.synchronize()
        result = [None]
        destination = self.run_path/f"checkpoint_model_{self.step:06d}"
        if self.is_main_process:
            temporary = self.run_path/f".checkpoint_model_{self.step:06d}.{os.getpid()}.tmp"
            try:
                if destination.exists():
                    raise FileExistsError(destination)
                temporary.mkdir()
                payloads = {"model.pt": {"generator": model_state, "step": self.step},
                    "training.pt": {"optimizer": optimizer_state, "step": self.step,
                        "controller": self.controller.state_dict(), "rng_states": rng_states,
                        "contract": self.contract}}
                if self.temporal_early_stop is not None:
                    payloads['training.pt']['temporal_early_stop'] = self.temporal_early_stop.state_dict()
                for filename, payload in payloads.items():
                    with (temporary/filename).open("wb") as stream:
                        torch.save(payload, stream)
                        stream.flush()
                        os.fsync(stream.fileno())
                temporary.rename(destination)
                # Keep optimizer/controller states so every saved checkpoint is resumable.
                result[0] = {"ok": True, "checkpoint": str(destination)}
            except BaseException as error:
                if temporary.exists():
                    shutil.rmtree(temporary)
                result[0] = {"ok": False, "error": repr(error)}
        dist.broadcast_object_list(result, src=0, group=self.io_group)
        if not result[0]["ok"]:
            raise RuntimeError("Checkpoint writer failed: "+result[0]["error"])
        self._event("checkpoint_complete", step=self.step, checkpoint=str(destination))
        del model_state, optimizer_state, rng_states
        gc.collect()

    def _restore(self, directory):
        self._event("restoring", checkpoint=str(directory))
        train_state = None
        packet = [None]
        if self.is_main_process:
            try:
                train_state = torch.load(directory/"training.pt", map_location="cpu", weights_only=False, mmap=True)
                from utils.legacy import canonical_metadata
                train_state['contract'] = canonical_metadata(train_state['contract'])
                train_state['controller'] = canonical_metadata(train_state['controller'])
                audit = validate_resume_contract(train_state['contract'], self.contract, train_state['step'],
                    fork=bool(self.config.get('resume_fork', False)),
                    disable_early_stop=bool(self.config.get('resume_disable_early_stop', False)))
                if len(train_state['rng_states']) != audit['source_world_size']:
                    raise ValueError('Saved RNG state count does not match the source process count')
                packet[0] = {"ok": True, "resume_audit": audit,
                             **{k: v for k, v in train_state.items() if k != "optimizer"}}
            except BaseException as error:
                packet[0] = {"ok": False, "error": repr(error)}
        dist.broadcast_object_list(packet, src=0, group=self.io_group)
        if not packet[0]["ok"]:
            raise RuntimeError(packet[0]["error"])
        model_state = torch.load(directory/"model.pt", map_location="cpu", weights_only=True, mmap=True)
        if model_state["step"] != packet[0]["step"]:
            raise ValueError("Model and optimizer steps differ")
        with self._full_state(rank0_only=False):
            self.model.generator.load_state_dict(model_state["generator"], strict=True)
        shard = FSDP.scatter_full_optim_state_dict(train_state["optimizer"] if self.is_main_process else None,
            self.model.generator, optim=self.generator_optimizer)
        self.generator_optimizer.load_state_dict(shard)
        self.step = self.generator_updates = int(packet[0]["step"])
        self.controller = self._controller_from_state_dict(packet[0]["controller"])
        if self.temporal_early_stop is not None:
            restored_stop = CheckpointOTStopper.from_state_dict(packet[0]['temporal_early_stop'])
            if restored_stop.policy != self.temporal_early_stop.policy:
                raise ValueError('Temporal OT stopping policy changed across resume')
            expected = self.step//restored_stop.policy['checkpoint_every']*restored_stop.policy['checkpoint_every']
            if (restored_stop.history[-1]['step'] if restored_stop.history else 0) != expected:
                raise ValueError('Checkpoint OT history does not match the restored update')
            self.temporal_early_stop = restored_stop
        audit = packet[0]['resume_audit']
        if audit['source_world_size'] == self.world_size:
            rng = packet[0]["rng_states"][self.rank]
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state(rng["cuda"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
        else:
            # Rank-local streams cannot be continued bit-for-bit after changing
            # the process count. Global sample IDs still derive from update index.
            seed = repartition_seed(self.config.seed, self.step, self.world_size, self.rank)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed % (2**32))
            random.seed(seed)
        audit = {**audit, 'checkpoint': str(directory.resolve()),
                 'new_run': str(self.run_path.resolve())}
        if self.is_main_process:
            (self.run_path/'resume_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
        del model_state, train_state, shard, packet
        gc.collect()
        torch.cuda.empty_cache()
        self._event("restored", step=self.step, cca_weight=self.controller.weight,
            optimizer_tensors=len(self.generator_optimizer.state), resume_mode=audit['mode'],
            source_world_size=audit['source_world_size'], target_world_size=self.world_size,
            rng_policy=audit['rng_policy'])

    def _observe_checkpoint_ot(self):
        packet = [None]
        if self.is_main_process:
            try:
                report = json.loads((self.run_path/'evaluation'/f'report_step{self.step:06d}.json').read_text())
                value = float(checkpoint_metric(report, self.temporal_early_stop.policy['metric']))
                if not math.isfinite(value) or value < 0:
                    raise ValueError('Nonfinite/negative checkpoint temporal OT')
                packet[0] = dict(ok=True, value=value)
            except Exception as error:
                packet[0] = dict(ok=False, error=repr(error))
        dist.broadcast_object_list(packet, src=0, group=self.io_group)
        if not packet[0]['ok']:
            raise RuntimeError('Checkpoint OT observation failed: '+packet[0]['error'])
        checkpoint = self.run_path/f'checkpoint_model_{self.step:06d}/model.pt'
        stopped = self.temporal_early_stop.observe(self.step, packet[0]['value'], checkpoint)
        self._event('checkpoint_ot', step=self.step, metric=self.temporal_early_stop.policy['metric'],
                    value=packet[0]['value'], consecutive_increases=self.temporal_early_stop.increases,
                    stopping=stopped)
        if self.is_main_process and not self.disable_wandb:
            wandb.log({'early_stop/consecutive_increases': self.temporal_early_stop.increases}, step=self.step)
        return stopped

    def _write_training_selection(self):
        if self.temporal_early_stop is None:
            return
        selected = self.temporal_early_stop.selection(
            self.step, self.max_steps, self.run_path/f'checkpoint_model_{self.step:06d}/model.pt')
        if self.is_main_process:
            path = self.run_path/'training_selection.json'
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(selected, indent=2, allow_nan=False)+'\n')
            temporary.replace(path)
            if not self.disable_wandb:
                wandb.log({'early_stop/selected_step': selected['selected_step']}, step=self.step)
        self._event('training_selection', step=self.step, selected_step=selected['selected_step'],
                    reason=selected['reason'])

    def train(self):
        if self.step == 0:
            self._evaluate("control")
            self._evaluate("report")
        while self.step < self.max_steps and not (self.temporal_early_stop and self.temporal_early_stop.stopped):
            target = self.step+1
            lr = configured_learning_rate(target, self.config, total=self.max_steps)
            for group in self.generator_optimizer.param_groups:
                group["lr"] = lr
            torch.manual_seed(int(self.config.seed)+1000003*target+self.rank)
            self._event("training", step=target, lr=lr, cca_weight=self.controller.weight)
            logs = self._generator_update()
            self.step = target
            if self.pending_batch:
                raise AssertionError("Training batch was not fully consumed")
            if any(p.dtype != torch.float32 for p in self.model.generator.parameters()):
                raise TypeError("Generator master parameters must remain FP32")
            if any(state[key].dtype != torch.float32 for state in self.generator_optimizer.state.values()
                   for key in ("exp_avg", "exp_avg_sq") if key in state):
                raise TypeError("Adam moments must remain FP32")
            numeric = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            numeric.update(lr=lr, step=self.step, cca_weight_used=self.controller.weight)
            if self.is_main_process:
                with (self.run_path/"metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(numeric, allow_nan=False)+"\n")
                if not self.disable_wandb:
                    wandb.log(self._training_metrics({**logs, 'lr': lr}), step=self.step)
            if self.step % int(self.config.control_every) == 0 or self.step == self.max_steps:
                self._evaluate("control")
            checkpoint_due = self.step % int(self.config.log_iters) == 0 or self.step == self.max_steps
            early_check_due = (self.temporal_early_stop is not None and
                               self.step % self.temporal_early_stop.policy['checkpoint_every'] == 0)
            if self.step % int(self.config.preview_every) == 0 or self.step == self.max_steps or early_check_due:
                self._evaluate("report")
            if early_check_due:
                self._observe_checkpoint_ot()
            if checkpoint_due:
                self.save()
        self._write_training_selection()
        self._event("complete", step=self.step, cca_weight=self.controller.weight)
