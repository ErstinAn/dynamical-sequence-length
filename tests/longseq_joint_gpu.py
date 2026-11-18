"""Check real long-document windows, joint shape changes, and fixed validation."""

import argparse
import json
import math
import os
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
from torchtitan.config import ConfigManager
from torchtitan.tools.logging import init_logger

from tests.loss_grad_gpu import CaptureLogger
from torchtitan_dynamic_seq_len.run_config import resolve_point


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw, _, _, options = resolve_point(args.config, 0)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ['DYNAMIC_SEQ_LEN_VALIDATION_DIR'] = raw['paths']['validation_dir']
    cfg = ConfigManager().parse_args(['--module', 'torchtitan_dynamic_seq_len.config_registry',
                                     '--config', raw['torchtitan_config'], *options])
    cfg.dump_folder = str(Path('/scratch/kangan/longseq_joint/debug') / args.output.name)
    cfg.metrics.enable_wandb = True
    # torchrun gives every rank the same rendezvous ID.
    run_id = os.environ.get('LONGSEQ_DEBUG_RUN_ID') or os.environ.get('TORCHELASTIC_RUN_ID')
    if not run_id:
        if int(os.environ.get('WORLD_SIZE', '1')) > 1:
            raise ValueError('Distributed debug requires a shared run ID')
        run_id = uuid.uuid4().hex
    os.environ.update(WANDB_PROJECT='dynamic-seq-bs-debug', WANDB_TEAM='dynamic_seq_length',
                      WANDB_ENTITY='dynamic_seq_length', WANDB_RUN_ID=run_id,
                      WANDB_RUN_NAME=raw['name'], WANDB_RESUME='never', WANDB_MODE='online')
    init_logger()
    trainer = cfg.build()
    capture = CaptureLogger(trainer.metrics_processor.logger, args.output / 'metrics.json')
    trainer.metrics_processor.logger = capture
    try:
        trainer.train()
        if dist.get_rank() == 0:
            rows = [r for r in capture.rows if 'loss_metrics/global_avg_loss' in r]
            assert [r['step'] for r in rows] == list(range(1, cfg.training.steps + 1))
            expected_tokens = 0
            for row in rows:
                step = row['step']
                length = trainer.seq_len_scheduler.length_for_step(step)
                batch = trainer.batch_size_scheduler.value_for_step(step)
                expected_tokens += batch * length
                assert row['seq_len'] == length and row['global_batch_size'] == batch
                assert row['n_valid_tokens_seen'] == expected_tokens
                assert all(math.isfinite(row[k]) for k in ['loss_metrics/global_avg_loss', 'grad_norm'])
                assert row['memory/num_ooms'] == 0
            validation = [r for r in capture.rows if 'validation_metrics/loss' in r]
            assert validation and all(math.isfinite(r['validation_metrics/loss']) for r in validation)
            result = {'passed': True, 'steps': len(rows), 'tokens': expected_tokens,
                      'max_memory_gib': max(r['memory/max_active(GiB)'] for r in rows),
                      'gpu': torch.cuda.get_device_name(), 'validation': trainer.validator.last_result,
                      'wandb_url': f'https://wandb.ai/dynamic_seq_length/dynamic-seq-bs-debug/runs/{run_id}'}
            (args.output / 'result.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
    finally:
        trainer.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
