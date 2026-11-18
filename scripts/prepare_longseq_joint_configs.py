"""Scale the six recorded joint schedules and check their exact token budgets."""

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from torchtitan_dynamic_seq_len.schedulers.batch_size import BatchSizeSchedulerConfig
from torchtitan_dynamic_seq_len.schedulers.sequence_length import SeqLenSchedulerConfig

NAMES = {
    'b857ac04616f4ed6a355b556e7d5c62a': 'bs_increasing_seq_decreasing',
    'c1dc9dfe632a47dc8cd937d237aa9a1c': 'bs_decreasing_seq_increasing',
    '6498f0a596184b32a652494b19526482': 'bs_cyclic_seq_inverse',
    '990e22f236454203a790973c27885bda': 'linear_bs_cyclic_seq_cosine_lr',
    '32e2ba93e7c64d58838cfa8a8911f1ec': 'linear_bs_cyclic_seq_constant_lr',
    'cfba39d1e5af4c7ea71ac16a3c0414ec': 'four_stage_cycle',
}


def compressed_schedule(values):
    anchors = {1, len(values)}
    for index in range(1, len(values)):
        if values[index] != values[index - 1]:
            anchors.update((index, index + 1))
    anchors = sorted(anchors)
    return {'kind': 'piecewise_linear', 'anchor_steps': anchors,
            'batch_sizes': [values[s - 1] for s in anchors], 'round_to_multiple': 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dp', type=int, default=4)
    parser.add_argument('--microbatch-tokens', type=int, default=49152)
    parser.add_argument('--cluster', default='tinkercliffs')
    parser.add_argument('--partition', default='a100_normal_q')
    parser.add_argument('--qos', default='tc_a100_normal_short')
    args = parser.parse_args()
    records = json.loads((ROOT / 'outputs/longseq_joint/finweb-llama-100m-seqbs-joint-scheduler.json').read_text())
    destination = ROOT / 'local/generated_configs/longseq_joint'
    destination.mkdir(parents=True, exist_ok=True)
    reports = []
    for record in records:
        name = NAMES[record['id']]
        ref = record['config']
        seq = copy.deepcopy(ref['seq_len_scheduler'])
        for key in ['length', 'min_length', 'max_length']:
            seq[key] *= 6
        seq['lengths'] = [x * 6 for x in seq['lengths']]
        if seq.get('round_to_multiple'):
            seq['round_to_multiple'] *= 6
        batch = copy.deepcopy(ref['batch_size_scheduler'])
        steps = ref['training']['steps']
        lengths = SeqLenSchedulerConfig(**seq).build(steps)
        batches = BatchSizeSchedulerConfig(**batch).build(steps)
        old_lengths = SeqLenSchedulerConfig(**ref['seq_len_scheduler']).build(steps)
        rows = []
        for step in range(1, steps + 1):
            length, global_batch = lengths.length_for_step(step), batches.value_for_step(step)
            assert length == 6 * old_lengths.length_for_step(step)
            assert global_batch % args.dp == 0 and length % 16 == 0
            limit = min(1024, args.microbatch_tokens // length, global_batch // args.dp)
            micro = max(n for n in range(1, limit + 1) if global_batch % (args.dp * n) == 0)
            rows.append({'step': step, 'seq_len': length, 'global_batch': global_batch,
                         'microbatch': micro, 'accumulation': global_batch // (args.dp * micro),
                         'tokens': global_batch * length})
        budget = sum(row['tokens'] for row in rows)
        assert budget == 12079595520, (name, budget)
        fixed = {
            'hf_assets_path': '${ASSETS_DIR}/hf/llama3-tokenizer',
            'dataloader.dataset': 'fineweb_edu_long',
            'dataloader.dataset_path': '/projects/jason_optai/kangan/datasets/fineweb-edu-long-30b',
            'dataloader.split': 'train', 'dataloader.infinite': True,
            'dataloader.expected_manifest_sha256': '4504314f51733ce085991de14b11ef595424d1a006b1ebb2475e4fc3d29579db',
            'dataloader.num_workers': 0, 'dataloader.prefetch_factor': None,
            'dataloader.persistent_workers': False, 'dataloader.pin_memory': True,
            'training.steps': steps, 'training.seq_len': 12288,
            'training.local_batch_size': rows[0]['microbatch'],
            'training.global_batch_size': rows[0]['global_batch'],
            'training.dtype': 'float32', 'training.mixed_precision_param': 'bfloat16',
            'training.mixed_precision_reduce': 'float32', 'training.max_norm': 1.0,
            'loss.num_chunks': 16, 'compile.enable': True,
            'checkpoint.enable': True, 'checkpoint.interval': 200,
            'checkpoint.keep_latest_k': 2, 'checkpoint.last_save_model_only': False,
            'checkpoint.folder': 'checkpoint', 'debug.seed': 1, 'debug.print_config': True,
            'metrics.log_freq': 1, 'metrics.disable_color_printing': True,
            'validator.enable': True, 'validator.freq': 20,
            'validator.eval_seq_len': 12288, 'validator.eval_local_batch_size': 4,
            'validator.token_budget': 1704 * 12288, 'validator.artifact_path': '',
            'validator.dataloader.dataset': 'fineweb_edu_long',
            'validator.dataloader.dataset_path': '/projects/jason_optai/kangan/datasets/fineweb-edu-long-30b',
            'validator.dataloader.split': 'validation',
            'validator.dataloader.expected_manifest_sha256': '4504314f51733ce085991de14b11ef595424d1a006b1ebb2475e4fc3d29579db',
            'comm.train_timeout_seconds': 1200,
        }
        for section in ['optimizer', 'lr_scheduler']:
            for key, value in ref[section].items():
                # Copy scalar options supported by the existing config.
                if not isinstance(value, (dict, list)):
                    fixed[f'{section}.{key}'] = value
        for section, values in [('seq_len_scheduler', seq), ('batch_size_scheduler', batch),
                                ('microbatch_scheduler', compressed_schedule([r['microbatch'] for r in rows]))]:
            fixed.update({f'{section}.{key}': value for key, value in values.items()})
        config = {
            'name': 'longseq-llama3-103m-' + name.replace('_', '-') + '-3840steps',
            'torchtitan_config': 'llama3_100m_long_documents_dynamic_seq_bs',
            'paths': {'dump_root': '/projects/jason_optai/kangan/experiments/longseq_joint/runs',
                      'manifest_root': '${OUTPUT_ROOT}/longseq_joint/experiments',
                      'slurm_log_root': '${OUTPUT_ROOT}/longseq_joint/slurm_logs',
                      'hf_home': '/projects/jason_optai/kangan/huggingface',
                      'torch_home': '/scratch/kangan/longseq_joint/cache/torch',
                      'validation_dir': '${ASSETS_DIR}/data/longseq_validation_llama3_12288'},
            'resources': {'account': 'jason_optai-paid', 'cluster': args.cluster,
                          'partition': args.partition, 'qos': args.qos,
                          'nodes': 1, 'gpus_per_node': args.dp, 'cpus_per_task': 16,
                          'mem': '64G', 'time': '12:00:00', 'max_concurrent': 1},
            'wandb': {'enabled': True, 'entity': 'dynamic_seq_length',
                      'project': 'longseq-finwebedu-llama-100m-seqbs-joint-scheduler',
                      'group': 'native-llama3-103m-12k-12b', 'mode': 'online'},
            'fixed': fixed, 'sweep': {},
        }
        path = destination / f'{name}.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        with (ROOT / f'outputs/longseq_joint/{name}_schedule.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        reports.append({'name': name, 'reference_url': record['url'], 'config': str(path),
                        'tokens': budget, 'steps': steps, 'minimum_length': min(r['seq_len'] for r in rows),
                        'maximum_length': max(r['seq_len'] for r in rows), 'dp': args.dp,
                        'max_microbatch_tokens': max(r['seq_len'] * r['microbatch'] for r in rows)})
    (ROOT / 'outputs/longseq_joint/schedule_plan.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
