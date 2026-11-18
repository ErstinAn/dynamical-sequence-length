"""Freeze document-local validation windows from the verified long corpus."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torchtitan_dynamic_seq_len.data.long_documents import LongDocumentDataset


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def prepare(dataset, output, tokenizer, length=12288, count=1704):
    output.mkdir(parents=True, exist_ok=False)
    source = LongDocumentDataset(dataset, split='validation', seq_len=length)
    if digest(tokenizer / 'tokenizer.json') != source.tokenizer_sha256:
        raise ValueError('Tokenizer differs from the source corpus')
    arrays = {name: np.lib.format.open_memmap(output / f'{name}.npy', mode='w+',
              dtype=np.int64, shape=(count, length)) for name in ['input', 'labels', 'positions']}
    iterator = iter(source)
    documents = []
    for row in range(count):
        inputs, labels = next(iterator)
        arrays['input'][row] = inputs['input'].numpy()
        arrays['positions'][row] = inputs['positions'].numpy()
        arrays['labels'][row] = labels.numpy()
        documents.append({'file': source.files[source.file_idx]['text']['path'],
                          'row': source.row_idx, 'raw_window_offset': source.offset - length})
    for array in arrays.values():
        array.flush()
    (output / 'documents.json').write_text(json.dumps(documents, indent=2) + '\n')
    manifest = {
        'schema_version': 1, 'sequence_length': length, 'num_sequences': count,
        'token_budget': count * length,
        'source': {'repo_id': 'long-sequence-length-dataset/fineweb-edu-long-30b',
                   'split': 'validation', 'manifest_sha256': source.manifest_hash},
        'tokenizer_files': {name: digest(tokenizer / name)
                            for name in ['tokenizer.json', 'tokenizer_config.json']},
        'arrays': {name: {'file': f'{name}.npy', 'sha256': digest(output / f'{name}.npy')}
                   for name in arrays},
        'evaluation_documents': {'file': 'documents.json', 'sha256': digest(output / 'documents.json')},
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'sequences': count, 'tokens': count * length}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--length', type=int, default=12288)
    parser.add_argument('--count', type=int, default=1704)
    parser.add_argument('--packing', choices=['none', 'stream'], default='none')
    parser.add_argument('--sample-seed', type=int, default=None,
                        help='Select packed sequences once from the complete validation pool')
    args = parser.parse_args()
    if args.packing == "stream":
        from prepare_packed_validation import prepare_packed
        prepare_packed(args.dataset, args.output, args.tokenizer, args.length, args.count,
                       sample_seed=args.sample_seed)
    else:
        if args.sample_seed is not None:
            parser.error('--sample-seed requires --packing stream')
        prepare(args.dataset, args.output, args.tokenizer, args.length, args.count)
