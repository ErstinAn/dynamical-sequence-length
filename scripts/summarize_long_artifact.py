"""Report raw document lengths and usable labels for fixed training windows."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_if_changed(path, text):
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def render_report(root, result, selection, verified, smoke=None, receipt=None):
    """Render the verified statistics without changing dataset files."""
    stats = result['splits']
    report = '''# fineweb-edu-long-30b statistics

[Overview](README.md) · [Source statistics](fineweb-edu.md)

This dataset contains complete text records selected from FineWeb-Edu default.
The builder retained each selected record without truncation or document concatenation.
Lengths use the original Llama 3 tokenizer, without added BOS or EOS tokens.
BOS and EOS are the tokens for the start and end of a document.

## Document lengths

'''
    report += f"The minimum selection length is {selection['minimum_llama3_tokens']:,} raw Llama 3 tokens.\n"
    report += 'Percentiles use linear interpolation and can have fractional values.\n\n'
    report += '| Metric | Train | Validation |\n| --- | ---: | ---: |\n'
    for title, key in [('Documents', 'documents'), ('Raw tokens', 'tokens'),
                       ('Minimum', 'minimum'), ('Maximum', 'maximum')]:
        report += f"| {title} | {stats['train'][key]:,} | {stats['validation'][key]:,} |\n"
    report += f"| Mean | {stats['train']['mean']:,.2f} | {stats['validation']['mean']:,.2f} |\n"
    for title, key in [('Median', '0.5'), ('P90', '0.9'), ('P95', '0.95'), ('P99', '0.99')]:
        report += f"| {title} | {stats['train']['quantiles'][key]:,.2f} | {stats['validation']['quantiles'][key]:,.2f} |\n"
    report += '''
## Labels from fixed windows

This table describes one complete pass through train at each fixed window length.
The reader keeps each window inside one document and discards incomplete tails.
A document with N raw tokens supplies N+1 labels after BOS/EOS insertion and the next-token shift.
A window length L retains `floor((N+1)/L) * L` labels from that document.
The retention percentage uses all N+1 labels as its denominator.

| Window length | Documents with a full window | Labels per pass | Labels retained |
| ---: | ---: | ---: | ---: |
'''
    for row in result['fixed_windows']:
        report += f"| {row['window']:,} | {row['documents_with_window']:,} | {row['label_tokens_per_pass']:,} | {row['fraction_of_available_labels']:.2%} |\n"
    report += '''
These rows describe separate passes. Their totals must not be added.
A changing schedule can discard different tails as its length changes.
The fixed-window table therefore does not predict the token budget for a changing schedule.
Compare experiments with the actual label count that contributes to loss.

## Selection limits and checks

'''
    report += f"The candidate snapshot contained {selection['input_documents']:,} records.\n"
    report += f"The builder removed {selection['duplicate_id_or_normalized_text_removed']:,} records with duplicate IDs or normalized text.\n"
    report += '''Train and validation have no canonical URL groups in common.
The process did not remove near-duplicate text across different URLs.

The Llama 3 candidate scan stopped after it supplied the required budget.
The selected threshold is the highest feasible tested threshold within that candidate pool.
It does not establish the highest feasible threshold across all default documents.
A complete text record does not establish complete source pages, intact formatting, or content quality.

'''
    report += "All file hashes, text hashes, token bounds, and document boundaries passed the dataset checks.\n"
    report += f"Retokenization checks matched token IDs for {verified['retokenized_documents']:,} documents from shard boundaries.\n"
    if smoke is not None:
        if not smoke['passed'] or smoke['manifest_sha256'] != result['manifest_sha256']:
            raise ValueError('Training receipt does not match this verified dataset')
        report += f"A 1B-class model completed {len(smoke['train_steps'])} training steps at length {smoke['sequence_length']:,}.\n"
        report += 'Training loss, gradient norms, and validation loss were finite. This checks integration, not convergence.\n'
    else:
        report += 'The dataset has no final training receipt.\n'
    report += '\n## Files and sharing\n\n'
    report += f"Compressed text files occupy {result['text_bytes']/1e9:.2f} GB.\n"
    report += f"The local uint32 token cache occupies {result['token_cache_bytes']/1e9:.2f} GB, excluding indexes.\n"
    report += 'One GB is 1,000,000,000 bytes. The shared version contains text and a script to rebuild the cache.\n\n'
    if receipt is not None:
        if not receipt['private'] or receipt['manifest_sha256'] != result['manifest_sha256']:
            raise ValueError('Upload receipt does not match this verified private dataset')
        url = f"https://huggingface.co/datasets/{receipt['repo_id']}/tree/{receipt['revision']}"
        report += f"[Private dataset at the checked revision]({url}).\n"
        report += 'The upload receipt records successful remote file checks and streaming reads from both splits.\n\n'
    else:
        report += 'The dataset has no private upload receipt.\n\n'
    report += f"Local artifact: `{root}`.\n\n"
    report += f"Manifest SHA-256: `{result['manifest_sha256']}`.\n\n"
    report += f"[Exact statistics]({root}/length_statistics.json) · [Construction and evidence](appendix/construction.md).\n"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset
    manifest = json.loads((root/'manifest.json').read_text())
    verified = json.loads((root/'verification.json').read_text())
    if not verified['passed'] or verified['manifest_sha256'] != digest(root/'manifest.json'):
        raise ValueError('The dataset must pass verification before length reporting')
    lengths = {}
    stats = {}
    for split in ['train','validation']:
        parts = []
        for entry in manifest['files']:
            if entry['split'] != split:
                continue
            path = root/entry['index']['path']
            if digest(path) != entry['index']['sha256']:
                raise ValueError('Length index checksum mismatch')
            parts.append(np.load(path)[:,1].astype(np.int64))
        values = np.concatenate(parts)
        lengths[split] = values
        expected = manifest['selection']['splits'][split]
        if len(values) != expected['documents'] or int(values.sum()) != expected['tokens']:
            raise ValueError('Length report disagrees with the verified split budget')
        stats[split] = {'documents':len(values),'tokens':int(values.sum()),
                        'minimum':int(values.min()),'maximum':int(values.max()),
                        'mean':float(values.mean()),
                        'quantiles':{str(q):float(np.quantile(values,q)) for q in [0.5,0.9,0.95,0.99]}}
    windows = []
    available_labels = lengths['train'] + 1
    for window in [4096,8192,12288,16384,24576,32768,65536]:
        full_windows = available_labels // window
        labels = int(full_windows.sum()) * window
        windows.append({'window':window,'documents_with_window':int(np.count_nonzero(full_windows)),
                        'label_tokens_per_pass':labels,
                        'fraction_of_available_labels':labels/int(available_labels.sum())})
    result = {'manifest_sha256':verified['manifest_sha256'],'splits':stats,'fixed_windows':windows,
              'text_bytes':sum(e['text']['bytes'] for e in manifest['files']),
              'token_cache_bytes':sum(e['token_ids']['bytes'] for e in manifest['files']),
              'formula':'A document with N raw tokens provides N+1 labels after BOS/EOS. A fixed window L retains floor((N+1)/L)*L labels. This table does not predict a changing scheduler.'}
    write_if_changed(root/'length_statistics.json',json.dumps(result,indent=2)+'\n')
    table = '| Window | Documents with a full window | Label tokens per pass | Labels retained |\n| ---: | ---: | ---: | ---: |\n'
    for row in windows:
        table += f"| {row['window']:,} | {row['documents_with_window']:,} | {row['label_tokens_per_pass']:,} | {row['fraction_of_available_labels']:.2%} |\n"
    block = '\n<!-- BEGIN LONG LENGTH STATISTICS -->\n\n## Fixed training windows\n\n'
    block += 'The table describes one complete pass through train with a fixed window. Each window stays within one document. Short tails do not contribute labels.\n\n'
    block += table+'\nCount actual loss labels when you compare sequence-length schedules. The stored raw-token budget does not equal the labels from one pass.\n'
    block += '<!-- END LONG LENGTH STATISTICS -->\n'
    card_path = root/'README.md'
    card = card_path.read_text().split('<!-- BEGIN LONG LENGTH STATISTICS -->')[0].rstrip()+'\n'+block
    write_if_changed(card_path,card)
    smoke_path, receipt_path = root/'training_smoke.json', root/'upload_receipt.json'
    smoke = json.loads(smoke_path.read_text()) if smoke_path.exists() else None
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    report = render_report(root, result, manifest['selection'], verified, smoke, receipt)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_if_changed(args.report, report)
    print(json.dumps({'event':'length_report','splits':stats,'text_bytes':result['text_bytes']}),flush=True)


if __name__ == '__main__':
    main()
