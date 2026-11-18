"""Check new candidate files and freeze a sufficient training corpus."""
import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import build_fineweb_long as pipeline
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--supplement', type=Path, required=True)
    args = parser.parse_args()
    roots = [args.root.resolve(), args.supplement.resolve()]
    plans = [pipeline.load_plan(root) for root in roots]
    folder = roots[0] / 'supply-watch'
    folder.mkdir(exist_ok=True)
    target = plans[0]['train_token_target']
    validation_minimum = plans[0]['validation_minimum_tokens']
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    processed = set()
    input_tokens = input_documents = 0
    with tempfile.TemporaryDirectory(prefix='fineweb-supply-', dir=os.environ.get('TMPDIR')) as temporary:
        db = sqlite3.connect(Path(temporary) / 'supply.sqlite')
        db.execute('PRAGMA journal_mode=OFF')
        db.execute('PRAGMA synchronous=OFF')
        db.execute('PRAGMA cache_size=-1048576')
        db.execute('CREATE TABLE docs(id TEXT UNIQUE, hash TEXT UNIQUE, tokens INTEGER, split TEXT)')
        while True:
            added = 0
            for root, plan in zip(roots, plans):
                for record_path in sorted((root / 'candidates').glob('*.json')):
                    key = str(record_path)
                    if key in processed:
                        continue
                    record = json.loads(record_path.read_text())
                    if record['plan_sha256'] != plan['plan_sha256']:
                        raise ValueError('Candidate plan changed during supply monitoring')
                    for batch in pq.ParquetFile(root / record['file']).iter_batches(batch_size=4096,
                            columns=['id','normalized_sha256','llama3_tokens','url_group'], use_threads=False):
                        rows = []
                        for doc in batch.to_pylist():
                            is_val = int(hashlib.sha256(('validation-v1:'+doc['url_group']).encode()).hexdigest()[:16],16) % plan['validation_hash_modulus'] < plan['validation_hash_cutoff']
                            rows.append((doc['id'],doc['normalized_sha256'],doc['llama3_tokens'],
                                         'validation' if is_val else 'train'))
                        db.executemany('INSERT OR IGNORE INTO docs VALUES(?,?,?,?)',rows)
                    db.commit()
                    processed.add(key)
                    added += 1
                    input_tokens += record['llama3_tokens']
                    input_documents += record['documents']
                    if added % 100 == 0:
                        print(json.dumps({'event':'supply_scan','files':len(processed)}),flush=True)
            if added:
                thresholds = []
                sufficient = False
                for threshold in pipeline.THRESHOLDS:
                    counts = {split:{'documents':n,'tokens':tokens or 0} for split,n,tokens in db.execute(
                        'SELECT split,COUNT(*),SUM(tokens) FROM docs WHERE tokens>=? GROUP BY split',(threshold,))}
                    thresholds.append({'minimum':threshold,'splits':counts})
                    # Completion order can choose a different representative of
                    # duplicate text. Require margin, then run the exact selector.
                    if counts.get('train',{}).get('tokens',0) >= target+200_000_000 and counts.get('validation',{}).get('tokens',0) >= validation_minimum+2_000_000:
                        sufficient = True
                status = {'updated_at':time.time(),'files':len(processed),'input_tokens':input_tokens,
                          'input_documents':input_documents,'provisional_thresholds':thresholds,
                          'note':'Duplicate representatives follow completion order. Only the exact frozen selection can authorize stopping extraction.'}
                pipeline.write_json(folder/'status.json',status)
                print(json.dumps(status),flush=True)
                if sufficient:
                    snapshot = roots[0] / f'snapshot-{len(processed):05d}-{time.time_ns()}'
                    selection_args = SimpleNamespace(root=snapshot,candidate_root=roots,snapshot=True)
                    pipeline.combine_stage(selection_args)
                    try:
                        pipeline.select_stage(selection_args)
                    except ValueError:
                        if not (snapshot/'insufficient_supply.json').exists():
                            raise
                        print(json.dumps({'event':'snapshot_needs_more_supply','root':str(snapshot)}),flush=True)
                    else:
                        selected = json.loads((snapshot/'selection.json').read_text())
                        receipt = {'selection_root':str(snapshot),'minimum_llama3_tokens':selected['minimum_llama3_tokens'],
                                   'splits':selected['splits'],'selection_sha256':pipeline.sha256(snapshot/'selection.json'),
                                   'plan_sha256':selected['plan_sha256']}
                        pipeline.write_json(folder/'ready.json',receipt)
                        print(json.dumps({'event':'supply_ready',**receipt}),flush=True)
                        return
            # Stop the monitor if the normal full-pool finalizer already selected
            # enough data. That job can continue without a competing snapshot.
            if (roots[0]/'combined-12k/selection.json').exists():
                print(json.dumps({'event':'full_pool_selection_exists'}),flush=True)
                return
            time.sleep(30)


if __name__ == '__main__':
    main()
