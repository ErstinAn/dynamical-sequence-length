# Construction and evidence

[Overview](../README.md) · [Source statistics](../fineweb-edu.md) · [Long-document statistics](../fineweb-edu-long-30b.md)

This appendix describes the completed build and its evidence.
The statistics reports contain the final document counts and token totals.

## Fixed inputs

| Input | Identity |
| --- | --- |
| Source | `HuggingFaceFW/fineweb-edu`, configuration `default` |
| Source revision | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| Source files | 2,410 Parquet files, 4,522,727,684,984 bytes |
| Tokenizer | Original `meta-llama/Meta-Llama-3-8B` tokenizer |
| Tokenizer revision | `8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920` |
| Train target | At least 30B raw Llama 3 tokens |
| Validation target | At least 10M raw Llama 3 tokens |
| Special tokens | The selection budget excludes BOS, EOS, and chat templates. |

## Completed process

1. The scan read `token_count` from every default source file.
   It saved lengths in source row order, histograms, and file checksums.
2. The first extraction selected records with at least 16,384 source tokens and at least 16,384 Llama 3 tokens.
   Its candidate target was 60B Llama 3 tokens.
3. The supplementary extraction selected source lengths in `[12,288, 16,384)` and Llama 3 lengths of at least 12,288.
   Its candidate target was 16B Llama 3 tokens.
4. The selector used a fixed snapshot of completed candidate shards.
   It removed duplicate IDs and duplicate text after Unicode NFC and whitespace normalization.
5. A fixed hash assigned approximately 0.2 percent of canonical URL groups to validation.
   The remaining groups supplied train. This rule does not guarantee an exact token fraction.
6. The selector chose the highest tested threshold that met both token targets within the candidate pool.
   It selected train documents by text hash and retained the final complete document when the total exceeded the target.
7. The writer produced text Parquet files, source references, token counts, content hashes, and split labels.
   It also produced uint32 token files and document boundary indexes.
8. The checker examined all file hashes, text hashes, boundaries, and token ranges.
   It retokenized the first and last documents in each output shard.
9. The final training check used a 1B-class model and the selected window length.
   The uploader then checked remote files, private access, and streaming reads from both splits.

The tested thresholds were 32,768, 28,672, 24,576, 20,480, 16,384, and 12,288.
The candidate scan stopped after the verified selection met the targets.
It did not retokenize every default document.
Records with source length at least 16,384 and Llama 3 length below 16,384 were absent from both candidate groups.
Thus, the candidate pool is not a complete Llama 3 length distribution for default.

The process did not check near-duplicate text across different URLs.
Canonical URL separation does not establish the absence of all semantic overlap.
The selected minimum length is a filter, not a truncation length.
Raw document lengths need not be multiples of eight.
Training windows must be multiples of eight under the current reader.

## Recorded resources

These are historical measurements, not current queue estimates or resource recommendations.
The jobs used account `jason_optai-paid`.

| Operation | Recorded resources or outcome |
| --- | --- |
| 100BT scan, job 8405368 | 140 files, 16 CPU cores, 32 readers, 5 minutes 35 seconds, peak process memory 880.69 MiB |
| Default scan, job 8439938 | 2,410 files, 4 CPU nodes, 64 CPU cores, 34 minutes 29 seconds, exit code 0 |
| Candidate extraction | Each extraction used 16 CPU nodes, 64 workers, and 4 cores per worker. |
| Output and checks | The plan allowed up to 32 processes and approximately 128M tokens per output shard. |
| Example extraction, jobs 8439266 and 8439315 | 4 CPU cores each, 43 and 59 seconds, both successful |
| Mathematics/programming check, job 8439617 | Successful, 5 seconds, excluding candidate selection and human review |

The 100BT document count matched the Dataset Viewer count recorded during the scan.
Bucket totals matched dataset totals. Three source files also matched the earlier test.
All 2,410 default length indexes passed the archive checksum check.

## Files and reproducibility

Repository paths below are relative to the project root.

| Purpose | Location |
| --- | --- |
| 100BT results and indexes | `outputs/fineweb_edu_profile/full-100bt/` |
| Default results | `outputs/fineweb_edu_long/full-default-profile/` |
| Default archive | `/projects/jason_optai/kangan/datasets/fineweb-edu-default-profile-87f09149/` |
| Intermediate files | `/scratch/kangan/fineweb-edu-long-87f09149/` |
| Final artifact | `/projects/jason_optai/kangan/datasets/fineweb-edu-long-30b/` |
| Build logs and job records | `outputs/fineweb_edu_long/` |

The final artifact contains `manifest.json`, `verification.json`, `length_statistics.json`, `training_smoke.json`, and `upload_receipt.json`.
The default archive contains `archive_verification.json`.
These records retain exact counts, hashes, and the published revision.
Temporary supply counts do not replace the final selection record.

| Tool | Function |
| --- | --- |
| [profile_fineweb_edu_lengths.py](../../../scripts/profile_fineweb_edu_lengths.py) | Scan source lengths. |
| [profile_fineweb_default.py](../../../scripts/profile_fineweb_default.py) | Coordinate the default scan. |
| [build_fineweb_long.py](../../../scripts/build_fineweb_long.py) | Select, write, check, and upload the artifact. |
| [rebuild_long_token_cache.py](../../../scripts/rebuild_long_token_cache.py) | Rebuild token files from shared text. |
| [summarize_long_artifact.py](../../../scripts/summarize_long_artifact.py) | Calculate lengths and generate the long-document report. |
| [long_documents.py](../../../torchtitan_dynamic_seq_len/data/long_documents.py) | Read windows within documents. |
| [Final training check](../../../legacy/configs/examples/llama3_1b_long_documents_smoke.yaml) | Check the final artifact with a 1B-class model. |

The shared artifact contains text, metadata, checks, and a cache reconstruction script.
Text permits another tokenizer and direct content inspection.
The local token cache avoids repeated encoding but fixes the tokenizer.
The upload receipt records the private organization repository and commit.
This appendix does not specify current account storage limits.
