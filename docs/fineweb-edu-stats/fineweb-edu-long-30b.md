# fineweb-edu-long-30b statistics

[Overview](README.md) · [Source statistics](fineweb-edu.md)

This dataset contains complete text records selected from FineWeb-Edu default.
The builder retained each selected record without truncation or document concatenation.
Lengths use the original Llama 3 tokenizer, without added BOS or EOS tokens.
BOS and EOS are the tokens for the start and end of a document.

## Document lengths

The minimum selection length is 12,288 raw Llama 3 tokens.
Percentiles use linear interpolation and can have fractional values.

| Metric | Train | Validation |
| --- | ---: | ---: |
| Documents | 1,348,067 | 2,725 |
| Raw tokens | 30,000,004,070 | 62,360,433 |
| Minimum | 12,288 | 12,290 |
| Maximum | 216,687 | 152,566 |
| Mean | 22,254.09 | 22,884.56 |
| Median | 18,084.00 | 18,522.00 |
| P90 | 33,974.00 | 35,222.00 |
| P95 | 46,035.70 | 50,794.00 |
| P99 | 93,898.70 | 93,804.64 |

## Labels from fixed windows

This table describes one complete pass through train at each fixed window length.
The reader keeps each window inside one document and discards incomplete tails.
A document with N raw tokens supplies N+1 labels after BOS/EOS insertion and the next-token shift.
A window length L retains `floor((N+1)/L) * L` labels from that document.
The retention percentage uses all N+1 labels as its denominator.

| Window length | Documents with a full window | Labels per pass | Labels retained |
| ---: | ---: | ---: | ---: |
| 4,096 | 1,348,067 | 27,587,948,544 | 91.96% |
| 8,192 | 1,348,067 | 24,298,799,104 | 80.99% |
| 12,288 | 1,348,067 | 23,663,136,768 | 78.87% |
| 16,384 | 853,433 | 18,512,674,816 | 61.71% |
| 24,576 | 306,110 | 9,944,629,248 | 33.15% |
| 32,768 | 146,873 | 6,311,608,320 | 21.04% |
| 65,536 | 32,718 | 2,232,942,592 | 7.44% |

These rows describe separate passes. Their totals must not be added.
A changing schedule can discard different tails as its length changes.
The fixed-window table therefore does not predict the token budget for a changing schedule.
Compare experiments with the actual label count that contributes to loss.

## Selection limits and checks

The candidate snapshot contained 3,119,411 records.
The builder removed 1,752,095 records with duplicate IDs or normalized text.
Train and validation have no canonical URL groups in common.
The process did not remove near-duplicate text across different URLs.

The Llama 3 candidate scan stopped after it supplied the required budget.
The selected threshold is the highest feasible tested threshold within that candidate pool.
It does not establish the highest feasible threshold across all default documents.
A complete text record does not establish complete source pages, intact formatting, or content quality.

All file hashes, text hashes, token bounds, and document boundaries passed the dataset checks.
Retokenization checks matched token IDs for 472 documents from shard boundaries.
A 1B-class model completed 3 training steps at length 12,288.
Training loss, gradient norms, and validation loss were finite. This checks integration, not convergence.

## Files and sharing

Compressed text files occupy 52.03 GB.
The local uint32 token cache occupies 120.25 GB, excluding indexes.
One GB is 1,000,000,000 bytes. The shared version contains text and a script to rebuild the cache.

[Private dataset at the checked revision](https://huggingface.co/datasets/long-sequence-length-dataset/fineweb-edu-long-30b/tree/f6e47756feff0186a052aaedeb4b258aef4454d4).
The upload receipt records successful remote file checks and streaming reads from both splits.

Local artifact: `/projects/jason_optai/kangan/datasets/fineweb-edu-long-30b`.

Manifest SHA-256: `4504314f51733ce085991de14b11ef595424d1a006b1ebb2475e4fc3d29579db`.

[Exact statistics](/projects/jason_optai/kangan/datasets/fineweb-edu-long-30b/length_statistics.json) · [Construction and evidence](appendix/construction.md).
