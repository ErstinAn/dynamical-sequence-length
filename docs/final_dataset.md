# Fixed-order dataset and fixed validation

Use [getting_started.md](getting_started.md) for the current environment, download, and training procedure.
This document defines dataset contents and records historical configuration compatibility.

The active dataset is `assets/data/fineweb_edu_long_30b_shuffled_v1`.
The original train and validation document lists remain separate.
Token IDs and document text remain unchanged.

The training split contains 1,348,067 documents and 30,000,004,070 raw text tokens.
The complete validation split contains 2,725 documents and 62,360,433 raw text tokens.
These counts exclude added BOS and EOS tokens.
The dataset's `README.md` gives the measured document-length histograms, quantiles,
and fixed-validation token fractions. The training document median is 18,084 tokens.
The complete validation document median is 18,522 tokens.

Shuffling changes the reading order, not the corpus length histogram.
Across 32 equal-document-count blocks, training document means range from
22,112.75 to 22,414.24 tokens. This removes the original source blocks.
These statistics do not establish that every batch or loss value is stable.

## Document order

`orders/train.npy` contains one permutation of all training document IDs.
The seed is 42. The generator is NumPy PCG64.
`orders/validation.npy` uses seed 43.
The stored arrays define the order. A seed alone is not the dataset identity.

Use this repository's `LongDocumentBatchDataLoader` with `packing: stream`.
The reader applies the order before assigning documents to data-parallel ranks.
It preserves document tails when the training sequence length changes.
It preserves document boundaries, label masks, and position rules.
It saves the order cursor in each checkpoint.

The Parquet files retain their original row order.
Calling `datasets.load_dataset()` alone does not apply the stored permutation.
Use the matching reader from this source snapshot.

Set `dataloader.infinite: false` for one pass.
Check the required parent positions against each rank's available positions before training.
The original 3,840-step loss/gradient run consumes 60.67B parent positions and does not fit one pass.
A gradient-token budget below 30B does not establish that the parent stream fits one pass.
`statistics.json` includes the available physical positions for four ranks.

## Fixed validation

The `validation/` directory contains 936 fixed sequences of length 16,384.
Each sequence stays within one document. There is no packing or padding.
Long documents can provide multiple nonoverlapping sequences.
Blocks do not share input tokens or the extra token used for the final prediction label.
Each candidate block contains 16,385 positions, including that extra token.

The source document has virtual BOS/EOS tokens at its actual boundaries.

The preparation script creates 2,407 candidate blocks from 1,759 eligible documents.
It chooses a random starting offset within each document's unused space, with seed 45.
It then selects 936 blocks without replacement, with seed 44.
Longer documents can contribute more candidate blocks. Documents are not sampled uniformly.
Every evaluation uses the same input, label, and position arrays.

The sample has 15,335,424 valid prediction labels.
It uses 798 distinct documents: 793 original-pool documents and 5 supplement documents.
One hundred documents provide multiple selected windows, with a maximum of five windows per document.
Supplement text supplies 0.5342% of sampled raw tokens.
The complete validation split still contains 2,725 documents, including 982 supplement documents.
However, 966 documents cannot supply a full block and are excluded from this evaluation sample.

The sample measures long-document performance. It does not represent the complete mixed validation distribution.
The earlier packed-validation representation tolerances do not apply to this selection rule.
Do not compare its loss directly with the former 22,256-token packed validation loss.
Different windows from one document can remain statistically related even when they share no tokens.
`statistics.json` contains all counts and distributions.

`independent_validation_check.json` records a separate reconstruction of every selected sequence.

Validation manifest SHA256:
`eb2faf62a36e324d07e08da73170e1268182cdfea1f38352362b86232f47155b`.

Training manifest (`manifest.json`) SHA256:
`4bf6946f7b7f7cbad0799746519baaa9ef2518f974d2239a8df2f184db004806`.

Dataset manifest (`dataset_manifest.json`) SHA256:
`7e284ce4fc90a93e5bdcee160a60051dc29ed17775b8ecb261feafd93f047d80`.

## Configuration

`training_settings.json` contains the data overrides for an experiment YAML's `fixed` section.
Use a native Llama 3 tokenizer/model configuration.
Set the model's supported context capacity to at least 16,384 for this validation length.
Keep the experiment's training length and batch schedules explicit.
The main validator reads the fixed arrays directly.
Keep `validator.eval_seq_len: 16384` and `validator.token_budget: 15335424`.

Use `configs/smoke.yaml` for the eight-step smoke.
Use section 6 of [the run guide](getting_started.md#6-run-automatic-checks) to check the plan.

To reproduce the fixed sample in a new output directory:

```bash
python scripts/prepare_document_validation.py \
  --dataset assets/data/fineweb_edu_long_30b_shuffled_v1 \
  --output /path/to/new-validation-directory \
  --tokenizer assets/hf/llama3-tokenizer \
  --length 16384 --count 936 --sample-seed 44 --offset-seed 45
```

## Transfer to a collaborator

The original local `data/` and `tokens/` entries are directory links.
The collaborator must create standalone directories from the pinned remote files.
Allow approximately 52 GB for text files and another 120 GB for token IDs.
The main guide includes additional space for checkpoints, environments, and download caches.

Clone the default GitHub branch and record its commit.
Install the runtime with `requirements-portable.txt`.
The original lock alone cannot resolve every historical nightly wheel.
Each collaborator needs Hugging Face access to the private dataset.
The Git checkout supplies the tokenizer in `assets/tokenizer/`.

Use `scripts/download_dataset.py` from the delivered repository to download the pinned revision.
Use `scripts/rebuild_long_token_cache.py` to reconstruct the token cache.
Follow the complete commands and success checks in [the main guide](getting_started.md#5-download-and-rebuild-the-dataset).
The current dataset revision is `b6ea2fde77bab36dff963aba5e6f482477959933`.
Preserve its exact document orders and fixed validation arrays.

The private Hub repository is `long-sequence-length-dataset/fineweb-edu-long-30b`.
It supplies text, metadata, document orders, validation arrays, and supporting code.
It does not supply the training token cache.
The delivered repository contains the tested preparation and training code.
The older Hub code archive does not replace that delivery.

## Archived local data

Fourteen older assets moved to `archive/data/20261116/`.
`archive/data/manifest.json` records their paths and original manifest hashes.
The data path resolver redirects missing historical local paths to these exact archived versions.
It never substitutes the new dataset for an old identity.
The archive is local and is not included in the collaborator transfer package.

The original local package links to the unchanged corpus at
`/projects/jason_optai/kangan/datasets/fineweb-edu-long-30b`.
The independent handoff test created a separate complete cache from downloaded text.
