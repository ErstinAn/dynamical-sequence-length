# Data procedure

## Data identity

The source is `HuggingFaceFW/fineweb`, with configuration `sample-100BT`.
The source revision is `9bb295ddab0e05d785b879661af7260fed5140fc`.
The file `sample/100BT/000_00000.parquet` supplies validation documents.
The other 149 files supply training documents.

The training loader excludes validation document IDs and normalized text hashes.
This check removes exact copies under the defined text normalization.
It does not identify all documents with similar meanings.
The manifest records the file list, source sizes, and source hashes.

Each validation uses 20,971,520 label tokens.
The five Llama2 artifacts contain the same ordered input and label tokens.
The sequence length changes their grouping and position arrays.
The code checks the source, tokenizer, document list, and array hashes.

## Prepare the tokenizer

Run the tokenizer procedure from the main README.
The default destination is `assets/hf/llama-tokenizer`.
For another asset directory, pass its tokenizer path with `--output`.

## Prepare validation data

1. Download the validation source file.

   ```bash
   python - <<'PY'
   from huggingface_hub import hf_hub_download
   print(hf_hub_download(
       repo_id="HuggingFaceFW/fineweb",
       repo_type="dataset",
       revision="9bb295ddab0e05d785b879661af7260fed5140fc",
       filename="sample/100BT/000_00000.parquet",
   ))
   PY
   ```

2. Set `SOURCE_PARQUET` to the path from the command output.

   ```bash
   export SOURCE_PARQUET=/path/to/000_00000.parquet
   ```

3. Create the artifact for each sequence length.

   ```bash
   for length in 256 512 1024 2048 4096; do
     python scripts/prepare_fineweb100_validation.py \
       --source-parquet "$SOURCE_PARQUET" \
       --tokenizer assets/hf/llama-tokenizer \
       --seq-len "$length" \
       --output "assets/data/fineweb100_validation_llama2_$length"
   done
   ```

4. Check the artifact identities.

   ```bash
   DYNAMIC_SEQ_LEN_ARTIFACT_TEST=1 PYTHONPATH=.:cluster/vt_arc OMP_NUM_THREADS=1 \
     python -m unittest \
     tests.test_formal_experiment.FormalExperimentTest.test_built_artifact_checksums_vocabulary_and_training_exclusions
   ```

The preparation command does not overwrite an existing artifact.
Each artifact occupies approximately 484 MiB.
For a different hash, examine the tokenizer and data preparation.
A different hash is not a reason to remove the identity check.

## Training data access

The default loader streams data from Hugging Face.
Network access is necessary for data that are not in the cache.
An offline data directory must contain all 149 training files.
The loader checks the file names and sizes in that directory.
It does not calculate all training file hashes at each start.

## Data files in Git

Git excludes the tokenizer files and all validation arrays.
The repository contains only their source records and reference hashes.
The package keeps the Qwen reference for identity checks on historical data.
New experiment configurations use the Llama2 tokenizer.
