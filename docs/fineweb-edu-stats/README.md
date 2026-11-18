# FineWeb-Edu statistics

This folder contains statistics for FineWeb-Edu and our derived long-document dataset.
The reports describe fixed dataset versions. They do not describe later source updates.

| Dataset | Scope | Mean document length | Token basis | Details |
| --- | --- | ---: | --- | --- |
| FineWeb-Edu sample-100BT | All 140 files | 1,031.72 | Source `token_count` | [Source statistics](fineweb-edu.md) |
| FineWeb-Edu default | All 2,410 files | 1,037.64 | Source `token_count` | [Source statistics](fineweb-edu.md) |
| fineweb-edu-long-30b | Selected train split | 22,254.09 | Llama 3, without special tokens | [Long-document statistics](fineweb-edu-long-30b.md) |

The source field uses GPT-2 token units. The derived dataset uses the original Llama 3 tokenizer.
Counts from these tokenizers are not interchangeable.
The long-document report also includes the validation split and usable training labels.

## Terms

| Term | Meaning |
| --- | --- |
| Document | One text record in the dataset. |
| Document length | The token count for one complete text record. |
| Tokenizer | The program that converts text into token IDs. |
| Source token | One unit in the source `token_count` field. |
| Raw Llama 3 token | One token from the document text, without added special tokens. |
| Label token | One prediction target that contributes to loss. |
| Window | A fixed-length part of one document used by the training reader. |
| Split | A separate group of documents for training or validation. |
| P90 | The length at the 90th percentile of the document distribution. |
| B | One billion, or 1,000,000,000. |

A minimum-length table includes complete documents that meet each threshold.
The threshold does not truncate each document.
The rows overlap, so their counts must not be added.
Window tables instead count labels after the reader discards incomplete tails.

## Supporting material

- [Construction and evidence](appendix/construction.md): selection, duplicate removal, data split, checks, and source records.
- [Mathematics and programming](appendix/math-code.md): nine directed examples, without estimates of category prevalence.
- [Reading examples](appendix/examples/README.md): eight documents with original text and tokenizer counts.
- [Distribution figure](figures/length-distribution.svg): a vector copy for reports.

The explanatory text follows the ASD-STE100 structural rules available in the project skill.
This check does not establish full compliance with the official ASD dictionary.
Quoted dataset text remains unchanged and is outside this writing check.
