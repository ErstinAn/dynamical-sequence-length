# Documents at different lengths

[Statistics overview](../../README.md)

These eight examples contain two documents near each target length.
The targets use source `token_count`, with a two-percent tolerance.
The first target is 2,096, as originally requested. It is not 2,048.

| Target source tokens | Document | Actual source tokens | Llama 2 tokens | Llama 3 tokens |
| --- | --- | ---: | ---: | ---: |
| 2,096 | [Heat precautions](2096_01_heat_precautions.md) | 2,106 | 2,456 | 1,993 |
| 2,096 | [Civil War troop movements](2096_02_civil_war.md) | 2,069 | 2,319 | 2,090 |
| 8,192 | [July 2012 climate report](8192_01_climate_report.md) | 8,148 | 10,053 | 8,407 |
| 8,192 | [Bacterial resistance study](8192_02_research_article.md) | 8,167 | 9,630 | 7,857 |
| 16,384 | [Internet access definitions](16384_01_internet_access.md) | 16,215 | 19,156 | 15,736 |
| 16,384 | [Eastern forest ecoregions](16384_02_ecoregions.md) | 16,574 | 19,958 | 16,469 |
| 32,768 | [The first Fort Union](32768_01_fort_union.md) | 32,772 | 37,009 | 33,115 |
| 32,768 | [Southeastern archaeology](32768_02_archaeology.md) | 32,585 | 40,788 | 33,869 |

## Interpretation

Different tokenizers can give different lengths for the same document.
Llama counts exclude added special tokens and chat templates.
Llama 3 counts use the original Meta-Llama-3-8B tokenizer.
The examples contain no truncation, padding, or document concatenation.

The examples include instructions, historical narratives, research, and repeated regional entries.
Document length, training window length, and necessary context distance are different measurements.
A long document does not necessarily require long context for each prediction.
These examples do not measure long-distance dependencies.

Compare the [short history](2096_02_civil_war.md) with the [long history](32768_01_fort_union.md).
Compare the [ecoregions](16384_02_ecoregions.md) with the [archaeological entries](32768_02_archaeology.md).
Their topics and source dates differ. These comparisons are not controlled experiments.

## Selection and integrity

Source: `HuggingFaceFW/fineweb-edu`, configuration `sample-100BT`.
Revision: `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.
The selection used existing length indexes and random seed `20261104`.
Candidate counts were 1,024, 256, 256, and 1,024 for the four targets, respectively.
The reviewer then preferred institutional publications with different structures.
This intentional selection is not a representative sample of each length range.

Each Markdown file includes the full text record and its source coordinates.
The `raw/` files preserve the original UTF-8 bytes.
The [manifest](manifest.json) records hashes, source positions, and tokenizer counts.
The renderer checks that escaped Markdown restores the original text.
A full record can represent one page or chapter. It need not represent a complete book or original webpage layout.

The source text remains unchanged. Its historical claims and numbers belong to its original source.
Quoted source text is outside the ASD-STE100 check for the explanatory text.

## Reproduction

[prepare_fineweb_edu_examples.py](../../../../scripts/prepare_fineweb_edu_examples.py) extracts source rows and measures lengths.
[render_fineweb_edu_examples.py](../../../../scripts/render_fineweb_edu_examples.py) creates this folder from the extracted rows.
Tokenizer identities are in the `tokenizer_source.json` files under `assets/hf/llama-tokenizer` and `assets/hf/llama3-tokenizer`.
These examples remain local. This documentation change does not upload or publish them.
