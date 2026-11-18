"""Write complete, attributed reading examples from the extracted source rows."""

import hashlib
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/fineweb_edu_profile/document_examples"
OUTPUT = ROOT / "docs/fineweb-edu-stats/appendix/examples"

# The selection is intentional: two different document structures at each length.
EXAMPLES = [
    (2106, '2096_01_heat_precautions', 'Heat precautions', 'OSHA', 'Protective Measures to Take at Each Risk Level', 'The text gives instructions for water, work schedules, monitoring, and training in hot conditions. Agency service information follows the main content.', 'Compare the instructions and lists within one topic.'),
    (2069, '2096_02_civil_war', 'Civil War troop movements', 'National Park Service', 'Part I: A New Thrust into Virginia', 'The text describes Civil War troop movements, people, locations, and decisions.', 'Follow the links between people and actions. This record is one historical page, not a complete book.'),
    (8148, '8192_01_climate_report', 'July 2012 climate report', 'NOAA / NCDC', 'National Climate Report - July 2012', 'The report contains national and regional temperatures, rainfall, records, and lists.', 'Compare information across regions. The opening also contains navigation text for maps and other material.'),
    (8167, '8192_02_research_article', 'Bacterial resistance study', 'Emerging Infectious Diseases / CDC', 'Carbapenem Resistance in Clonally Distinct Clinical Strains of Vibrio fluvialis Isolated from Diarrheal Samples', 'The research article contains methods, results, discussion, and references. It includes gene names, measurements, and citations.', 'Compare the main text with the references. Both increase length but serve different purposes.'),
    (16215, '16384_01_internet_access', 'Internet access definitions', 'Preventing Chronic Disease / CDC', 'What Do We Mean by Internet Access? A Framework for Health Researchers', 'The article compares definitions of Internet access and presents a research framework. Author details, citations, and table content also remain.', 'Follow the use of definitions across the text. A short extracted line is not necessarily a paragraph.'),
    (16574, '16384_02_ecoregions', 'Eastern forest ecoregions', 'USDA Forest Service', 'Eastern Broadleaf Forest (Continental)', 'The page describes several ecoregions through repeated fields for landforms, soils, vegetation, and climate.', 'Compare repeated field names with their different content. Repeated structure does not establish duplicate text.'),
    (32772, '32768_01_fort_union', 'The first Fort Union', 'National Park Service', 'THE FIRST FORT UNION', 'The chapter describes Fort Union from its 1851 background through construction, supplies, personnel, and decisions. It includes quotations.', 'Follow recurring people and places across the chapter. The example does not measure long-distance dependencies.'),
    (32585, '32768_02_archaeology', 'Southeastern archaeology', 'National Park Service', 'SOUTHEAST PROPERTY TYPES', 'The text describes archaeological sites, categories, distributions, and table content.', 'Compare local entries with the continuous Fort Union narrative. Similar lengths can contain different structures.'),
]


def escape_line(line):
    """Display source characters as text, rather than executable HTML or Markdown."""
    return re.sub(r"([\\`*_{}\[\]()#+.!|~>\-])", r"\\\1", html.escape(line, quote=False))


def main():
    documents = json.loads((INPUT / "agency_documents.json").read_text())
    selection = json.loads((INPUT / "selection.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "raw").mkdir(exist_ok=True)
    rows = []
    manifest = {"source": selection, "examples": []}
    for tokens, slug, title, publisher, source_title, description, observation in EXAMPLES:
        matches = [d for d in documents if d["source_tokens"] == tokens]
        if len(matches) != 1:
            raise ValueError(f"Expected one source row for {slug}")
        d = matches[0]
        original = d["text"]
        raw = original.encode("utf-8")
        assert hashlib.sha256(raw).hexdigest() == d["text_sha256"]
        assert abs(tokens / d["target"] - 1) <= .02
        lengths = d["tokenizer_lengths_without_special_tokens"]
        raw_path = OUTPUT / "raw" / f"{slug}.txt"
        raw_path.write_bytes(raw)
        # Markdown hard breaks retain source line boundaries. Renderers can wrap long lines.
        body = "  \n".join(escape_line(line) for line in original.split("\n"))
        header = f"""# {title}

[Example index](README.md) · [Original UTF-8 text](raw/{slug}.txt)

## Reading notes

{description}

{observation}

The source block contains the complete text record from the historical dataset snapshot.
The renderer escapes display characters and preserves line boundaries. It does not translate or rewrite the source.
A complete record does not guarantee a complete webpage, book, image, or table layout.
Quoted source text is outside the ASD-STE100 check for the explanatory text.

## Length and source

| Item | Value |
| --- | --- |
| Target source length | {d['target']:,} |
| Source `token_count` | {tokens:,} |
| Difference from target | {tokens - d['target']:+,} ({(tokens / d['target'] - 1) * 100:+.2f}%) |
| Llama 2 tokens | {lengths['llama2']:,} |
| Llama 3 tokens | {lengths['llama3']:,} |
| Characters | {d['characters']:,} |
| Publisher | {publisher} |
| Crawl batch | `{d['dump']}` |

Llama counts exclude added BOS/EOS tokens and chat templates.
Counts apply only to source text, without these notes or Markdown escape characters.

Original title: {source_title}

[Source webpage]({d['url']}). Its current content can differ from the snapshot.

Dataset: `{selection['repo_id']}`, configuration `{selection['config']}`.

Revision: `{selection['revision']}`.

Parquet file: `{d['path']}`. Row index: `{d['row']}`, starting at zero.

Document ID: `{d['id']}`.

Original text SHA-256: `{d['text_sha256']}`.

## Source text

<!-- BEGIN SOURCE TEXT -->
"""
        result = header + body + "\n<!-- END SOURCE TEXT -->\n"
        (OUTPUT / f"{slug}.md").write_text(result, encoding="utf-8")
        # Verify the displayed content decodes to the complete original string.
        recovered = "\n".join(html.unescape(re.sub(r"\\(.)", r"\1", line)) for line in body.split("  \n"))
        assert recovered == original
        assert raw_path.read_bytes() == raw
        rows.append(f"| {d['target']:,} | [{title}]({slug}.md) | {tokens:,} | {lengths['llama2']:,} | {lengths['llama3']:,} |")
        manifest["examples"].append({**{k: v for k, v in d.items() if k != "text"},
                                    "markdown": f"{slug}.md", "raw_text": f"raw/{slug}.txt",
                                    "display_roundtrip_verified": True})

    readme = """# Documents at different lengths

[Statistics overview](../../README.md)

These eight examples contain two documents near each target length.
The targets use source `token_count`, with a two-percent tolerance.
The first target is 2,096, as originally requested. It is not 2,048.

| Target source tokens | Document | Actual source tokens | Llama 2 tokens | Llama 3 tokens |
| --- | --- | ---: | ---: | ---: |
""" + "\n".join(rows) + """

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
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote and verified {len(rows)} complete examples and README in {OUTPUT}")


if __name__ == "__main__":
    main()
