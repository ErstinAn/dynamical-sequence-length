# Mathematics and programming examples

[Overview](../README.md)

The directed check found mathematics, programming, and related educational content in sample-100BT.
It did not estimate category prevalence or category length distributions.

## Measured examples

Source counts use `token_count`. Llama counts use the original text without added special tokens.
The check did not truncate, concatenate, or pad the text.

| Category | Source page | Source tokens | Llama 2 | Llama 3 | Observation |
| --- | --- | ---: | ---: | ---: | --- |
| Mathematics and programming | [Isometric coordinate conversion](http://clintbellanger.net/articles/isometric_math/) | 2,080 | 2,196 | 1,665 | Equations and code are present. Some code occupies one line. |
| Applied mathematics concepts | [Asymmetric cryptography and PKI](https://resources.infosecinstitute.com/mathematical-algorithms-asymmetric-cryptography-introduction-public-key-infrastructure/) | 2,091 | 2,199 | 1,921 | Concepts are present. The check did not assess technical correctness. |
| Programming study guide | [ArcGIS and Python learning guide](http://geoawesomeness.com/programming-arcgis-python-beginners-guide/) | 2,108 | 2,154 | 1,978 | The page discusses learning Python. It is not code-dense. |
| Programming tutorial | [Python and wxPython menus](http://www.dreamincode.net/forums/topic/54399-python-wxpython-adding-a-menu-bar-and-menus/) | 2,119 | 2,160 | 1,928 | Classes and functions are present. Some Python indentation and line breaks are absent. |
| Programming and GIS | [Visual programming for GIS](https://gistbok.ucgis.org/bok-topics/2023-quarter-01/visual-programming-gis-applications) | 8,149 | 9,181 | 7,823 | The page describes workflows, tools, and references. |
| Mathematics lecture | [MIT graphs and networks lecture](http://ocw.mit.edu/courses/mathematics/18-085-computational-science-and-engineering-i-fall-2008/video-lectures/lecture-12-graphs-and-networks/) | 8,197 | 8,628 | 8,055 | The transcript includes graphs, matrices, null spaces, and positive definiteness. Figures are absent. |
| Computer concepts | [Cloud computing course notes](https://kvspgtcsamit.wordpress.com/cloud-computing-2/) | 16,360 | 17,443 | 15,545 | The page explains concepts and architecture. It is not primarily source code. |
| Low-quality mathematics candidate | [Math solver keyword page](https://softmath.com/math-com-calculator/adding-matrices/domain-of-square-root-function.html) | 32,425 | 35,151 | 29,887 | Keyword lists dominate. Continuous mathematical reasoning is absent from the inspected sections. |
| Programming textbook | [SAS Studio programming textbook](https://dokumen.pub/getting-started-with-sas-programming-using-sas-studio-in-the-cloud-hardcover-edition-1953329209-9781953329202.html) | 32,614 | 35,410 | 31,051 | SAS code is present. Download-site text and an incomplete ending are also present. |

## Scope and limits

Candidates came from narrow intervals within two percent of 2,096, 8,192, 16,384, and 32,768 source tokens.
The reviewer then selected nine pages by URL and inspected their content.
The sampling design explains the concentration near those lengths.
It provides no estimate for shorter or longer category members.

A mathematics keyword does not establish mathematical reasoning.
A computer topic does not establish source code content.
The longest examples include keyword aggregation or evidence of incomplete text.
Record length therefore does not establish quality, completeness, or a need for long context.

Category averages and percentiles require sampling across the full length distribution and checks of category labels.
Those measurements are not available from this directed check.

## Evidence

Dataset: `HuggingFaceFW/fineweb-edu`, configuration `sample-100BT`.
Revision: `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.

- [Inspection records](../../../outputs/fineweb_edu_profile/math_code_inspection/overview.json): document references, hashes, and counts.
- [Inspection script](../../../scripts/inspect_fineweb_math_code.py): extraction and tokenizer measurements.
- [Source data card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu#dataset-curation): dataset design and filtering context.

This report describes the inspected snapshot. Linked webpages can change.
