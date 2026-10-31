# Review of new English text

Review date: 2026-10-31.
Reference: [ASD-STE100 Issue 9, 2025-01-15](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf).

## Scope

This review covers the new English text for the dynamic training integration.
The scope includes additions to `README.md`, `cluster/vt_arc/README.md`, and `docs/tests.md`.
It also includes `docs/dynamic_seq_len.md`, new configuration comments, and new Python comments and docstrings.
The technical term list in `docs/writing.md` includes the terms for this feature.

Code identifiers, configuration values, shell commands, legal notices, and historical archives keep their spelling.
The review does not assess the earlier Chinese conversation against an English-language standard.
Existing text outside the integration changes is outside the scope of this review.

## Findings and changes

The text before review did not obey all applicable STE rules.
The changes include the following corrections:

| Finding | Correction | Rule |
| --- | --- | --- |
| One instruction combined package installation and tokenizer preparation. | Two numbered instructions give the two actions. | 5.2 |
| One instruction combined W&B configuration and authentication. | Five numbered instructions give the necessary actions. | 5.2, 5.3 |
| Some sentences put a condition after its instruction. | The condition now comes before the instruction. | 5.4 |
| Descriptions used passive voice where the actor was known. | The revised descriptions name the actor. | 3.6 |
| Prose and table cells contained semicolons. | Periods or different sentences replace the semicolons. | 8.1 |
| Some verbs had nonapproved uses, such as `retain` and `remain`. | The text uses alternatives such as `keep` and `stay`. | 1.1, 1.2 |
| Code comments used expressions such as `drain buffers` and `strand its peers`. | The comments give direct descriptions of the data and rank operations. | 4.1 |
| Some new technical terms had no project definitions. | The writing guide now gives their meanings and applicable uses. | 1.5, 1.11, 1.12 |

The review also examined sentence length, paragraph length, noun groups, and terminology.
Procedural sentences have a limit of 20 words.
Descriptive sentences have a limit of 25 words.
The terminology review used the official dictionary and the project term list.

## Verification and limits

The project text check gives no errors after the corrections.
It now detects semicolons in prose, including table cells.
It permits semicolons in executable code examples.
The Python syntax and format checks also give no errors.
The review changes descriptions and comments, not training parameters or training operations.

The script examines only some STE rules.
It cannot identify every incorrect word meaning, part of speech, or technical-term category.
The results record an editorial review, not a formal declaration of full ASD-STE100 conformity.
