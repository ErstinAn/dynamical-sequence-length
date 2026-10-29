# Writing guide

The reference is ASD-STE100 Issue 9, dated January 15, 2025.
The [official standard](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf) supplies the rules and dictionary.
The new project text uses technical nouns for software and mathematics.
The table below defines the main project terms.

## Project terms

| Technical noun | Meaning in this project |
| --- | --- |
| accumulation group | One backward operation on each DP rank before an optimizer update. |
| artifact | A fixed set of validation arrays and identity records. |
| batch | A set of sequences for one operation. |
| checkpoint | Saved state for recovery of a training run. |
| configuration | Values that select the model and program behavior. |
| data parallelism (DP) | Training with model replicas and separate data on each rank. |
| gradient | The derivative of the loss with respect to a parameter. |
| gradient norm | The length of the complete gradient vector. |
| gradient variance | The variance of synchronized accumulation-group gradients. |
| manifest | A file that records the identity of data or an experiment. |
| optimizer | The component that changes model parameters from their gradients. |
| parameter group | Parameters that use the same optimizer settings. |
| rank | The integer identity of a distributed process. |
| repository | The project files under Git control. |
| sequence length | The number of token positions in one training sequence. |
| smoke test | A short test of the main program path. |
| tokenizer | The component that converts text to token IDs. |
| training run | One execution of the training program. |
| validation | Loss measurement on the fixed evaluation data. |
| wheel | A Python installation archive. |

Names of packages, classes, functions, and configuration keys keep their spelling.
The code identifiers and mathematical expressions are not prose.
The legal notice keeps its wording.
Historical source archives keep their previous text.

## Technical verbs

These terms describe computer or mathematical operations under STE rule 1.12.

| Technical verb | Meaning in this project |
| --- | --- |
| compile | Convert program code to an executable form. |
| download | Transfer a file from a server to this computer. |
| evaluate | Calculate model loss on evaluation data. |
| export | Put a named value in a process environment. |
| import | Load a Python module. |
| normalize | Divide a gradient or loss by its defined scale. |
| pack | Put document tokens into sequences of a specified length. |
| run | Start a program or test. |
| stream | Read dataset records as the program consumes them. |
| synchronize | Complete the defined communication across ranks. |
| tokenize | Convert text to token IDs. |

## Review procedure

1. Check each ordinary word against the STE dictionary.
2. Use each word with its approved meaning and part of speech.
3. Use the same technical term for the same item.
4. Give one action in each instruction.
5. Put a necessary condition before its instruction.
6. Check the sentence lengths against the applicable STE rule.
7. Use active voice when the actor is known.
8. Remove unnecessary comments.
9. Run the text check.

   ```bash
   python scripts/check_text.py
   ```

The text check detects long sentences and non-ASCII prose.
It also detects selected contractions and phrases for review.
It does not check every dictionary meaning or grammatical rule.
A check that passes alone does not prove STE compliance.
