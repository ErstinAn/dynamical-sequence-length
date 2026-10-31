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

The dynamic training descriptions also use these technical nouns.

| Technical noun | Meaning in this project |
| --- | --- |
| allocation account | The Slurm account that pays for requested resources. |
| array task | One independent job in a Slurm job array. |
| buffer | Memory that keeps tokens for later batches. |
| cyclic schedule | A schedule that increases and decreases sequence length at each step, then repeats. |
| data loader | The component that gives batches to the Trainer. |
| dataset | The source of documents or token sequences. |
| dynamic training | Training that changes sequence length between optimizer steps. |
| global batch size | The number of sequences in one optimizer step across all ranks. |
| ignore value | A label value that tells the loss function to omit that label. |
| label | The token ID that the model must predict. |
| local batch size | The number of sequences in one batch on one rank. |
| metric | A measured or calculated value that describes a training run. |
| optimizer step | One update of model parameters after all accumulation groups. |
| prefetch | Preparation of batches before the Trainer requests them. |
| replay mode | A benchmark mode that uses cached batches on the CPU. |
| scheduler | The component that selects sequence length from the optimizer step number. |
| source hash | A calculated value that identifies the program source files. |
| subclass | A class that extends another class in Python. |
| sweep | A set of separate training runs with different configuration values. |
| tensor | An array of values that the model or data loader uses. |
| throughput | The number of tokens per second. |
| token | One element of a tokenized document. |
| token budget | The total number of label tokens for an experiment. |
| warmup | Initial steps that precede a measurement or increase a learning rate. |

| Abbreviation | Meaning |
| --- | --- |
| CPU | Central processing unit. |
| DCP | Distributed checkpoint, the PyTorch checkpoint system. |
| FLOP | Floating-point operation. |
| GPU | Graphics processing unit. |
| LR | Learning rate. |
| MFU | Model FLOPs utilization. |
| QoS | Quality of service, which selects Slurm scheduling and billing settings. |
| RoPE | Rotary position embedding. |

Names of packages, classes, functions, and configuration keys keep their spelling.
The code identifiers and mathematical expressions are not prose.
The legal notice keeps its wording.
Historical source archives keep their previous text.

## Technical verbs

These terms describe computer or mathematical operations under STE rule 1.12.

| Technical verb | Meaning in this project |
| --- | --- |
| compile | Convert program code to an executable form. |
| authenticate | Confirm access to a computer service with account credentials. |
| cache | Keep data in memory or on disk for reuse. |
| download | Transfer a file from a server to this computer. |
| evaluate | Calculate model loss on evaluation data. |
| export | Put a named value in a process environment. |
| import | Load a Python module. |
| install | Put a software package in a software environment. |
| load | Read saved data into program memory. |
| log | Write metrics to an output record. |
| normalize | Divide a gradient or loss by its defined scale. |
| pack | Put document tokens into sequences of a specified length. |
| parse | Read configuration text as structured program values. |
| restore | Load saved state from a checkpoint into a program. |
| run | Start a program or test. |
| save | Write program data to storage. |
| stream | Read dataset records as the program uses them. |
| submit | Send a job request to Slurm. |
| synchronize | Complete the defined communication across ranks. |
| tokenize | Convert text to token IDs. |
| train | Update model parameters from examples and their labels. |

## Review procedure

1. Check each ordinary word against the STE dictionary.
2. Use each word with its approved meaning and part of speech.
3. Use the same technical term for the same item.
4. Give one action in each instruction.
5. Put a necessary condition before its instruction.
6. Use a maximum of 20 words in each procedural sentence.
7. Use a maximum of 25 words in each descriptive sentence.
8. Use active voice when the actor is known.
9. Remove unnecessary comments.
10. Run the text check.

   ```bash
   python scripts/check_text.py
   ```

The text check detects sentences with more than 25 words, semicolons, and non-ASCII prose.
It also detects selected contractions and phrases for review.
The ASCII restriction is a project rule.
The script does not identify procedural sentences or apply their limit of 20 words.
Manual review is necessary for procedural sentences, tables, code comments, and examples.

The script cannot identify every incorrect word meaning or grammatical form.
A check that passes alone does not prove STE compliance.
