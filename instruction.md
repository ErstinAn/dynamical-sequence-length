# Run instructions

The main procedure is [the run guide](docs/codebase_kick_off/getting_started.md).
Start with that document.

The current experiment uses a dense 1B model with the native Llama3 vocabulary and the custom dataset of long documents.
Historical FineWeb 100BT experiments and the Llama2 tokenizer are outside the default procedure.
The current configuration is the eight-step smoke with global batch size 256.
Follow the main guide to measure microbatches and check saving and recovery.
Formal schedules and learning rate sweeps remain pending.
