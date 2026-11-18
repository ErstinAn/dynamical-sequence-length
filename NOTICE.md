# Source and asset notice

This project derives from the local TorchTitan sequence length experiment code.
Some source files derive from TorchTitan code by Meta Platforms and its affiliates.
The applicable files contain their copyright notices.
The `LICENSE` file contains the BSD 3-Clause terms from TorchTitan.
The legal text is unchanged.

The dependency source is the public PyTorch TorchTitan repository.
The environment record identifies its source revision.
The repository does not include the full TorchTitan source.

The three tokenizer JSON files in `assets/tokenizer/` come from Meta Llama 3.
The [source record](assets/tokenizer/tokenizer_source.json) identifies their revision and SHA256 hashes.
The [Meta license](assets/tokenizer/LICENSE) and [use policy](assets/tokenizer/USE_POLICY.md) apply to these files.
Retain the [attribution notice](assets/tokenizer/Notice) and license when sharing the tokenizer.

FineWeb data and the historical Llama2 tokenizer are separate external assets.
Their source records identify the repositories, revisions, and hashes.
The project license does not grant new rights to those assets.
The preparation scripts download the assets from their source repositories.
