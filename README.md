# model2vec-zig

[![CI](https://github.com/PaytonWebber/model2vec-zig/actions/workflows/ci.yml/badge.svg)](https://github.com/PaytonWebber/model2vec-zig/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/PaytonWebber/model2vec-zig)](https://github.com/PaytonWebber/model2vec-zig/releases)
[![License: MIT](https://img.shields.io/github/license/PaytonWebber/model2vec-zig)](LICENSE)

[Model2Vec](https://github.com/MinishLab/model2vec) inference in pure Zig.
Static embeddings: a text becomes a vector through tokenization, a table
lookup, and a mean. There is no transformer at runtime. About 4 us per embed
from a ~30 MB model file, with quantized formats down to 3.9 MB, in a binary
with zero dependencies.

## Quickstart

Add the dependency:

```bash
zig fetch --save=model2vec "git+https://github.com/PaytonWebber/model2vec-zig#v0.1.0"
```

```zig
// build.zig
const model2vec = b.dependency("model2vec", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("model2vec", model2vec.module("model2vec"));
```

Fetch a model and embed (the script lives in this repo; equivalently,
download `tokenizer.json`, `model.safetensors`, and `config.json` from the
model's HuggingFace page into a directory):

```bash
./scripts/fetch-model.sh potion-base-8M
```

```zig
const m2v = @import("model2vec");

var model = try m2v.Model.load(gpa, io, "models/potion-base-8M");
defer model.deinit();

const vec = try model.embed(allocator, "the daemon owns the store");
// []f32 of model.dim values, L2-normalized
```

`Model.loadFromBytes` takes the tokenizer and safetensors as byte slices, so
a model can be compiled into the binary with `@embedFile` and the program
ships as a single file.

## Main features

- **Small**: potion-base-8M is ~30 MB on disk as published. The bundled
  quantizer reduces it to 7.6 MB (i8) or 3.9 MB (4-bit) with measured
  quality cost (see [Quantization](#quantization)).
- **Fast inference**: 4.1 us per embed of a 17-token text on potion-base-8M
  (x86_64 Linux, ReleaseFast), about 240k embeds per second.
- **Zero dependencies**: Zig std only. No model server, no network, no
  native libraries. Embedding works offline on first run.
- **Reference parity**: output vectors match the Python implementation to
  an absolute difference under 1e-5, the i8 quantizer is byte-identical to
  the reference quantizer, and MinishLab's published MTEB scores reproduce
  per-task on this implementation's models (see [Quantization](#quantization)).
- **Allocation-free hot path**: `Model.embedInto` writes into a caller-owned
  buffer and uses its allocator only for tokenization scratch, so an arena
  reset between calls embeds with no per-call heap growth.

## What is this?

Model2Vec is MinishLab's technique for turning a sentence transformer into a
static embedding model: the vocabulary is passed through the transformer once
at distillation time, leaving a single embedding matrix. Inference is then
tokenize, look up, mean-pool, and normalize, which runs in microseconds and
reaches roughly 82-92% of all-MiniLM-L6-v2's quality (see the
[Model2Vec results](https://github.com/MinishLab/model2vec/blob/main/results/README.md)).

This repository implements that inference path in Zig for the potion model
family, plus quantization tooling. It targets programs that want semantic
similarity without operating a model server: CLI tools, agent hooks, daemons,
and anything that ships as a static binary.

## Models

Models load directly from their HuggingFace layout: a directory containing
`tokenizer.json`, `model.safetensors`, and `config.json`.

| Model | Dimensions | Disk | Notes |
|---|---|---|---|
| [potion-base-2M](https://huggingface.co/minishlab/potion-base-2M) | 64 | ~8 MB | smallest |
| [potion-base-8M](https://huggingface.co/minishlab/potion-base-8M) | 256 | ~30 MB | fetch-model.sh default; benchmarked below |
| [potion-retrieval-32M](https://huggingface.co/minishlab/potion-retrieval-32M) | 512 | ~125 MB | tuned for retrieval |

## Quantization

`m2v-quantize` converts a published f32 model:

```bash
m2v-quantize model.safetensors model.i8.safetensors        # i8, 4x smaller
m2v-quantize --tq4 model.safetensors model.tq4.safetensors # 4-bit, 8x smaller
```

The i8 scheme is the reference implementation's: output is byte-identical to
a Python-quantized model, and pooling uses the raw i8 values because the
global scale cancels under L2 normalization.

The 4-bit format follows the TurboQuant recipe: rows are rotated by a fixed
random orthonormal matrix, which makes uniform scalar quantization
near-optimal, then stored as signed nibbles with one f32 scale per row. The
rotation is never stored; cosine similarity is rotation-invariant and queries
pool from the same matrix. Vectors from a tq4 model are only comparable
within that quantized artifact, so persist them keyed to
`Model.fingerprint()`.

Measured quality on MTEB(eng, v2) with potion-retrieval-32M, using a harness
that reproduces MinishLab's published per-task scores (9 of 10 retrieval
tasks and 8 of 9 STS tasks match exactly):

| Format | Matrix size | Retrieval (mean NDCG@10) | STS (mean Spearman) |
|---|---|---|---|
| f32 | 129 MB | 0.35061 | 0.73302 |
| i8 | 32 MB | 0.35019 | 0.73319 |
| tq4 | 16.4 MB | 0.34861 | 0.73249 |

The same comparison on potion-base-8M measures tq4 within 0.0004 of f32 on
both suites. The reasoning and full per-task results are in
[docs/turboquant.md](docs/turboquant.md).

## Performance

Per format on potion-base-8M, same machine and text as above:

| Format | Matrix on disk | Embed |
|---|---|---|
| f32 | 30.2 MB | 4.2 us |
| i8 | 7.6 MB | 4.8 us |
| tq4 | 3.9 MB | 4.8 us |

The model is read-only after load; concurrent embeds are safe if each call
has its own scratch allocator. `zig build bench` reproduces the numbers.

## Limitations

- WordPiece tokenizers only, which covers the potion family. BPE and Unigram
  models are rejected at load.
- F32, I8, and this repo's tq4 safetensors are read; f16 is not.
- The normalizer folds Latin accents with a table instead of full Unicode
  NFD (Zig's std has no normalization). Latin-script and code text matches
  the reference exactly; other scripts pass through unfolded and may
  tokenize to [UNK] where the reference would not.
- Single-text API. There is no batch interface; for batch workloads use the
  [Python](https://github.com/MinishLab/model2vec) or
  [Rust](https://github.com/MinishLab/model2vec-rs) implementations.

## Testing

`zig build test` runs unit tests for the tokenizer, the safetensors reader,
and the pooling math against handcrafted fixtures. When a model is present
under `models/potion-base-8M`, it also runs a parity test: ten texts covering
accents, emoji, identifiers, and overlong words, compared against vectors
produced by the Python reference implementation, with a maximum absolute
difference under 1e-5.

## License

MIT. The potion models are MinishLab's, also MIT.

## Citation

If you use this library in research, cite Model2Vec:

```bibtex
@software{minishlab2024model2vec,
  authors = {Stephan Tulkens, Thomas van Dongen},
  title = {Model2Vec: Turn any Sentence Transformer into a Small Fast Model},
  year = {2024},
  url = {https://github.com/MinishLab/model2vec}
}
```
