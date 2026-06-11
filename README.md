# model2vec-zig

[model2vec](https://github.com/MinishLab/model2vec) inference in pure Zig.
Static embeddings: a text becomes a vector through tokenization, a table
lookup, and a mean. There is no transformer at runtime, so there is nothing to
install and nothing to wait for.

```zig
const m2v = @import("model2vec");

var model = try m2v.Model.load(gpa, io, "models/potion-base-8M");
defer model.deinit();

const vec = try model.embed(allocator, "the daemon owns the store");
// []f32 of model.dim values, L2-normalized
```

Measured on potion-base-8M (x86_64 Linux, ReleaseFast): 4.1 us per embed of a
17-token text, about 240k embeds/s, from a ~30 MB model file. A warm local
Ollama round trip for the same job is 20-30 ms, and a cold one is seconds.

The bundled quantizer cuts that further: a TurboQuant-style 4-bit format runs
the 129 MB retrieval-tuned model from a 16 MB matrix at the same speed. On
the 10 retrieval tasks of MTEB(eng, v2) it scores 34.86 mean NDCG@10 against
35.06 for f32, measured on a harness that reproduces MinishLab's published
per-task scores. Why static embedders compress this well is written up in
[docs/turboquant.md](docs/turboquant.md).

## Why

Plenty of programs want semantic similarity but can't justify a model server:
CLI tools, agent hooks that run on every prompt, daemons that should work
offline on first run. Static embeddings make that trade explicit: roughly
82-92% of all-MiniLM-L6-v2's quality (see the
[model2vec results](https://github.com/MinishLab/model2vec/blob/main/results/README.md))
at microsecond latency with zero dependencies.

Pair it with a vector index and you have local semantic search inside one
static binary.

## Usage

Models load straight from their HuggingFace layout: a directory containing
`tokenizer.json`, `model.safetensors`, and `config.json`.

```bash
./scripts/fetch-model.sh potion-base-8M
```

| model | dim | disk | notes |
|---|---|---|---|
| potion-base-2M | 64 | ~8 MB | smallest |
| potion-base-8M | 256 | ~30 MB | fetch-model.sh default; benchmarked below |
| potion-retrieval-32M | 512 | ~125 MB | tuned for retrieval |

`Model.embed` allocates the output vector; `Model.embedInto` writes into a
caller-owned buffer and only uses its allocator for tokenization scratch, so
an arena reset between calls embeds with no per-call heap growth. The model is
read-only after load; concurrent embeds are fine if each call has its own
allocator.

## Scope

This runs the potion family, not every model on the Hub:

- WordPiece tokenizers only. BPE and Unigram models are rejected at load.
- F32 and I8 safetensors; f16 is not read yet. I8 models keep the quantized
  matrix in memory (4x smaller) and pool the raw values, which matches the
  reference because the global scale cancels under L2 normalization. The
  bundled `m2v-quantize` tool converts an f32 model to i8 with output
  byte-identical to the reference implementation's quantizer; embedding drift
  from quantization measures ~0.9997 cosine.
- A TurboQuant-style 4-bit format (`m2v-quantize --tq4`): rows are rotated by
  a fixed random orthonormal matrix (which makes uniform scalar quantization
  near-optimal), stored as signed nibbles with one scale per row, 8x smaller
  than f32. The rotation is never stored or undone; cosine is rotation
  invariant and queries pool from the same matrix, so only similarity
  structure is preserved, not coordinates. Vectors from a tq4 model are not
  comparable with any other build of the same model; persist them keyed to
  `Model.fingerprint()`. On the MTEB(eng, v2) retrieval suite, tq4 scores
  34.86 mean NDCG@10 against 35.06 for f32 (i8: 35.02). The reasoning and
  full measurements are in [docs/turboquant.md](docs/turboquant.md).
- The normalizer folds Latin accents with a table instead of full Unicode NFD
  (Zig's std has no normalization). Latin-script and code text matches the
  reference exactly; other scripts pass through unfolded and may tokenize to
  [UNK] where the reference would not.

Measured per format on potion-base-8M (same text and machine as above):

| format | matrix on disk | quality vs f32 | embed |
|---|---|---|---|
| f32 | 30.2 MB | exact | 4.2 us |
| i8 | 7.6 MB | ~0.9997 cosine | 4.8 us |
| tq4 | 3.9 MB | similarity drift < 0.05 | 4.8 us |

## Compared with model2vec-rs

The official Rust port ([model2vec-rs](https://github.com/MinishLab/model2vec-rs))
is the more complete implementation: it runs any tokenizer the Hub produces
through the HuggingFace `tokenizers` engine, reads f16 and i8 weights, has a
batch API, and fetches models from the Hub directly. This library covers the
potion family's single-text hot path and deliberately nothing else.

That difference in scope is also where the performance difference comes from.
Same machine, same model file (potion-base-8M), same 17-token text, 50k
iterations, release builds of both:

| | model2vec-zig | model2vec-rs 0.2.1 |
|---|---|---|
| single-text embed | 4.1-4.3 us (~240k/s) | 24.4 us (~41k/s) |
| peak RSS | 69.0 MB | 77.9 MB |
| model load | ~40 ms | ~68 ms |

The gap is design, not language. `encode_single` in the Rust crate goes
through its batch machinery and allocates a fresh vector per call, and the
general tokenizers engine pays for flexibility this library dropped;
`embedInto` here writes into a caller-owned buffer with arena scratch, so
steady-state embedding does not allocate. A Rust implementation shaped the
same way would close most of the distance. Both processes are dominated by
the same ~29 MB f32 matrix; the RSS difference is tokenizer structures and
allocator behavior, not the model.

For batch workloads the Rust crate amortizes far better and is the right
choice. This library is for the other case: one text at a time on a latency
budget, where the per-call overhead is the whole story.

## Testing

`zig build test` runs unit tests for the tokenizer, the safetensors reader,
and the pooling math against handcrafted fixtures. When a model is present
under `models/potion-base-8M`, it also runs a parity test: ten texts covering
accents, emoji, identifiers, and overlong words, compared against vectors
produced by the Python reference implementation. Max absolute difference is
under 1e-5. `zig build bench` prints embed throughput.

## License

MIT. The potion models are MinishLab's, also MIT.
