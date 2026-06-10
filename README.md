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

## Why

Plenty of programs want semantic similarity but can't justify a model server:
CLI tools, agent hooks that run on every prompt, daemons that should work
offline on first run. Static embeddings make that trade explicit: you get
roughly 82-92% of all-MiniLM-L6-v2's quality (see the
[model2vec results](https://github.com/MinishLab/model2vec/blob/main/results/README.md))
at microsecond latency with zero dependencies. For small corpora of short
texts, especially paired with lexical search, that is usually the right side
of the trade.

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
| potion-base-8M | 256 | ~30 MB | good default |
| potion-retrieval-32M | 512 | ~125 MB | tuned for retrieval |

`Model.embed` allocates the output vector; `Model.embedInto` writes into a
caller-owned buffer and only uses its allocator for tokenization scratch, so
an arena reset between calls embeds with no per-call heap growth. The model is
read-only after load; concurrent embeds are fine if each call has its own
allocator.

## Scope

This runs the potion family, not every model on the Hub:

- WordPiece tokenizers only. BPE and Unigram models are rejected at load.
- F32 safetensors only; quantized f16/i8 model files are not read yet.
- The normalizer folds Latin accents with a table instead of full Unicode NFD
  (Zig's std has no normalization). Latin-script and code text matches the
  reference exactly; other scripts pass through unfolded and may tokenize to
  [UNK] where the reference would not.

## Testing

`zig build test` runs unit tests for the tokenizer, the safetensors reader,
and the pooling math against handcrafted fixtures. When a model is present
under `models/potion-base-8M`, it also runs a parity test: ten texts covering
accents, emoji, identifiers, and overlong words, compared against vectors
produced by the Python reference implementation. Max absolute difference is
under 1e-5. `zig build bench` prints embed throughput.

## License

MIT. The potion models are MinishLab's, also MIT.
