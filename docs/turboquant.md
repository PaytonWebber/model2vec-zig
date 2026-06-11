# TurboQuant-style 4-bit quantization of static embedding matrices

Notes on why the tq4 format works, how it was validated, and where its limits
are. The implementation lives in `src/tq.zig`; this documents the reasoning
and the measurements behind it.

## The observation

model2vec inference is a mean over token vectors. Quantization noise on
individual matrix rows is roughly independent across the tokens of a text, so
pooling averages it down by about the square root of the token count, and the
L2 normalization at the end discards magnitude error entirely. A quantization
level that looks lossy per row can therefore be effectively lossless for the
pooled vectors a consumer actually compares.

TurboQuant's preconditioning makes the per-row part cheap to do well: rotating
by a fixed random orthonormal matrix concentrates coordinates toward
N(0, 1/d), so a plain uniform scalar quantizer per row is near-optimal. No
codebooks, no training, one f32 scale per row.

## The trick that removes the rotation from the runtime

Cosine similarity is invariant under orthonormal rotation, and every vector a
consumer compares is pooled from the same matrix. So the rotation is applied
once, at quantization time, and never stored or undone: the model simply
lives in a rotated basis nobody can observe from similarity scores. The
runtime cost of tq4 over i8 is unpacking two signed nibbles per byte, which
measures as zero (4.8 us per embed for both).

The corollary is the format's one sharp edge: vectors from a tq4 model are
not comparable with vectors from any other build of the same model, including
its own f32 or i8 form. `Model.fingerprint()` exists so consumers that
persist vectors can detect this and re-embed.

## Measurements

Quantizing potion-retrieval-32M (63,091 x 512):

| | f32 | i8 | tq4 (4-bit) | 3-bit (simulated) |
|---|---|---|---|---|
| matrix size | 129 MB | 32 MB | 16.4 MB | ~12 MB |
| row cosine vs f32 (mean) | 1.0 | ~0.9997 | ~0.993 | ~0.973 |
| embed latency | 4.2 us | 4.8 us | 4.8 us | n/a |

Retrieval, on a 20-document corpus of short developer notes with 5 paraphrase
queries (zero content-word overlap with their targets, the hardest case for a
static model): target ranks were unchanged between f32 and the 4-bit
simulation, and the full downstream product suite (hybrid dense+BM25 recall)
produced identical ranks under tq4 and i8 across paraphrase, identifier, and
mixed queries. Pairwise cosines between test texts drift by less than 0.05.

End to end in a consumer (agent-waymark's daemon, which holds the matrix
resident): 272 MB RSS with the f32 model, 87 MB with i8, 40.6 MB with tq4.

## Why 4-bit ships and 3-bit does not

The simulation says 3-bit also preserves retrieval ranks at this corpus
scale. 4-bit ships anyway because: row reconstruction is 0.993 vs 0.973
cosine, and downstream consumers apply absolute cosine thresholds (duplicate
detection, relevance floors) whose calibration should not absorb an extra
±0.02 of noise; nibbles pack two per byte while 3-bit values span byte
boundaries; and the difference is 4 MB on the largest potion model.

## Honest limits

- These measurements are at small-corpus scale (tens of documents, thousands
  of store entries). The mechanism predicts they hold generally, but no
  MTEB-scale evaluation has been run yet; do that before claiming
  "lossless" anywhere louder than this file.
- There is no external reference for the format. i8 is byte-identical to the
  reference implementation's quantizer and testable as such; tq4 correctness
  rests on the orthonormality, round-trip, and similarity-structure tests in
  this repo.
- The rotation is built with O(d^3) Gram-Schmidt at quantization time.
  Irrelevant at d=512; a structured rotation (Hadamard) would be the move if
  someone quantizes a much wider model.
