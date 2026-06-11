# TurboQuant-style 4-bit quantization of static embedding matrices

Notes on why the tq4 format works, how it was validated, and where its limits
are. The implementation lives in `src/tq.zig`; this documents the reasoning
and the measurements behind it.

## The observation

model2vec inference is a mean over token vectors. Quantization noise on
individual matrix rows is roughly independent across the tokens of a text, so
pooling averages it down by about the square root of the token count, and the
L2 normalization at the end discards magnitude error entirely. A quantization
level that is visibly lossy per row should therefore cost much less on the
pooled vectors a consumer actually compares. The benchmark section below
tests that prediction.

TurboQuant's preconditioning makes the per-row part cheap to do well: rotating
by a fixed random orthonormal matrix concentrates coordinates toward
N(0, 1/d), so a plain uniform scalar quantizer per row is near-optimal. No
codebooks, no training, one f32 scale per row.

## The rotation is not stored

Cosine similarity is invariant under orthonormal rotation, and every vector a
consumer compares is pooled from the same matrix. So the rotation is applied
once, at quantization time, and never stored or undone: the model lives in a
rotated basis that is not observable from similarity scores. The
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

## MTEB retrieval benchmark

MinishLab's published retrieval score for potion-retrieval-32M (35.06 on the
model card) is the mean NDCG@10 over the 10 retrieval tasks of MTEB(eng, v2),
produced with mteb 2.11.2 and a StaticModel-based loader (recorded in their
[published results](https://github.com/embeddings-benchmark/results/tree/main/results/minishlab__potion-retrieval-32M)).
The same setup (mteb 2.11.2, the `Model2VecModel` loader, the same pinned
dataset revisions) was run three times, changing only the embedding matrix:
the original f32, and the i8 and tq4 files written by `m2v-quantize`.
`scripts/mteb_eval.py` reproduces the run; per-task output including
Recall@100 is in `docs/mteb_results.json`.

The f32 run matches MinishLab's published scores on 9 of 10 tasks to all
five published decimals; ClimateFEVERHardNegatives differs by 0.00006. This
establishes that the harness is the same evaluation before the quantized
columns are read.

NDCG@10:

| task | published | f32 | i8 | tq4 |
|---|---|---|---|---|
| ArguAna | 0.44878 | 0.44878 | 0.44833 | 0.44666 |
| CQADupstackGamingRetrieval | 0.38125 | 0.38125 | 0.38061 | 0.38074 |
| CQADupstackUnixRetrieval | 0.19771 | 0.19771 | 0.19725 | 0.19501 |
| ClimateFEVERHardNegatives | 0.23433 | 0.23439 | 0.23450 | 0.23278 |
| FEVERHardNegatives | 0.46266 | 0.46266 | 0.46350 | 0.46003 |
| FiQA2018 | 0.18761 | 0.18761 | 0.18747 | 0.18835 |
| HotpotQAHardNegatives | 0.50085 | 0.50085 | 0.49993 | 0.50211 |
| SCIDOCS | 0.13693 | 0.13693 | 0.13691 | 0.13671 |
| TRECCOVID | 0.44690 | 0.44690 | 0.44274 | 0.43768 |
| Touche2020Retrieval.v3 | 0.50905 | 0.50905 | 0.51068 | 0.50601 |
| **mean** | 0.35061 | 0.35061 | 0.35019 | 0.34861 |

Against f32, i8 costs 0.0004 mean NDCG@10 and tq4 costs 0.0020 (0.57%
relative). Per task, nine of the ten tq4 deltas fall within ±0.0031, two of
them positive; the largest is TRECCOVID at -0.0092, a task with 50 queries.
Mean Recall@100 is 0.5925 for f32, 0.5920 for i8, and 0.5927 for tq4.

## Why 4-bit ships and 3-bit does not

3-bit was not run on MTEB; its numbers come from row reconstruction and a
small-corpus simulation, where it also preserved retrieval ranks. 4-bit
ships because: row reconstruction is 0.993 vs 0.973
cosine, and downstream consumers apply absolute cosine thresholds (duplicate
detection, relevance floors) whose calibration should not absorb an extra
±0.02 of noise; nibbles pack two per byte while 3-bit values span byte
boundaries; and the difference is 4 MB on the largest potion model.

## Limits

- The MTEB run covers retrieval only, on one model. MTEB(eng, v2) also
  scores classification, clustering, STS, and reranking; none of those were
  run, and neither were other models. The pooling argument is not
  retrieval-specific, but the measurements are.
- There is no external reference for the format. i8 is byte-identical to the
  reference implementation's quantizer and testable as such; tq4 correctness
  rests on the orthonormality, round-trip, and similarity-structure tests in
  this repo.
- The rotation is built with O(d^3) Gram-Schmidt at quantization time. This
  is negligible at d=512; a structured rotation (Hadamard) is the standard
  replacement at much larger d.
