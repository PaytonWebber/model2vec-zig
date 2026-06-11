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

## MTEB benchmarks

MinishLab's published scores come from mteb with a StaticModel-based loader
(recorded in their
[published results](https://github.com/embeddings-benchmark/results/tree/main/results/minishlab__potion-retrieval-32M);
the model card's retrieval score of 35.06 for potion-retrieval-32M is the
mean NDCG@10 over the 10 retrieval tasks of MTEB(eng, v2)). The same setup
(mteb 2.11.2, the `Model2VecModel` loader, tasks taken from the MTEB(eng, v2)
benchmark object so dataset revisions and language subsets match) was run
three times per model, changing only the embedding matrix: the original f32,
and the i8 and tq4 files written by `m2v-quantize`. Two models were
evaluated, the 10 retrieval tasks and the 9 STS tasks each.
`scripts/mteb_eval.py` reproduces the runs; per-task output is in
`docs/mteb_results.json`.

The f32 runs match MinishLab's published scores before the quantized columns
are read: 9 of 10 retrieval tasks and 8 of 9 STS tasks exactly for
potion-retrieval-32M, 8 of 10 retrieval tasks and 5 of 7 comparable STS
tasks exactly for potion-base-8M, every remaining difference under 0.0001.
(potion-base-8M's published STS17 and STS22.v2 values come from a
multilingual run that averages all language subsets and are excluded from
the comparison; every other number follows the English-restricted protocol.)

Mean main scores (NDCG@10 for retrieval, Spearman for STS):

| model / suite | published | f32 | i8 | tq4 | tq4 vs f32 |
|---|---|---|---|---|---|
| retrieval-32M retrieval (10 tasks) | 0.35061 | 0.35061 | 0.35019 | 0.34861 | -0.0020 |
| retrieval-32M STS (9 tasks) | 0.73303 | 0.73302 | 0.73319 | 0.73249 | -0.0005 |
| base-8M retrieval (10 tasks) | 0.31111 | 0.31111 | 0.31095 | 0.31078 | -0.0003 |
| base-8M STS (9 tasks) | n/a | 0.72908 | 0.72927 | 0.72879 | -0.0003 |

The largest mean cost across the four suites is retrieval-32M retrieval at
0.0020 (0.57% relative); the other three sit at or under 0.0005. i8 is
within 0.0005 of f32 everywhere and above it on both STS suites.

Per-task, potion-retrieval-32M retrieval (NDCG@10):

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

Nine of the ten tq4 deltas fall within ±0.0031, two of them positive; the
largest is TRECCOVID at -0.0092, a task with 50 queries. Mean Recall@100 is
0.5925 for f32, 0.5920 for i8, and 0.5927 for tq4.

potion-retrieval-32M STS (Spearman):

| task | published | f32 | i8 | tq4 |
|---|---|---|---|---|
| BIOSSES | 0.78780 | 0.78780 | 0.78915 | 0.78409 |
| SICK-R | 0.66508 | 0.66508 | 0.66527 | 0.66439 |
| STS12 | 0.61378 | 0.61377 | 0.61373 | 0.61261 |
| STS13 | 0.73606 | 0.73606 | 0.73568 | 0.73723 |
| STS14 | 0.70464 | 0.70463 | 0.70462 | 0.70451 |
| STS15 | 0.81120 | 0.81120 | 0.81124 | 0.81078 |
| STS17 | 0.86113 | 0.86113 | 0.86143 | 0.86015 |
| STS22.v2 | 0.67658 | 0.67658 | 0.67648 | 0.67756 |
| STSBenchmark | 0.74096 | 0.74096 | 0.74113 | 0.74110 |

potion-base-8M retrieval (NDCG@10):

| task | published | f32 | i8 | tq4 |
|---|---|---|---|---|
| ArguAna | 0.41966 | 0.41961 | 0.41947 | 0.42147 |
| CQADupstackGamingRetrieval | 0.39235 | 0.39235 | 0.39212 | 0.39253 |
| CQADupstackUnixRetrieval | 0.20062 | 0.20062 | 0.19918 | 0.19761 |
| ClimateFEVERHardNegatives | 0.18724 | 0.18724 | 0.18703 | 0.18982 |
| FEVERHardNegatives | 0.33518 | 0.33518 | 0.33564 | 0.33636 |
| FiQA2018 | 0.16619 | 0.16619 | 0.16547 | 0.16507 |
| HotpotQAHardNegatives | 0.39040 | 0.39040 | 0.39021 | 0.39185 |
| SCIDOCS | 0.12387 | 0.12387 | 0.12298 | 0.12402 |
| TRECCOVID | 0.45747 | 0.45744 | 0.45902 | 0.45217 |
| Touche2020Retrieval.v3 | 0.43816 | 0.43816 | 0.43843 | 0.43685 |

tq4 scores above f32 on six of the ten tasks.

potion-base-8M STS (Spearman; published column omitted for STS17 and
STS22.v2 per the protocol note above):

| task | published | f32 | i8 | tq4 |
|---|---|---|---|---|
| BIOSSES | 0.75858 | 0.75858 | 0.75855 | 0.75714 |
| SICK-R | 0.64675 | 0.64675 | 0.64698 | 0.64635 |
| STS12 | 0.62249 | 0.62248 | 0.62275 | 0.62174 |
| STS13 | 0.77276 | 0.77276 | 0.77260 | 0.77261 |
| STS14 | 0.71914 | 0.71914 | 0.71918 | 0.71885 |
| STS15 | 0.79753 | 0.79753 | 0.79762 | 0.79779 |
| STS17 | n/a | 0.85967 | 0.86030 | 0.85914 |
| STS22.v2 | n/a | 0.63080 | 0.63109 | 0.63143 |
| STSBenchmark | 0.75405 | 0.75405 | 0.75432 | 0.75403 |

Over the seven comparable tasks, the published mean and our f32 mean are
both 0.72447.

## Why 4-bit ships and 3-bit does not

3-bit was not run on MTEB; its numbers come from row reconstruction and a
small-corpus simulation, where it also preserved retrieval ranks. 4-bit
ships because: row reconstruction is 0.993 vs 0.973
cosine, and downstream consumers apply absolute cosine thresholds (duplicate
detection, relevance floors) whose calibration should not absorb an extra
±0.02 of noise; nibbles pack two per byte while 3-bit values span byte
boundaries; and the difference is 4 MB on the largest potion model.

## Limits

- The MTEB runs cover retrieval and STS on two models (potion-retrieval-32M
  and potion-base-8M). Classification, clustering, and reranking were not
  run. The pooling argument is not task-specific, but the measurements stop
  there.
- There is no external reference for the format. i8 is byte-identical to the
  reference implementation's quantizer and testable as such; tq4 correctness
  rests on the orthonormality, round-trip, and similarity-structure tests in
  this repo.
- The rotation is built with O(d^3) Gram-Schmidt at quantization time. This
  is negligible at d=512; a structured rotation (Hadamard) is the standard
  replacement at much larger d.
