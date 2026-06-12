# Changelog

## 0.2.0 (2026-06-11)

### Added

- Zero-copy model loading: on little-endian targets the matrix points into
  the file or embedded bytes when alignment allows, instead of being copied
  to the heap. Load time on potion-base-8M drops from 36 ms to 21 ms (f32)
  and `@embedFile` consumers stop paying RSS for a second copy of the
  matrix.
- tq4 files carry a `tq4_version` field in the safetensors metadata, and
  the reader rejects unknown versions, so a future layout change is an
  error instead of silently incomparable vectors.
- The safetensors parser is fuzzed: a randomized harness (raw bytes,
  length-framed bytes, corrupted valid files) runs on every `zig build
  test`, and the same invariants hook into `zig build test --fuzz`.
- `zig build bench` reports model load time.

### Changed

- `safetensors.parse` takes an options argument (`.{ .borrow = bool }`)
  and `Matrix` slices are now const.
- `Model.loadFromBytes` borrows the safetensors bytes, which must now
  outlive the Model. `@embedFile` data needs an aligned copy to get the
  zero-copy path; see the doc comment.
- tq4 files written by earlier versions are rejected (no `tq4_version`
  field); re-run `m2v-quantize --tq4` to regenerate them.

### Fixed

- Three parser overflow bugs against crafted files: a near-max header
  length wrapped the bounds check into a crash, tensor offset checks used
  wrap-prone addition, and an overflowing shape product could match an
  empty data region and hand out a matrix backed by no data.

## 0.1.0 (2026-06-10)

Initial release: model2vec/potion inference (WordPiece tokenization, mean
pooling, L2 normalization) parity-tested against the Python reference;
i8 quantizer byte-identical to the reference implementation's; a
TurboQuant-style 4-bit format validated on MTEB(eng, v2) retrieval and STS
against MinishLab's published per-task scores; `Model.loadFromBytes` for
`@embedFile`-shipped models; `Model.fingerprint()` for keying persisted
vectors.
