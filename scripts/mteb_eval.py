"""MTEB retrieval eval of an f32 model vs its m2v-quantize i8/tq4 forms.

Reproduces the setup behind MinishLab's published retrieval score for
potion-retrieval-32M: mteb==2.11.2, the Model2VecModel loader
(StaticModel.encode), and the 10 retrieval tasks of MTEB(eng, v2) at their
pinned dataset revisions. Only the embedding matrix differs between
variants; the i8 and tq4 matrices are the files written by the Zig
m2v-quantize tool. Numbers in docs/turboquant.md come from this script.

Setup:

    pip install "mteb==2.11.2" model2vec
    ./scripts/fetch-model.sh potion-retrieval-32M
    zig build -Doptimize=ReleaseFast
    ./zig-out/bin/m2v-quantize models/potion-retrieval-32M/model.safetensors i8.st
    ./zig-out/bin/m2v-quantize --tq4 models/potion-retrieval-32M/model.safetensors tq4.st
    python scripts/mteb_eval.py models/potion-retrieval-32M i8.st tq4.st f32
    python scripts/mteb_eval.py models/potion-retrieval-32M i8.st tq4.st i8
    python scripts/mteb_eval.py models/potion-retrieval-32M i8.st tq4.st tq4

Published per-task scores to compare against:
https://github.com/embeddings-benchmark/results/tree/main/results/minishlab__potion-retrieval-32M
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import mteb
from mteb.models.model_implementations.model2vec_models import (
    Model2VecModel,
    potion_base_8m,
)

TASKS = [
    "ArguAna",
    "CQADupstackGamingRetrieval",
    "CQADupstackUnixRetrieval",
    "ClimateFEVERHardNegatives",
    "FEVERHardNegatives",
    "FiQA2018",
    "HotpotQAHardNegatives",
    "SCIDOCS",
    "TRECCOVID",
    "Touche2020Retrieval.v3",
]


def read_safetensors(path):
    raw = path.read_bytes()
    hlen = struct.unpack("<Q", raw[:8])[0]
    return json.loads(raw[8 : 8 + hlen]), raw[8 + hlen :]


def load_i8_matrix(path):
    header, data = read_safetensors(path)
    t = header["embeddings"]
    assert t["dtype"] == "I8"
    rows, cols = t["shape"]
    b, e = t["data_offsets"]
    return np.frombuffer(data[b:e], dtype=np.int8).reshape(rows, cols).astype(np.float32)


def load_tq4_matrix(path):
    header, data = read_safetensors(path)
    t = header["embeddings_tq4"]
    assert t["dtype"] == "U8"
    rows, half = t["shape"]
    b, e = t["data_offsets"]
    packed = np.frombuffer(data[b:e], dtype=np.uint8).reshape(rows, half)
    s = header["scales"]
    b, e = s["data_offsets"]
    scales = np.frombuffer(data[b:e], dtype=np.float32)
    lo = (packed & 0xF).astype(np.int8)
    lo[lo >= 8] -= 16
    hi = (packed >> 4).astype(np.int8)
    hi[hi >= 8] -= 16
    m = np.empty((rows, half * 2), dtype=np.float32)
    m[:, 0::2] = lo
    m[:, 1::2] = hi
    m *= scales[:, None]
    return m


def main():
    model_dir, i8_path, tq4_path, variant = sys.argv[1:5]
    task_names = sys.argv[5:] or TASKS

    model = Model2VecModel(model_dir)
    if variant == "i8":
        model.model.embedding = load_i8_matrix(Path(i8_path))
    elif variant == "tq4":
        model.model.embedding = load_tq4_matrix(Path(tq4_path))
    else:
        assert variant == "f32"

    meta = potion_base_8m.model_copy(
        update={
            "name": f"local/{Path(model_dir).name}-{variant}",
            "revision": "local",
            "embed_dim": model.model.embedding.shape[1],
        }
    )
    model.mteb_model_meta = meta

    tasks = mteb.get_tasks(tasks=task_names)
    results = mteb.evaluate(
        model,
        tasks=tasks,
        cache=mteb.cache.ResultCache(f"mteb-out/{variant}"),
        co2_tracker=False,
        show_progress_bar=True,
    )

    scores = {}
    for r in results.task_results:
        scores[r.task_name] = {
            "ndcg_at_10": r.get_score(getter=lambda s: s["ndcg_at_10"]),
            "recall_at_100": r.get_score(getter=lambda s: s["recall_at_100"]),
        }
    out = {"variant": variant, "scores": scores}
    print(json.dumps(out, indent=1))
    Path(f"mteb-out/{variant}-summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
