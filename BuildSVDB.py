import os
import re
import io
import sys
import json
import shutil
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def _env(key, default):
    val = os.environ.get(key)
    if val is None:
        return default
    val = val.strip()
    return val if val else default


def _env_int(key, default):
    try:
        return int(_env(key, str(default) if default is not None else ""))
    except ValueError:
        return default


DATA_DIR = _env("SVDB_DATA_DIR", os.path.join(HERE, "Data"))
DB_DIR = _env("SVDB_DB_DIR", os.path.join(HERE, "DataBase"))

OLLAMA_URL = _env("SVDB_OLLAMA_URL", "http://localhost:11434")
MODEL = _env("SVDB_MODEL", "qwen3-embedding:8b-q8_0")
EXPECT_DIM = _env_int("SVDB_EXPECT_DIM", None)
BATCH_SIZE = max(1, _env_int("SVDB_BATCH_SIZE", 48))
KEEP_ALIVE = _env("SVDB_KEEP_ALIVE", "10m")

HEADING_RE = re.compile(r"^#{1,6}\s")


def embed_batch(texts):
    payload = json.dumps(
        {"model": MODEL, "input": texts, "keep_alive": KEEP_ALIVE}
    ).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL + "/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as res:
        data = json.loads(res.read().decode("utf-8"))
    embs = data.get("embeddings")
    if not embs:
        raise RuntimeError("empty embeddings from Ollama: %r" % data)
    return embs


def split_by_heading(text):
    chunks = []
    cur = []
    for line in text.split("\n"):
        if HEADING_RE.match(line):
            if cur:
                chunks.append("\n".join(cur).strip())
            cur = [line]
        else:
            cur.append(line)
    if cur:
        chunks.append("\n".join(cur).strip())
    return [c for c in chunks if c]


def list_source_files():
    out = []
    for name in sorted(os.listdir(DATA_DIR)):
        path = os.path.join(DATA_DIR, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() in (".md", ".txt"):
            out.append((name, path))
    return out


def main():
    import numpy as np

    if not os.path.isdir(DATA_DIR):
        print("Data folder not found: %s" % DATA_DIR)
        sys.exit(1)

    files = list_source_files()
    if not files:
        print("No .md / .txt files in %s" % DATA_DIR)
        sys.exit(1)

    if os.path.isdir(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)

    records = []
    for name, path in files:
        with io.open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = split_by_heading(text)
        print("%s -> %d chunks" % (name, len(chunks)))
        for ch in chunks:
            records.append({"text": ch, "source": name})

    if not records:
        print("No chunks produced.")
        sys.exit(1)

    print("Embedding %d chunks with %s ..." % (len(records), MODEL))
    dim = EXPECT_DIM
    vecs = []
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        embs = embed_batch([r["text"] for r in batch])
        for vec in embs:
            if dim is None:
                dim = len(vec)
                print("Embedding dimension: %d" % dim)
            if len(vec) != dim:
                raise RuntimeError(
                    "dimension mismatch: got %d, expected %d (model=%s)"
                    % (len(vec), dim, MODEL)
                )
            vecs.append(vec)
        print("  [%d/%d]" % (min(start + BATCH_SIZE, len(records)), len(records)))

    arr = np.asarray(vecs, dtype=np.float32)
    np.save(os.path.join(DB_DIR, "vectors.npy"), arr)

    with io.open(os.path.join(DB_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(
            [{"text": r["text"], "source": r["source"]} for r in records],
            f, ensure_ascii=False, indent=1,
        )

    with io.open(os.path.join(DB_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"model": MODEL, "dim": dim, "count": len(records),
             "ollama_url": OLLAMA_URL},
            f, ensure_ascii=False, indent=1,
        )

    print("Done. %d vectors saved to %s" % (len(records), DB_DIR))


if __name__ == "__main__":
    main()
