import os
import io
import json
import urllib.request

import numpy as np
from mcp.server.fastmcp import FastMCP

HERE = os.path.dirname(os.path.abspath(__file__))


def _env(key, default):
    val = os.environ.get(key)
    if val is None:
        return default
    val = val.strip()
    return val if val else default


DB_DIR = _env("SVDB_DB_DIR", os.path.join(HERE, "DataBase"))
DEFAULT_MODEL = _env("SVDB_MODEL", "qwen3-embedding:8b-q8_0")
DEFAULT_OLLAMA_URL = _env("SVDB_OLLAMA_URL", "http://localhost:11434")
KEEP_ALIVE = _env("SVDB_KEEP_ALIVE", "30m")
SERVER_NAME = _env("SVDB_SERVER_NAME", "svdb")
TOOL_BASE = _env("SVDB_TOOL_BASE", "search_svdb")

QUERY_INSTRUCT = _env(
    "SVDB_QUERY_INSTRUCT",
    "Given a search query, retrieve passages from the document collection "
    "that explain or answer it.",
)

TRIGGER_MODE = _env("SVDB_TRIGGER_MODE", "marker").lower()

_DESC_HEAD = (
    "Search the local document collection by semantic similarity and return "
    "the passages that match best. Returns the top {count} passages, each with "
    "its source file name.\n\n"
)

_DESC_MARKER = (
    "TRIGGER (strict): call this tool only when the latest user message "
    "encloses a term in double square brackets, like [[term]]. That marker is "
    "the only thing that authorizes a search. Nothing else is a trigger: plain "
    "unmarked text, quoted text, single brackets [term], and any backtick or "
    "code formatting must never call this tool, no matter how much the "
    "question looks like it needs reference material. With no [[term]] present "
    "in the latest user message, answer from the conversation and your own "
    "knowledge instead.\n\n"
    "Args:\n"
    "    query: The exact text found between [[ and ]], and nothing else. Do "
    "not append the surrounding topic words: extra words blur the match and "
    "surface general overview passages instead of the exact section. If the "
    "first result set misses, you may search again with a reworded query."
)

_DESC_AUTO = (
    "Use it whenever the indexed material could answer the question or ground "
    "the reply with specific facts. If the topic is clearly unrelated to the "
    "collection, answer without it.\n\n"
    "Args:\n"
    "    query: The concept or phrase to look up. Keep it short and specific. "
    "Do not append broader topic names: extra words blur the match and surface "
    "general overview passages instead of the exact section you want. If the "
    "first result set misses, you may search again with a reworded query."
)

TOOL_DESCRIPTION = _env(
    "SVDB_TOOL_DESCRIPTION",
    _DESC_HEAD + (_DESC_AUTO if TRIGGER_MODE == "auto" else _DESC_MARKER),
)


def _parse_choices(raw):
    out = []
    for part in str(raw).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            n = int(part)
        except ValueError:
            continue
        if n >= 1 and n not in out:
            out.append(n)
    return out


TOP_K_CHOICES = _parse_choices(_env("SVDB_TOP_K_CHOICES", "1,2,3,4,5,6,7,8")) or [5]

mcp = FastMCP(SERVER_NAME)

_state = {"vectors": None, "chunks": None, "model": DEFAULT_MODEL,
          "ollama_url": DEFAULT_OLLAMA_URL, "loaded": False}


def _load():
    vec_path = os.path.join(DB_DIR, "vectors.npy")
    chunks_path = os.path.join(DB_DIR, "chunks.json")
    meta_path = os.path.join(DB_DIR, "meta.json")
    if not (os.path.isfile(vec_path) and os.path.isfile(chunks_path)):
        raise RuntimeError(
            "SVDB is empty. Run BuildSVDB first to create DataBase/vectors.npy."
        )
    vectors = np.load(vec_path).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    with io.open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    model = DEFAULT_MODEL
    ollama_url = DEFAULT_OLLAMA_URL
    if os.path.isfile(meta_path):
        with io.open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        model = meta.get("model", model)
        ollama_url = meta.get("ollama_url", ollama_url)
    if os.environ.get("SVDB_MODEL"):
        model = DEFAULT_MODEL
    if os.environ.get("SVDB_OLLAMA_URL"):
        ollama_url = DEFAULT_OLLAMA_URL
    _state.update(vectors=vectors, chunks=chunks, model=model,
                  ollama_url=ollama_url, loaded=True)


def _embed_query(text):
    prompt = "Instruct: %s\nQuery: %s" % (QUERY_INSTRUCT, text)
    payload = json.dumps(
        {"model": _state["model"], "prompt": prompt, "keep_alive": KEEP_ALIVE}
    ).encode("utf-8")
    req = urllib.request.Request(
        _state["ollama_url"] + "/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as res:
        data = json.loads(res.read().decode("utf-8"))
    vec = np.asarray(data.get("embedding"), dtype=np.float32)
    if vec.size == 0:
        raise RuntimeError("empty embedding from Ollama")
    n = np.linalg.norm(vec)
    if n != 0:
        vec = vec / n
    return vec


def _search(query, top_k):
    if not _state["loaded"]:
        _load()
    if not query or not query.strip():
        return "Empty query."
    k = max(1, min(int(top_k), len(_state["chunks"])))
    qv = _embed_query(query)
    scores = _state["vectors"] @ qv
    idx = np.argsort(-scores)[:k]
    parts = []
    for rank, i in enumerate(idx, 1):
        c = _state["chunks"][int(i)]
        parts.append(
            "## Result %d  (source: %s, score: %.3f)\n%s"
            % (rank, c.get("source", "?"), float(scores[int(i)]), c.get("text", ""))
        )
    return "\n\n".join(parts)


def _make_search_tool(top_k):
    def search(query: str) -> str:
        return _search(query, top_k)
    search.__name__ = "%s_%d" % (TOOL_BASE, top_k)
    return search


def _register_tools():
    for k in TOP_K_CHOICES:
        fn = _make_search_tool(k)
        description = TOOL_DESCRIPTION.replace("{count}", str(k))
        mcp.tool(name=fn.__name__, description=description)(fn)


_register_tools()


if __name__ == "__main__":
    mcp.run()
