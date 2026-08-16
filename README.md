# svdb-mcp

**SVDB (Simple Vector Data Base)** is a local semantic search over your own Markdown and text
files, exposed to an LLM as an MCP server.

Everything runs on your machine. Your documents are embedded by a local
[Ollama](https://ollama.com/) instance on `127.0.0.1` and stored as plain files next to the
script. Nothing is uploaded anywhere.

- Two Python files, no framework, no database server, no cloud account.
- Chunking is by Markdown heading, so a retrieved passage is a section you wrote.
- Search results carry the source file name, so the model can cite where a fact came from.
- By default the model may only search when you explicitly ask, using a `[[term]]` marker.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/) running locally
- An embedding model pulled into Ollama (see [Choosing an embedding model](#choosing-an-embedding-model))
- `numpy` and `mcp` (installed by `BuildSVDB.bat`, or `pip install -r requirements.txt`)

Windows users get two `.bat` helpers. Everything else is plain Python and runs anywhere.

## Quick start

1. **Start Ollama and pull an embedding model.**

   ```
   ollama pull qwen3-embedding:8b-q8_0
   ```

   That model is 8 GB and wants a GPU with 16 GB of VRAM. On a smaller machine, pull
   `qwen3-embedding:0.6b` or `nomic-embed-text` instead and set `SVDB_MODEL` accordingly.

2. **Put your documents in `Data/`.**

   Only `.md` and `.txt` directly inside `Data/` are indexed; subfolders are ignored.
   `Data/example.md` ships with this repository so the first build succeeds out of the box.

3. **Build the database.**

   ```
   python BuildSVDB.py
   ```

   Windows users can double-click `BuildSVDB.bat`, which installs the dependencies first.
   This writes `DataBase/vectors.npy`, `DataBase/chunks.json` and `DataBase/meta.json`.
   The build always regenerates the whole database; there is no incremental update.

4. **Register the MCP server** with your client (see below), then ask it something like:

   ```
   What does [[heading based chunking]] mean?
   ```

## MCP registration

The server speaks stdio. Point your client at the absolute path of `svdb_mcp.py`:

```json
{
  "mcpServers": {
    "svdb": {
      "command": "python",
      "args": ["C:\\path\\to\\svdb-mcp\\svdb_mcp.py"],
      "env": {}
    }
  }
}
```

Adjust the shape of that snippet to your client's config format. The important parts are the
command, the absolute path, and (optionally) the environment variables described below.

## How many passages come back

The server exposes eight tools that are identical except for how many passages they return:
`search_svdb_1` through `search_svdb_8`. The count is baked into the tool name instead of
being a parameter, so the model cannot quietly decide to pull eight sections into your context
when you wanted two.

**Expose exactly one of them.** How depends on your client:

- **[Prompoid](https://github.com/Nelson-Demarchi/Prompoid)** can enable and disable individual
  tools of a server. Tools named `<base>_<number>` are treated as a radio group there, so
  turning on `search_svdb_5` turns the other seven off. Leave the default and pick the count in
  the MCP settings pane whenever you like.
- **Any client without per-tool switches** (most of them) would show the model all eight
  near-identical tools, which is confusing at best. Restrict the set at the source instead:

  ```json
  "env": { "SVDB_TOP_K_CHOICES": "5" }
  ```

  Now only `search_svdb_5` exists. Change the number and restart the server to change the count.

## When the model is allowed to search

By default `SVDB_TRIGGER_MODE` is `marker`: the tool description tells the model to search
**only** when your message wraps a term in double square brackets.

```
What does [[heading based chunking]] mean?     -> searches for "heading based chunking"
What does heading based chunking mean?         -> no search
What does `heading based chunking` mean?       -> no search
```

This exists because an always-available search tool gets called constantly, on questions that
did not need it, spending time and context. An explicit marker keeps retrieval under your
control. `[[ ]]` was chosen after backtick markers failed: models cannot reliably tell one
backtick from two from three, and treat them all as generic code formatting, while `[[ ]]` is
familiar from wiki links and is never typed by accident.

If you would rather have the model decide for itself, set `SVDB_TRIGGER_MODE=auto`. The tool
then advertises itself as usable whenever the collection could help.

Either way the trigger is a description, not a hard rule — it steers the model, it does not
constrain it. Replace the whole text with `SVDB_TOOL_DESCRIPTION` if you want different
behaviour.

## Choosing an embedding model

Set `SVDB_MODEL` for **both** the build and the server, then rebuild. Vector dimensions are
detected automatically, and the model name is recorded in `DataBase/meta.json` so the server
queries with whatever the database was built with.

| Model | Size | Dim | Notes |
|---|---|---|---|
| `qwen3-embedding:8b-q8_0` | ~8 GB | 4096 | The default. Strong multilingual quality, wants a 16 GB GPU. |
| `qwen3-embedding:0.6b` | ~1.2 GB | 1024 | Same family, runs on modest hardware. |
| `nomic-embed-text` | ~270 MB | 768 | Very small and fast, English-centric. |

```
set SVDB_MODEL=qwen3-embedding:0.6b
python BuildSVDB.py
```

and the matching entry in your MCP config:

```json
"env": { "SVDB_MODEL": "qwen3-embedding:0.6b" }
```

**Changing the model invalidates the database.** Vectors from different models are not
comparable, so always rebuild after switching.

## Chunking

A chunk is a Markdown heading line (`#` through `######`) plus everything up to the next
heading. A file with no headings becomes a single chunk, which makes retrieval coarse — give
long documents headings.

One chunk should be one topic. That is what makes its vector specific enough to be found: a
whole document averaged into one vector matches everything vaguely and nothing precisely.

### How much text fits in one chunk

A vector never stores your words, so nothing is ever paraphrased or lost: `chunks.json` keeps
the text verbatim and hands it back in full when a chunk is retrieved. The vector is only the
key used to find it. What varies with chunk length is not fidelity but *findability*.

Two limits are worth knowing:

- **The embedding model's context.** `qwen3-embedding` accepts 40960 tokens, roughly 30,000
  English words, in a single chunk. Anything past that is truncated. In practice you will never
  reach it with heading-sized sections.
- **Meaning resolution.** One chunk becomes one point in meaning space, and everything it
  discusses is averaged into that point. Roughly: **50–500 words retrieves excellently**, up to
  about **1,000 words is still sharp if the section stays on one topic**, and several thousand
  words of mixed topics starts matching everything weakly. Splitting by heading usually lands in
  the good range by itself.

The dimension count (4096 for the default model, 1024 or 768 for the smaller ones) is not a
storage budget either. It is how finely the model can distinguish one meaning from another —
4096 numbers, 16 KB per chunk. More dimensions separate near-synonymous passages better; fewer
are perfectly adequate for a collection of a few thousand chunks, and cost less memory and time.

## Environment variables

Empty values are treated as unset and fall back to the default.

| Variable | Used by | Default |
|---|---|---|
| `SVDB_TOP_K_CHOICES` | server | `1,2,3,4,5,6,7,8` |
| `SVDB_TRIGGER_MODE` | server | `marker` (or `auto`) |
| `SVDB_TOOL_DESCRIPTION` | server | generated from the trigger mode, `{count}` is substituted |
| `SVDB_QUERY_INSTRUCT` | server | generic retrieval instruction prefixed to queries |
| `SVDB_TOOL_BASE` | server | `search_svdb` |
| `SVDB_SERVER_NAME` | server | `svdb` |
| `SVDB_MODEL` | both | `qwen3-embedding:8b-q8_0` |
| `SVDB_OLLAMA_URL` | both | `http://localhost:11434` |
| `SVDB_KEEP_ALIVE` | both | `30m` (server) / `10m` (build) |
| `SVDB_DB_DIR` | both | `DataBase/` next to the scripts |
| `SVDB_DATA_DIR` | build | `Data/` next to the scripts |
| `SVDB_BATCH_SIZE` | build | `48` |
| `SVDB_EXPECT_DIM` | build | unset, detected from the first embedding |

`SVDB_QUERY_INSTRUCT` is worth knowing about: Qwen3-Embedding models are trained to receive a
short instruction on the query side only. The default is deliberately generic. Tailoring it to
what your collection actually contains can measurably improve ranking.

## `run-ollama-embed.bat`

Optional Windows helper that starts an Ollama instance serving models from `ollama-models/`
inside this folder, so a large embedding model does not have to live in your global Ollama
store.

**It terminates every running Ollama process first**, so it will kill an instance you were
already using. It asks for confirmation before doing so. Skip this script entirely if you are
happy with your normal Ollama setup — SVDB only needs some Ollama reachable at
`SVDB_OLLAMA_URL`.

Machine-specific settings (GPU pinning and the like) go in `ollama-env.bat` next to it, which
is gitignored and sourced if present:

```bat
@echo off
set "CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
```

## Files

```
svdb_mcp.py            MCP server (stdio). Loads the database, embeds the query, ranks by cosine similarity.
BuildSVDB.py           Chunks Data/, embeds in batches, writes DataBase/.
BuildSVDB.bat          Windows: install dependencies, then build.
run-ollama-embed.bat   Windows: optional dedicated Ollama launcher.
Data/                  Your .md and .txt files. Only example.md is tracked.
DataBase/              Generated. vectors.npy, chunks.json, meta.json.
```

## Troubleshooting

**`SVDB is empty. Run BuildSVDB first`** — there is no `DataBase/vectors.npy`. Build it.

**HTTP 404 from Ollama** — the server is up but does not have the model. `ollama list` will
show an empty list if `OLLAMA_MODELS` points somewhere other than where you pulled it.

**The model never searches** — you are in `marker` mode and did not write `[[term]]`. Note the
trigger looks at your latest message only.

**The model searches all the time** — you are in `auto` mode. Switch to `marker`.

**Results miss the obvious section** — check the query the model sent. Appending the broader
topic ("russian postmodernism narrative self-display") blurs the query vector and favours long
overview sections over the precise one. The bracket content alone works better. Both default
descriptions tell the model this, but a short query you control beats a long one it invents.

## License

MIT. See [LICENSE](LICENSE).
