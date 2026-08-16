# SVDB example document

This file exists so that a freshly cloned repository can be built and searched immediately.
It also happens to describe how SVDB works, so the answers you get while testing are true.
Replace it with your own material once you are done experimenting.

## What SVDB is

SVDB is a small semantic search index over local Markdown and text files, exposed to a language
model through the Model Context Protocol. It has three moving parts: a build script that turns
documents into vectors, a database of plain files, and an MCP server that answers queries.

There is no server process to administer and no schema to design. The database is a NumPy array
of vectors, a JSON list of chunks, and a small JSON metadata file.

## Heading based chunking

A chunk is a heading line together with everything that follows it, up to the next heading of
any level. Headings from `#` to `######` all count as boundaries. The heading line stays inside
the chunk, so its words are part of what gets embedded and can be matched.

A file without headings becomes one single chunk. That is legal but rarely useful: the vector
of a long unstructured document is an average of everything it discusses, which is close to
every query and precisely on target for none of them. Give long documents headings, and let one
section cover one topic.

## How search works

Every chunk is converted into a vector, a list of numbers describing its meaning, at build time.
When you search, your query is converted with the same model, then compared against every chunk
vector by cosine similarity. The highest scoring chunks come back, each labelled with the file
it came from.

Nothing here is keyword matching. A passage can rank first without containing a single word of
the query, and a passage containing the exact phrase can rank low if the surrounding text pulls
its meaning elsewhere. That is the point: you can find a section whose wording you do not
remember.

## Why query wording matters

Because the whole query becomes one vector, every extra word shifts that vector. Adding the
broad topic of a document to a narrow question averages the two, and broad summary sections
start beating the specific section you wanted.

Search for the concept alone. If the answer does not appear, rephrase and search again rather
than adding context words.

## Query instructions

The embedding models this project targets are trained to take a short instruction on the query
side, describing what kind of passage you are looking for. Documents are embedded plain, without
one. The default instruction is generic on purpose; matching it to what your collection actually
contains tends to improve ranking.

## Rebuilding

The build always regenerates the whole database. There is no incremental update, and no
migration path between builds. Add or edit files in `Data/`, run the build again, and the old
database is replaced.

Switching the embedding model also requires a rebuild. Vectors produced by different models live
in different spaces and cannot be compared, so an old database with a new model returns
meaningless rankings rather than an error.

## Privacy

Documents and queries are sent to an Ollama instance over a local HTTP connection, by default
`http://localhost:11434`. Nothing leaves the machine unless you deliberately point
`SVDB_OLLAMA_URL` at a remote host.
