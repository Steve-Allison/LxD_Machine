# LxD Machine - Entity Matching

## 1. Core Strategy

Rules:

- use exact-string matching only in V1
- use `pyahocorasick` for runtime matching
- construct the matcher as `ahocorasick.Automaton(ahocorasick.STORE_ANY, ahocorasick.KEY_STRING)`
- require `canonical_id` on every entity definition
- treat `aliases` and `indicators` as optional lists defaulting to `[]`
- build the matcher from explicit matcher terms only: `canonical_id`, `aliases`, and `indicators`
- add all normalized matcher terms before calling `make_automaton()`
- finalize once with `make_automaton()` before any search
- run matching with `iter()` over normalized unicode strings
- do not persist pickled or saved automatons in V1; rebuild from ontology at startup because the upstream docs warn serialized formats are not safe for untrusted input
- do not use GLiNER, PyTorch, or any ML extraction runtime for the core system

## 2. Ontology Inputs

The matcher is built from the resolved ontology loaded from the full `Yamls` tree.

Only entity definitions contribute matcher terms.

Taxonomy, methodology, and shared YAML files influence the graph and snapshot hash, but they do not directly add matcher phrases unless those phrases are surfaced through entity definitions.

No-silent-loss rule:

- all resolved YAML fields must still be preserved through the ontology loader as structured metadata or graph input
- matcher selectivity does not justify dropping non-matcher YAML data; it only narrows which fields become Aho-Corasick terms

Matcher term-set contract:

- after normalization, emit one canonical record per matcher term: `entity_id`, `term_source`, `normalized_term`
- `term_source` is exactly one of: `canonical_id`, `alias`, `indicator`
- deduplicate identical canonical records before hashing or building the automaton
- compute `matcher_termset_hash` from the canonical record set before `make_automaton()`

## 3. Normalization

Normalize both matcher terms and scanned text with:

- casefolding
- map curly apostrophes and quotes to ASCII apostrophe `'`
- map Unicode dash variants to ASCII hyphen `-`
- collapse all Unicode whitespace runs to a single ASCII space
- do not generate singular/plural variants automatically; every accepted surface form must be explicit in ontology data
- normalize before both `add_word()` and `iter()`

## 4. Execution

Mention detection runs over committed searchable chunk text:

- markdown chunks
- Docling JSON chunks

PNG assets are not scanned for entity mentions in V1 because V1 does not perform OCR.

## 5. Output

Detected mentions populate the `mentions` table and support:

- query-time entity context
- `find_documents_for_concept` or similar later tools
- debugging and evaluation

Mention indexing must never block the first usable search build.

When mentions are present, they must be derived from the committed `ontology_snapshot` for the same run.

Matcher audit rule:

- mention rebuild must record the `matcher_termset_hash` it used
- if the rebuilt matcher term set does not reproduce the committed `matcher_termset_hash`, mention indexing must fail for that run

Overlap resolution policy:

- collect all `iter()` matches
- convert each match to `(start_char, end_char, entity_id, term_source, matched_text)`
- resolve overlapping spans by longest match first
- break equal-length ties by matcher term priority: `canonical_id` > `alias` > `indicator`
- break any remaining ties by ascending `start_char`, then ascending `entity_id`
