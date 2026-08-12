# Production Interviewer Walkthrough Evidence

## Provenance

This bundle records production-compose run `knowledge-walkthrough-20260812T051218600Z` at source_revision `42bc93989ed39ea25d811a9e51f45f0950860d57`. The run used generation model `deepseek-ai/DeepSeek-V4-Flash` and embedding model `Qwen/Qwen3-Embedding-8B`.

## Method

The authenticated product path performed a real Candidate Build and explicit publication, admitted a Knowledge User by Team Invitation, opened the Knowledge Map, and ran direct, paraphrased, and Boundary queries. Normal, SSE, and history projections were checked by persisted answer identity and identical Evidence Summary snapshots. The run also opened a Public Source Citation, submitted structured feedback, and verified deactivation.

## Results

All 12 normalized checks in `walkthrough.json` passed. Direct and paraphrased paths returned `evidence_gated_answer`; the unsupported Boundary query returned `insufficient_evidence_reply`. The production migration, dense indexing, generation contract, feedback persistence, and access revocation paths were active in the same observation window.

## Limits

- This is one bounded production-compose run over one reviewed Pilot entry, not full-corpus live acceptance.
- Independent model generations are compared by their Evidence Summary; each normal or SSE answer is compared byte-for-byte only with its own persisted history projection.
- Questions, answers, excerpts, user identities, credentials, hosts, and private corpus text are excluded.
