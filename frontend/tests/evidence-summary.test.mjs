import assert from 'node:assert/strict';
import test from 'node:test';
import { getEvidenceCoverageLabel, getEvidenceSourceLabel, getEvidenceSourceUrl } from '../src/app/evidence-summary.js';

test('Evidence Summary maps only categorical coverage states', () => {
  assert.equal(getEvidenceCoverageLabel('sufficient'), '证据充分');
  assert.equal(getEvidenceCoverageLabel('insufficient'), '证据不足');
  assert.equal(getEvidenceCoverageLabel('unavailable'), '证据不可用');
});

test('Evidence Summary prefers returned source metadata before stable source identifiers', () => {
  assert.equal(
    getEvidenceSourceLabel({ source_id: 'chunk-7', metadata: { source_file: 'deploy-runbook.md' } }),
    'deploy-runbook.md'
  );
  assert.equal(getEvidenceSourceLabel({ source_id: 'chunk-7', metadata: {} }), 'chunk-7');
});

test('Public Source Citation prefers the public source title and permits only safe HTTPS URLs', () => {
  const citation = {
    citation_id: 'S1',
    entry_title: 'Prefer deterministic workflows',
    source_title: 'Building effective agents',
    source_url: 'https://www.anthropic.com/engineering/building-effective-agents'
  };

  assert.equal(getEvidenceSourceLabel(citation), 'Building effective agents');
  assert.equal(getEvidenceSourceUrl(citation), citation.source_url);
  assert.equal(getEvidenceSourceUrl({ source_url: 'http://example.com/source' }), '');
  assert.equal(getEvidenceSourceUrl({ source_url: 'https://user:secret@example.com/source' }), '');
  assert.equal(getEvidenceSourceUrl({ source_url: 'https://example.com/source?access_token=secret' }), '');
  assert.equal(getEvidenceSourceUrl({ source_url: 'https://example.com/source?redirect=https://evil.example' }), '');
});
