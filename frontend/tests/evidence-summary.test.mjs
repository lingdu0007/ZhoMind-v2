import assert from 'node:assert/strict';
import test from 'node:test';
import { getEvidenceCoverageLabel, getEvidenceSourceLabel } from '../src/app/evidence-summary.js';

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
