import { hasSafeEvidenceSourceLocator } from './evidence-summary.js';

const completedOutcomePresentations = {
  evidence_gated_answer: {
    kind: 'supported',
    label: 'Supported by published knowledge',
    retryable: false
  },
  insufficient_evidence_reply: {
    kind: 'insufficient',
    label: 'Insufficient Evidence Reply',
    retryable: false
  },
  non_knowledge_base_reply: {
    kind: 'non-knowledge-base',
    label: 'Non-Knowledge-Base Reply',
    retryable: false
  },
  generation_unavailable: {
    kind: 'generation-unavailable',
    label: 'Generation Unavailable',
    retryable: true
  }
};

const terminalStatePresentations = {
  stopped: { kind: 'stopped', label: 'Stopped', retryable: true },
  failed: { kind: 'failed', label: 'Failed', retryable: true },
  throttled: { kind: 'throttled', label: 'Throttled', retryable: true },
  rejected: { kind: 'rejected', label: 'Rejected', retryable: true }
};

const insufficientEvidencePresentations = {
  no_eligible_published_evidence: {
    title: 'No eligible published evidence',
    detail: 'No access-appropriate published knowledge can support this request.'
  },
  decision_not_covered: {
    title: 'The requested decision is not covered',
    detail: 'Refine the decision, scope, or operating conditions to search published coverage.'
  },
  decisive_condition_missing: {
    title: 'A decisive condition is missing',
    detail: 'Add the version, environment, scale, target, or other required condition explicitly.'
  },
  material_evidence_conflict: {
    title: 'Published evidence has a material conflict',
    detail: 'Do not choose between conflicting material without reviewed knowledge.'
  },
  assurance_support_missing: {
    title: 'Required assurance support is missing',
    detail: 'The available material cannot meet the required assurance boundary.'
  },
  evidence_budget_exceeded: {
    title: 'Required evidence exceeds the bounded evidence set',
    detail: 'Narrow the request so a complete answer can remain within its frozen evidence boundary.'
  },
  knowledge_needs_review: {
    title: 'Published knowledge needs review',
    detail: 'The relevant published knowledge is not currently eligible to support a new answer.'
  }
};

const sectionDefinitions = new Map([
  ['Recommendation', { id: 'decision-summary', label: 'Decision Summary' }],
  ['suggestion', { id: 'decision-summary', label: 'Decision Summary' }],
  ['Suggestion', { id: 'decision-summary', label: 'Decision Summary' }],
  ['建议', { id: 'decision-summary', label: 'Decision Summary' }],
  ['Applicability Limits', { id: 'applicability-limits', label: 'Applicability Limits' }],
  ['适用边界', { id: 'applicability-limits', label: 'Applicability Limits' }],
  ['Alternatives', { id: 'alternatives', label: 'Alternatives' }],
  ['备选方案', { id: 'alternatives', label: 'Alternatives' }],
  [
    'Minimal Implementation or Acceptance Check',
    { id: 'minimum-check', label: 'Minimal Implementation or Acceptance Check' }
  ],
  ['最小实现或验收检查', { id: 'minimum-check', label: 'Minimal Implementation or Acceptance Check' }],
  [
    'Missing Conditions and Version Scope',
    { id: 'conditions-and-version-scope', label: 'Missing Conditions and Version Scope' }
  ],
  ['缺失条件与版本范围', { id: 'conditions-and-version-scope', label: 'Missing Conditions and Version Scope' }]
]);

const citationMarker = /\[(S[1-9][0-9]*)\]/g;
const frozenIdentity = /^[a-f0-9]{64}$/;
const supportedStatusMarker = '【Supported by published knowledge】';

export const getAnswerExecutionPresentation = (execution) => {
  const state = execution?.state;
  if (state === 'completed') {
    return (
      completedOutcomePresentations[execution?.outcome] || {
        kind: 'contract-failure',
        label: 'Closed result unavailable',
        retryable: false
      }
    );
  }
  if (terminalStatePresentations[state]) return terminalStatePresentations[state];
  if (state === 'admitted' || state === 'queued' || state === 'running') {
    return { kind: 'in-progress', label: 'In progress', retryable: false };
  }
  return { kind: 'unverified', label: 'Execution projection unavailable', retryable: false };
};

export const getInsufficientEvidencePresentation = (reason) =>
  insufficientEvidencePresentations[reason] || null;

const splitCitationMarkers = (line, citationIds) => {
  const fragments = [];
  let cursor = 0;
  for (const match of line.matchAll(citationMarker)) {
    const [marker, citationId] = match;
    if (!citationIds.has(citationId)) continue;
    if (match.index > cursor) {
      fragments.push({ type: 'text', value: line.slice(cursor, match.index) });
    }
    fragments.push({ type: 'citation', citationId });
    cursor = (match.index || 0) + marker.length;
  }
  if (cursor < line.length) {
    fragments.push({ type: 'text', value: line.slice(cursor) });
  }
  return fragments.length ? fragments : [{ type: 'text', value: line }];
};

export const parseFrozenDecisionAnswer = (answerText, citationIds = []) => {
  if (typeof answerText !== 'string' || !answerText.trim()) return [];
  const allowedCitations = new Set(citationIds.filter((value) => typeof value === 'string'));
  const sections = [];
  let current = null;
  let unknownSectionIndex = 0;

  for (const line of answerText.split(/\r?\n/)) {
    if (line.trim() === supportedStatusMarker) continue;
    const heading = line.match(/^##\s+(.+?)\s*$/);
    if (heading) {
      const headingText = heading[1].trim();
      const definition = sectionDefinitions.get(headingText);
      current = definition
        ? { ...definition, heading: headingText, blocks: [] }
        : {
            id: `frozen-section-${unknownSectionIndex++}`,
            label: headingText,
            heading: headingText,
            blocks: []
          };
      sections.push(current);
      continue;
    }
    if (!line.trim()) continue;
    if (!current) {
      current = { id: 'frozen-preamble', label: '', heading: '', blocks: [] };
      sections.push(current);
    }
    current.blocks.push(splitCitationMarkers(line, allowedCitations));
  }

  return sections;
};

const validFrozenCitationSource = (source) =>
  source &&
  typeof source.citation_id === 'string' &&
  /^S[1-9][0-9]*$/.test(source.citation_id) &&
  typeof source.citation_identity === 'string' &&
  frozenIdentity.test(source.citation_identity) &&
  typeof source.entry_id === 'string' &&
  source.entry_id &&
  typeof source.entry_title === 'string' &&
  source.entry_title &&
  typeof source.section_id === 'string' &&
  source.section_id &&
  typeof source.snapshot_id === 'string' &&
  frozenIdentity.test(source.snapshot_id) &&
  typeof source.publication_version === 'string' &&
  source.publication_version &&
  (typeof source.withdrawal_notice === 'string' && source.withdrawal_notice
    ? true
    : hasSafeEvidenceSourceLocator(source));

export const isRenderableFrozenSupportedAnswer = (_answerText, sources = []) => {
  if (!Array.isArray(sources) || !sources.length || sources.some((source) => !validFrozenCitationSource(source))) {
    return false;
  }

  const citationIds = sources.map((source) => source.citation_id);
  return (
    new Set(citationIds).size === citationIds.length &&
    sources.some(
      (source) => source.section_id === 'recommendation_or_reviewed_branches'
    )
  );
};
