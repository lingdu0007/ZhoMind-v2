const coverageLabels = {
  sufficient: '证据充分',
  insufficient: '证据不足',
  unavailable: '证据不可用'
};

const assuranceLabels = {
  source_grounded: 'Source-grounded assurance',
  claim_linked: 'Claim-linked assurance',
  release_assured: 'Release-assured assurance'
};

const controlledSourceLocator = /^controlled:\/\/[a-z0-9][a-z0-9._/-]{2,159}$/;
const unsafeLocatorParts = ['credential', 'password', 'redirect', 'secret', 'signature', 'token'];

export const getEvidenceCoverageLabel = (coverage) => coverageLabels[coverage] || coverageLabels.unavailable;
export const getKnowledgeAssuranceLabel = (assurance) => assuranceLabels[assurance] || '';

export const getEvidenceSourceLabel = (source) => {
  const safeSource = source || {};
  const metadata = safeSource.metadata || {};
  return (
    safeSource.source_title ||
    safeSource.entry_title ||
    metadata.filename ||
    metadata.source_file ||
    metadata.title ||
    metadata.document_name ||
    metadata.source ||
    metadata.path ||
    safeSource.citation_id ||
    safeSource.source_id ||
    '未标记来源'
  );
};

export const getEvidenceSourceUrl = (source) => {
  if (
    source?.source_access_scope === 'controlled_internal' ||
    (source?.source_access_scope && source.source_access_scope !== 'public')
  ) {
    return '';
  }
  const value = source?.source_url;
  if (typeof value !== 'string' || !value) return '';
  try {
    const parsed = new URL(value);
    const hostname = parsed.hostname.toLowerCase().replace(/\.$/, '');
    const unsafeQuery = [...parsed.searchParams.keys()].some((key) =>
      ['credential', 'password', 'redirect', 'secret', 'signature', 'token'].some((part) => key.toLowerCase().includes(part))
    );
    if (
      parsed.protocol !== 'https:' ||
      parsed.username ||
      parsed.password ||
      (parsed.port && parsed.port !== '443') ||
      !hostname.includes('.') ||
      hostname === 'localhost' ||
      hostname.endsWith('.local') ||
      hostname.endsWith('.internal') ||
      unsafeQuery
    ) {
      return '';
    }
    return value;
  } catch {
    return '';
  }
};

export const getControlledEvidenceSourceLocator = (source) => {
  const value = source?.source_url;
  if (
    source?.source_access_scope !== 'controlled_internal' ||
    typeof value !== 'string' ||
    !controlledSourceLocator.test(value) ||
    unsafeLocatorParts.some((part) => value.toLowerCase().includes(part))
  ) {
    return '';
  }
  return value;
};

export const hasSafeEvidenceSourceLocator = (source) => {
  if (source?.source_access_scope === 'controlled_internal') {
    return Boolean(getControlledEvidenceSourceLocator(source));
  }
  if (source?.source_access_scope && source.source_access_scope !== 'public') {
    return false;
  }
  return Boolean(getEvidenceSourceUrl(source));
};
