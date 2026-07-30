const coverageLabels = {
  sufficient: '证据充分',
  insufficient: '证据不足',
  unavailable: '证据不可用'
};

export const getEvidenceCoverageLabel = (coverage) => coverageLabels[coverage] || coverageLabels.unavailable;

export const getEvidenceSourceLabel = (source) => {
  const safeSource = source || {};
  const metadata = safeSource.metadata || {};
  return (
    metadata.filename ||
    metadata.source_file ||
    metadata.title ||
    metadata.document_name ||
    metadata.source ||
    metadata.path ||
    safeSource.source_id ||
    '未标记来源'
  );
};
