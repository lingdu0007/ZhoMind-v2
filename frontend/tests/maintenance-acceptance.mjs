import assert from 'node:assert/strict';

export const publicationAcceptance = async (request, api, token, entry, replayIdentity, blockingIdentity) => {
  const bundles = await request(api, '/reviewed-release-bundles', token);
  const imported = bundles.items.flatMap((bundle) => bundle.items).find((item) => item.entry_identity === `entry:${entry}`);
  assert.ok(imported?.job_id);
  const job = await request(api, `/reviewed-release-bundles/jobs/${imported.job_id}`, token);
  const published = await request(api, `/reviewed-release-bundles/candidates/${job.candidate_id}/publication`, token);
  const version = published.current_published_knowledge_version;
  assert.ok(version?.identity);
  const route = (await request(api, '/settings/generation-route', token)).active;
  const original = await request(api, `/acceptance/records/${route.providers[0].validation_evidence.record_identity}`, token);
  const payload = Object.fromEntries([
    'stage', 'affected_scope', 'content_identities', 'product_identities', 'conditions',
    'assumptions', 'checks', 'known_limits', 'risks', 'evidence_links', 'reacceptance_triggers'
  ].map((key) => [key, original[key]]));
  payload.candidate_publication_binding = {
    candidate_identity: version.candidate_id, published_knowledge_version_identity: version.identity,
    ...Object.fromEntries([
      'inspection_record_identity', 'acceptance_record_identity', 'entry_identity',
      'configuration_identity', 'bundle_sha256', 'frozen_input_sha256'
    ].map((key) => [key, version[key]]))
  };
  const identities = Object.entries(payload.candidate_publication_binding)
    .filter(([key]) => !key.endsWith('sha256')).map(([, value]) => value);
  payload.affected_scope.entry_identities = [version.entry_identity, version.identity];
  payload.affected_scope.expected_blocking_scope = 'entry_version';
  payload.affected_scope.blocking_scope_identity = blockingIdentity || version.identity;
  payload.affected_scope.configuration_identities.push(version.configuration_identity);
  payload.product_identities.push(version.configuration_identity);
  payload.content_identities = identities.filter((identity) => identity !== version.configuration_identity);
  const links = replayIdentity ? [`evidence://maintenance/artifacts/${replayIdentity}`] : ['evidence://maintenance/publication-containment'];
  payload.evidence_links.push(...links);
  for (const check_id of ['check:entry-supported-query', 'check:entry-boundary-query', 'check:evidence-citation-identity']) {
    const check = payload.checks.find((check) => check.check_id === check_id);
    assert.ok(check);
    check.evidence_links = links;
    check.identity_dependencies = identities;
  }
  const accepted = await request(api, '/acceptance/records', token, payload);
  await request(api, `/acceptance/records/${accepted.record_id}/status`, token, {
    status: 'active', reason_code: 'checks_verified',
    verified_checks: accepted.checks.filter((check) => ['passed', 'carried_forward'].includes(check.result))
      .map(({ check_id, evidence_links }) => ({ check_id, evidence_links }))
  });
  return { identity: accepted.record_id, publication: version.identity };
};
