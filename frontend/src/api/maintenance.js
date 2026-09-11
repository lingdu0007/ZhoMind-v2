import http from './http';

const unwrap = (response) => response.data?.data ?? response.data;
const path = (identity) => `/maintenance/items/${encodeURIComponent(identity)}`;

export const maintenanceApi = {
  async editorialEntry(entryId) {
    return unwrap(await http.get(`/editorial/entries/${encodeURIComponent(entryId)}`));
  },
  async requestFreshnessReview(entryId, payload) {
    return unwrap(await http.post(`/editorial/entries/${encodeURIComponent(entryId)}/freshness-review`, payload));
  },
  async recordIntegrityReview(entryId, payload) {
    return unwrap(await http.post(`/editorial/entries/${encodeURIComponent(entryId)}/integrity-review`, payload));
  },
  async context() {
    return unwrap(await http.get('/maintenance/context'));
  },
  async items() {
    return unwrap(await http.get('/maintenance/items'));
  },
  async inbox() {
    return unwrap(await http.get('/maintenance/inbox'));
  },
  async assign(username) {
    return unwrap(await http.post('/maintenance/assignments', { username }));
  },
  async accept(identity) {
    return unwrap(await http.post(`/maintenance/assignments/${encodeURIComponent(identity)}/accept`));
  },
  async create(payload) {
    return unwrap(await http.post('/maintenance/items', payload));
  },
  async transition(identity, expectedRevision, state) {
    return unwrap(await http.post(`${path(identity)}/transition`, { expected_revision: expectedRevision, state }));
  },
  async joinAdministrator(identity, expectedRevision) {
    return unwrap(await http.post(`${path(identity)}/administrator`, { expected_revision: expectedRevision }));
  },
  async reproductionInputs(identity) {
    return unwrap(await http.get(`${path(identity)}/reproduction-inputs`));
  },
  async authorizeProviderVerification(identity, payload) {
    return unwrap(await http.post(`${path(identity)}/provider-verification-authorizations`, payload));
  },
  async providerVerificationAuthorization(identity) {
    return unwrap(await http.get(`/maintenance/provider-verification-authorizations/${encodeURIComponent(identity)}`));
  },
  async reproduce(identity, payload) {
    return unwrap(await http.post(`${path(identity)}/reproductions`, payload));
  },
  async fixture(identity) {
    return unwrap(await http.get(`/maintenance/fixtures/${encodeURIComponent(identity)}`));
  },
  async diagnose(identity, payload) {
    return unwrap(await http.post(`${path(identity)}/diagnosis`, payload));
  },
  async approveFinding(identity, payload) {
    return unwrap(await http.post(`${path(identity)}/findings`, payload));
  },
  async finding(identity) {
    return unwrap(await http.get(`/maintenance/findings/${encodeURIComponent(identity)}`));
  },
  async replay(identity, payload) {
    return unwrap(await http.post(`${path(identity)}/replays`, payload));
  },
  async replayResult(identity) {
    return unwrap(await http.get(`/maintenance/replays/${encodeURIComponent(identity)}`));
  },
  async resolve(identity, payload) {
    return unwrap(await http.post(`${path(identity)}/resolution`, payload));
  },
  async roadmap() {
    return unwrap(await http.get('/maintenance/roadmap'));
  },
  async qualifyRoadmap(identity, payload) {
    return unwrap(await http.post(`${path(identity)}/roadmap`, payload));
  },
  async reviewRoadmap(identity, payload) {
    return unwrap(await http.post(`/maintenance/roadmap/${encodeURIComponent(identity)}/review`, payload));
  },
  async map(identity) {
    return unwrap(await http.get(`/maintenance/maps/${encodeURIComponent(identity)}`));
  },
  async reviewContext() {
    return unwrap(await http.get('/maintenance/review-context'));
  },
  async dashboard() {
    return unwrap(await http.get('/maintenance/dashboard'));
  },
  async recordCadence(payload) {
    return unwrap(await http.post('/maintenance/cadence', payload));
  }
};
