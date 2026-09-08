<template>
  <section class="reviewed-bundles" aria-labelledby="reviewed-bundles-title">
    <header class="reviewed-bundles__header">
      <div>
        <p class="reviewed-bundles__eyebrow">知识库运维</p>
        <h1 id="reviewed-bundles-title">Reviewed Release Bundles</h1>
        <p class="reviewed-bundles__description">导入已批准的 editorial export，并跟踪其独立 Candidate Build。</p>
      </div>
      <div v-if="isDesktop" class="reviewed-bundles__header-actions">
        <button type="button" :disabled="loading" @click="openImportDialog">
          <FileUp :size="16" aria-hidden="true" />
          <span>导入 Bundle</span>
        </button>
        <button type="button" :disabled="loading" @click="loadBundles">
          <RefreshCw :size="16" :class="{ 'reviewed-bundles__refresh-icon--spinning': loading }" aria-hidden="true" />
          <span>{{ loading ? '正在刷新' : '刷新' }}</span>
        </button>
        <button
          v-if="publicationSelections.length"
          type="button"
          :disabled="publicationLoading"
          @click="openPublicationConfirmation"
        >
          <Send :size="16" aria-hidden="true" />
          <span>发布已选择 {{ publicationSelections.length }} 项</span>
        </button>
      </div>
    </header>

    <div v-if="!isDesktop" class="reviewed-bundles__desktop-notice" role="status">
      <Monitor :size="18" aria-hidden="true" />
      <p>Reviewed Release Bundle 管理当前仅支持桌面工作区。</p>
    </div>

    <template v-else>
      <p v-if="listError" class="reviewed-bundles__error" role="alert">
        <span>{{ listError }}</span>
        <button type="button" @click="loadBundles">重新加载</button>
      </p>
      <p
        v-if="actionMessage"
        class="reviewed-bundles__success"
        :class="{ 'reviewed-bundles__partial': publicationResults && !publicationResults.batch_complete }"
        role="status"
      >{{ actionMessage }}</p>
      <p v-if="actionError" class="reviewed-bundles__error" role="alert">{{ actionError }}</p>

      <div class="reviewed-bundles__workspace" :aria-busy="loading">
        <section class="reviewed-bundles__inventory" aria-labelledby="reviewed-bundle-list-title">
          <header class="reviewed-bundles__section-header">
            <h2 id="reviewed-bundle-list-title">已导入 Bundle</h2>
            <p aria-live="polite">{{ bundles.length }} 个 Bundle</p>
          </header>

          <div class="reviewed-bundles__table-wrap">
            <table>
              <caption class="sr-only">Reviewed Release Bundle 列表</caption>
              <thead>
                <tr>
                  <th scope="col">Bundle</th>
                  <th scope="col">Editorial revision</th>
                  <th scope="col">导出时间</th>
                  <th scope="col">项目</th>
                </tr>
              </thead>
              <tbody>
                <tr v-if="loading && !bundles.length">
                  <td colspan="4" class="reviewed-bundles__state">正在加载 Bundle...</td>
                </tr>
                <tr v-else-if="!bundles.length">
                  <td colspan="4" class="reviewed-bundles__state">尚未导入 Reviewed Release Bundle。</td>
                </tr>
                <tr
                  v-for="bundle in bundles"
                  :key="bundle.bundle_id"
                  :class="{ 'reviewed-bundles__row--selected': selectedBundleId === bundle.bundle_id }"
                >
                  <td>
                    <button
                      type="button"
                      class="reviewed-bundles__bundle-select"
                      :aria-pressed="selectedBundleId === bundle.bundle_id"
                      @click="selectBundle(bundle.bundle_id)"
                    >
                      {{ bundle.bundle_id }}
                    </button>
                  </td>
                  <td class="reviewed-bundles__identifier">{{ bundle.editorial_source_revision || '-' }}</td>
                  <td class="reviewed-bundles__timestamp">{{ formatTime(bundle.exported_at) }}</td>
                  <td>{{ bundle.items?.length || 0 }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        <section class="reviewed-bundles__detail" aria-labelledby="reviewed-bundle-detail-title">
          <template v-if="selectedBundle">
            <header class="reviewed-bundles__section-header">
              <div>
                <p class="reviewed-bundles__detail-eyebrow">不可变导入记录</p>
                <h2 id="reviewed-bundle-detail-title">{{ selectedBundle.bundle_id }}</h2>
              </div>
              <span class="reviewed-bundles__status" :class="`reviewed-bundles__status--${bundleStateTone(selectedBundle.state)}`">
                {{ bundleStateLabel(selectedBundle.state) }}
              </span>
            </header>

            <dl class="reviewed-bundles__facts">
              <div>
                <dt>Schema version</dt>
                <dd>{{ selectedBundle.schema_version || '-' }}</dd>
              </div>
              <div>
                <dt>Editorial source revision</dt>
                <dd class="reviewed-bundles__identifier">{{ selectedBundle.editorial_source_revision || '-' }}</dd>
              </div>
              <div>
                <dt>导出时间</dt>
                <dd>{{ formatTime(selectedBundle.exported_at) }}</dd>
              </div>
              <div>
                <dt>Bundle SHA-256</dt>
                <dd class="reviewed-bundles__identifier">{{ selectedBundle.bundle_sha256 || '-' }}</dd>
              </div>
            </dl>

            <div class="reviewed-bundles__item-table-wrap">
              <table>
                <caption class="sr-only">Bundle 项目和 Candidate Build 状态</caption>
                <thead>
                  <tr>
                    <th scope="col">项目</th>
                    <th scope="col">操作</th>
                    <th scope="col">状态</th>
                    <th scope="col">输入 SHA-256</th>
                    <th scope="col">Candidate Build</th>
                    <th scope="col">允许的下一步</th>
                    <th scope="col"><span class="sr-only">任务操作</span></th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="item in selectedBundle.items" :key="item.bundle_item_id">
                    <td>
                      <p class="reviewed-bundles__identifier">{{ item.bundle_item_id }}</p>
                      <p class="reviewed-bundles__entry">{{ item.entry_identity }}</p>
                    </td>
                    <td>{{ operationLabel(item.operation) }}</td>
                    <td>
                      <span class="reviewed-bundles__status" :class="`reviewed-bundles__status--${itemStateTone(item.state)}`">
                        {{ itemStateLabel(item.state) }}
                      </span>
                      <p v-if="item.failure_reason" class="reviewed-bundles__failure">
                        {{ item.failure_reason.field }}: {{ item.failure_reason.message }}
                      </p>
                    </td>
                    <td class="reviewed-bundles__identifier">
                      <p>{{ item.artifact_sha256 || '-' }}</p>
                      <p class="reviewed-bundles__item-hash">{{ item.bundle_item_sha256 || '-' }}</p>
                    </td>
                    <td>
                      <template v-if="item.job_id">
                        <p class="reviewed-bundles__identifier">{{ item.job_id }}</p>
                        <p v-if="jobFor(item)" class="reviewed-bundles__job-line">
                          {{ jobStatusLabel(jobFor(item).status, jobFor(item).stage) }} · {{ jobFor(item).progress }}% · {{ jobFor(item).attempt }} 次
                        </p>
                        <p v-if="jobFor(item)?.failure_reason" class="reviewed-bundles__failure">
                          {{ jobFor(item).failure_reason.code }}: {{ jobFor(item).failure_reason.message }}
                        </p>
                        <p v-if="jobFor(item)?.derived_cleanup_pending" class="reviewed-bundles__failure">
                          派生数据仍待协调
                        </p>
                        <p v-if="jobFor(item)" class="reviewed-bundles__job-time">
                          更新于 {{ formatTime(jobFor(item).updated_at) }}
                        </p>
                      </template>
                      <span v-else>-</span>
                    </td>
                    <td class="reviewed-bundles__next-action">{{ nextActionLabel(nextActionFor(item)) }}</td>
                    <td class="reviewed-bundles__actions">
                      <button
                        v-if="canDispatch(item)"
                        type="button"
                        :disabled="Boolean(actionLoading[item.job_id])"
                        :aria-label="`开始 Candidate Build ${item.job_id}`"
                        title="开始 Candidate Build"
                        @click="dispatchJob(item.job_id)"
                      >
                        <Play :size="16" aria-hidden="true" />
                      </button>
                      <button
                        v-if="canRetry(item)"
                        type="button"
                        :disabled="Boolean(actionLoading[item.job_id])"
                        :aria-label="`重试 Candidate Build ${item.job_id}`"
                        title="重试 Candidate Build"
                        @click="retryJob(item.job_id)"
                      >
                        <RotateCcw :size="16" aria-hidden="true" />
                      </button>
                      <button
                        v-if="canCancel(item)"
                        type="button"
                        class="reviewed-bundles__cancel"
                        :disabled="Boolean(actionLoading[item.job_id])"
                        :aria-label="`取消 Candidate Build ${item.job_id}`"
                        title="取消 Candidate Build"
                        @click="cancelJob(item.job_id)"
                      >
                        <XCircle :size="16" aria-hidden="true" />
                      </button>
                      <button
                        v-if="jobFor(item)?.candidate_id"
                        type="button"
                        :disabled="Boolean(candidateLoading[jobFor(item).candidate_id])"
                        :aria-label="`查看 Candidate ${jobFor(item).candidate_id}`"
                        title="查看 Candidate"
                        @click="viewCandidate(jobFor(item).candidate_id)"
                      >
                        <Eye :size="16" aria-hidden="true" />
                      </button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <section
              v-if="selectedCandidateDetail"
              class="reviewed-bundles__candidate-panel"
              aria-labelledby="reviewed-candidate-title"
            >
              <header class="reviewed-bundles__section-header">
                <div>
                  <p class="reviewed-bundles__detail-eyebrow">Candidate inspection</p>
                  <h2 id="reviewed-candidate-title">{{ selectedCandidateDetail.candidate.candidate_id }}</h2>
                </div>
                <div class="reviewed-bundles__candidate-actions">
                  <button
                    type="button"
                    :disabled="Boolean(candidateLoading[selectedCandidateDetail.candidate.candidate_id])"
                    :aria-label="`刷新 Candidate ${selectedCandidateDetail.candidate.candidate_id}`"
                    title="刷新 Candidate"
                    @click="viewCandidate(selectedCandidateDetail.candidate.candidate_id)"
                  >
                    <RefreshCw
                      :size="16"
                      :class="{ 'reviewed-bundles__refresh-icon--spinning': Boolean(candidateLoading[selectedCandidateDetail.candidate.candidate_id]) }"
                      aria-hidden="true"
                    />
                  </button>
                  <button
                    type="button"
                    :disabled="Boolean(candidateLoading[selectedCandidateDetail.candidate.candidate_id])"
                    :aria-label="`记录 Candidate inspection ${selectedCandidateDetail.candidate.candidate_id}`"
                    title="记录 Candidate inspection"
                    @click="recordCandidateInspection(selectedCandidateDetail.candidate.candidate_id)"
                  >
                    <ClipboardCheck :size="16" aria-hidden="true" />
                  </button>
                  <button
                    v-if="selectedCandidateDetail.inspection"
                    type="button"
                    :disabled="Boolean(candidateLoading[selectedCandidateDetail.candidate.candidate_id])"
                    :aria-label="`执行 Candidate 验收 ${selectedCandidateDetail.candidate.candidate_id}`"
                    title="执行 Candidate 验收"
                    @click="acceptCandidate(selectedCandidateDetail.candidate.candidate_id)"
                  >
                    <ShieldCheck :size="16" aria-hidden="true" />
                  </button>
                </div>
              </header>

              <dl class="reviewed-bundles__facts reviewed-bundles__candidate-facts">
                <div v-for="[field, label] in candidateIdentityFields" :key="field">
                  <dt>{{ label }}</dt>
                  <dd class="reviewed-bundles__identifier">{{ selectedCandidateDetail.candidate[field] || '-' }}</dd>
                </div>
                <div>
                  <dt>Candidate generation</dt>
                  <dd>{{ selectedCandidateDetail.candidate.generation }}</dd>
                </div>
                <div>
                  <dt>Configuration</dt>
                  <dd class="reviewed-bundles__identifier">{{ selectedCandidateDetail.candidate.configuration_identity }}</dd>
                </div>
                <div class="reviewed-bundles__candidate-configuration">
                  <dt>Effective configuration</dt>
                  <dd><pre>{{ formatStructured(selectedCandidateDetail.candidate.configuration) }}</pre></dd>
                </div>
                <div>
                  <dt>Inspection</dt>
                  <dd class="reviewed-bundles__identifier">{{ selectedCandidateDetail.inspection?.record_identity || '尚未记录' }}</dd>
                </div>
                <div>
                  <dt>Publication eligibility</dt>
                  <dd>{{ selectedCandidateEligibility?.eligible ? '可发布' : '尚未满足' }}</dd>
                </div>
              </dl>

              <details v-if="selectedCandidateDetail.candidate.metadata" class="reviewed-bundles__record-detail">
                <summary>Entry metadata</summary>
                <pre>{{ formatStructured(selectedCandidateDetail.candidate.metadata) }}</pre>
              </details>
              <details v-if="selectedCandidateDetail.inspection" class="reviewed-bundles__record-detail">
                <summary>Inspection 绑定</summary>
                <pre>{{ formatStructured(selectedCandidateDetail.inspection) }}</pre>
              </details>

              <section
                v-if="selectedCandidateDetail.acceptance"
                class="reviewed-bundles__acceptance"
                aria-label="Candidate 验收记录"
              >
                <p class="reviewed-bundles__detail-eyebrow">Candidate 验收记录</p>
                <p class="reviewed-bundles__identifier">{{ selectedCandidateDetail.acceptance.record_identity }}</p>
                <dl class="reviewed-bundles__facts">
                  <div><dt>Supported outcome</dt><dd>{{ selectedCandidateDetail.acceptance.supported.outcome }}</dd></div>
                  <div><dt>Boundary outcome</dt><dd>{{ selectedCandidateDetail.acceptance.boundary.outcome }}</dd></div>
                  <div>
                    <dt>Governing entry</dt>
                    <dd>{{ selectedCandidateDetail.acceptance.supported.expected_governing_entry_identity }}</dd>
                  </div>
                  <div>
                    <dt>Governing section</dt>
                    <dd>{{ selectedCandidateDetail.acceptance.supported.expected_governing_section_id }}</dd>
                  </div>
                  <div>
                    <dt>Answer Evidence Set</dt>
                    <dd>{{ selectedCandidateDetail.acceptance.supported.answer_evidence_set.identity }}</dd>
                  </div>
                  <div>
                    <dt>Citation markers</dt>
                    <dd>{{ selectedCandidateDetail.acceptance.supported.citation_markers.join(', ') }}</dd>
                  </div>
                  <div><dt>Boundary reason</dt><dd>{{ selectedCandidateDetail.acceptance.boundary.reason }}</dd></div>
                  <div>
                    <dt>Boundary provider calls / citations</dt>
                    <dd>{{ selectedCandidateDetail.acceptance.boundary.provider_call_count }} / {{ selectedCandidateDetail.acceptance.boundary.citation_markers.length }}</dd>
                  </div>
                </dl>
                <details class="reviewed-bundles__record-detail">
                  <summary>Evidence snapshots 与精确验收绑定</summary>
                  <pre>{{ formatStructured(selectedCandidateDetail.acceptance) }}</pre>
                </details>
              </section>

              <div class="reviewed-bundles__replacement">
                <div>
                  <p class="reviewed-bundles__detail-eyebrow">Replacement</p>
                  <p class="reviewed-bundles__replacement-title">
                    {{ selectedCandidateDetail.replacement.effect === 'replace' ? '替换当前 Published Knowledge Version' : '创建新的 Published Knowledge Version' }}
                  </p>
                  <p v-if="selectedCandidateDetail.replacement.current_published_knowledge_version" class="reviewed-bundles__identifier">
                    {{ selectedCandidateDetail.replacement.current_published_knowledge_version.identity }}
                  </p>
                  <details v-if="selectedCandidateDetail.replacement.current_published_knowledge_version" class="reviewed-bundles__record-detail">
                    <summary>Published version 绑定</summary>
                    <pre>{{ formatStructured(selectedCandidateDetail.replacement.current_published_knowledge_version) }}</pre>
                  </details>
                </div>
                <dl class="reviewed-bundles__diff">
                  <div>
                    <dt>新增 chunk</dt>
                    <dd>{{ selectedCandidateDetail.replacement.diff.added.length }}</dd>
                  </div>
                  <div>
                    <dt>变更 chunk</dt>
                    <dd>{{ selectedCandidateDetail.replacement.diff.changed.length }}</dd>
                  </div>
                  <div>
                    <dt>移除 chunk</dt>
                    <dd>{{ selectedCandidateDetail.replacement.diff.removed.length }}</dd>
                  </div>
                </dl>
                <div v-if="selectedCandidateDetail.replacement.diff.added.length" class="reviewed-bundles__replacement-content">
                  <div v-for="item in selectedCandidateDetail.replacement.diff.added" :key="`added-${item.chunk_index}`">
                    <p>Chunk {{ item.chunk_index }}：新增 Candidate 内容</p>
                    <pre>{{ item.candidate_content || selectedCandidateDetail.candidate.chunks.find((chunk) => chunk.chunk_index === item.chunk_index)?.content }}</pre>
                  </div>
                </div>
                <div v-if="selectedCandidateDetail.replacement.diff.changed.length" class="reviewed-bundles__replacement-content">
                  <div v-for="item in selectedCandidateDetail.replacement.diff.changed" :key="`changed-${item.chunk_index}`">
                    <p>Chunk {{ item.chunk_index }}：当前已发布内容</p>
                    <pre>{{ item.published_content }}</pre>
                    <p>Chunk {{ item.chunk_index }}：Candidate 内容</p>
                    <pre>{{ item.candidate_content }}</pre>
                  </div>
                </div>
                <div v-if="selectedCandidateDetail.replacement.diff.removed.length" class="reviewed-bundles__replacement-content">
                  <div v-for="item in selectedCandidateDetail.replacement.diff.removed" :key="`removed-${item.chunk_index}`">
                    <p>Chunk {{ item.chunk_index }}：将移除的已发布内容</p>
                    <pre>{{ item.published_content }}</pre>
                  </div>
                </div>
              </div>

              <div v-if="selectedCandidateEligibility?.reasons?.length" class="reviewed-bundles__candidate-reasons">
                {{ selectedCandidateEligibility.reasons.join('；') }}
              </div>

              <label
                v-if="selectedCandidateEligibility?.eligible || isPublicationSelected(selectedCandidateDetail.candidate.candidate_id)"
                class="reviewed-bundles__publication-choice"
              >
                <input
                  type="checkbox"
                  :checked="isPublicationSelected(selectedCandidateDetail.candidate.candidate_id)"
                  :disabled="!selectedCandidateEligibility?.eligible && !isPublicationSelected(selectedCandidateDetail.candidate.candidate_id)"
                  @change="togglePublicationSelection(selectedCandidateDetail.candidate.candidate_id)"
                />
                <span>选择或撤销此 Candidate 的显式批量发布</span>
              </label>

              <div class="reviewed-bundles__candidate-chunks">
                <p class="reviewed-bundles__detail-eyebrow">Candidate chunks</p>
                <article
                  v-for="chunk in selectedCandidateDetail.candidate.chunks"
                  :key="chunk.chunk_id"
                  class="reviewed-bundles__candidate-chunk"
                >
                  <header>
                    <span>Chunk {{ chunk.chunk_index + 1 }} · {{ chunk.metadata?.section_id || 'unknown section' }}</span>
                    <code>{{ chunk.content_sha256 }}</code>
                  </header>
                  <dl class="reviewed-bundles__chunk-facts">
                    <div>
                      <dt>Strategy</dt>
                      <dd>{{ chunk.metadata?.chunk_strategy_id || '-' }}</dd>
                    </div>
                    <div>
                      <dt>Sources</dt>
                      <dd>{{ (chunk.metadata?.source_identities || []).join(', ') || '-' }}</dd>
                    </div>
                  </dl>
                  <pre>{{ chunk.content }}</pre>
                  <pre class="reviewed-bundles__chunk-metadata">{{ formatStructured(chunk.metadata) }}</pre>
                </article>
              </div>
            </section>
          </template>

          <p v-else class="reviewed-bundles__detail-empty" role="status">选择一个 Bundle 以查看其不可变输入和任务状态。</p>
        </section>
      </div>
    </template>

    <el-dialog
      v-model="importDialogVisible"
      class="reviewed-bundles__import-dialog"
      width="min(92vw, 780px)"
      :close-on-click-modal="false"
      @closed="resetImport"
    >
      <template #header>
        <div>
          <p class="reviewed-bundles__detail-eyebrow">仅接受已批准的 editorial export</p>
          <h2>导入 Reviewed Release Bundle</h2>
        </div>
      </template>

      <div class="reviewed-bundles__import-form">
        <div class="reviewed-bundles__import-tools">
          <input ref="bundleFileInput" class="sr-only" type="file" accept="application/json,.json" @change="readBundleFile" />
          <button type="button" :disabled="importLoading" @click="bundleFileInput?.click()">
            <FileText :size="16" aria-hidden="true" />
            <span>选择 JSON 文件</span>
          </button>
          <span v-if="selectedFilename" class="reviewed-bundles__selected-file">{{ selectedFilename }}</span>
        </div>
        <label for="reviewed-bundle-manifest">
          <span>Bundle manifest</span>
          <textarea
            id="reviewed-bundle-manifest"
            v-model="manifestText"
            :disabled="importLoading"
            spellcheck="false"
            placeholder="粘贴 reviewed_release_bundle/v1 JSON"
          />
        </label>
        <p v-if="importError" class="reviewed-bundles__error" role="alert">{{ importError }}</p>
      </div>

      <template #footer>
        <button type="button" class="reviewed-bundles__dialog-button" :disabled="importLoading" @click="importDialogVisible = false">
          取消
        </button>
        <button
          type="button"
          class="reviewed-bundles__dialog-button reviewed-bundles__dialog-button--primary"
          :disabled="importLoading || !manifestText.trim()"
          @click="importBundle"
        >
          <FileUp v-if="!importLoading" :size="16" aria-hidden="true" />
          <RefreshCw v-else :size="16" class="reviewed-bundles__refresh-icon--spinning" aria-hidden="true" />
          <span>{{ importLoading ? '正在导入' : '导入 Bundle' }}</span>
        </button>
      </template>
    </el-dialog>

    <el-dialog
      v-model="publicationConfirmationVisible"
      class="reviewed-bundles__publication-dialog"
      width="min(92vw, 720px)"
      :close-on-click-modal="false"
      @closed="publicationConfirmationAcknowledged = false"
    >
      <template #header>
        <div>
          <p class="reviewed-bundles__detail-eyebrow">Explicit selected batch</p>
          <h2>确认发布 Candidate</h2>
        </div>
      </template>

      <div class="reviewed-bundles__publication-confirmation">
        <ul>
          <li v-for="item in publicationConfirmationSelections" :key="item.candidate_id">
            <code>{{ item.candidate_id }}</code>
            <span>{{ item.effect === 'replace' ? '替换' : '创建' }}</span>
            <code v-if="item.current_published_knowledge_version">{{ item.current_published_knowledge_version }}</code>
            <dl class="reviewed-bundles__confirmation-bindings">
              <div><dt>Inspection</dt><dd>{{ item.inspection_record_identity }}</dd></div>
              <div><dt>Acceptance</dt><dd>{{ item.acceptance_record_identity }}</dd></div>
            </dl>
          </li>
        </ul>
        <label class="reviewed-bundles__publication-choice">
          <input v-model="publicationConfirmationAcknowledged" type="checkbox" />
          <span>我确认发布以上精确选择项。</span>
        </label>
        <p
          v-if="publicationResults"
          class="reviewed-bundles__publication-outcome"
          :class="{ 'reviewed-bundles__partial': !publicationResults.batch_complete }"
          role="status"
        >
          已发布 {{ publicationResults.published.length }} 项，失败 {{ publicationResults.failed.length }} 项，跳过 {{ publicationResults.skipped.length }} 项。
        </p>
        <ul v-if="publicationResults" class="reviewed-bundles__publication-result-list">
          <li v-for="item in publicationResults.published" :key="`published-${item.candidate_id}`">
            <code>{{ item.candidate_id }}</code><span>已发布</span><code>{{ item.publication_identity }}</code>
          </li>
          <li v-for="item in publicationResults.failed" :key="`failed-${item.candidate_id}`">
            <code>{{ item.candidate_id }}</code><span>失败</span><code>{{ item.reason }}</code>
          </li>
          <li v-for="item in publicationResults.skipped" :key="`skipped-${item.candidate_id}`">
            <code>{{ item.candidate_id }}</code><span>跳过</span><code>{{ item.reason }}</code>
          </li>
        </ul>
      </div>

      <template #footer>
        <button type="button" class="reviewed-bundles__dialog-button" :disabled="publicationLoading" @click="publicationConfirmationVisible = false">
          取消
        </button>
        <button
          type="button"
          class="reviewed-bundles__dialog-button reviewed-bundles__dialog-button--primary"
          :disabled="publicationLoading || !publicationConfirmationAcknowledged || !publicationConfirmationSelections.length"
          @click="confirmPublication"
        >
          <Send v-if="!publicationLoading" :size="16" aria-hidden="true" />
          <RefreshCw v-else :size="16" class="reviewed-bundles__refresh-icon--spinning" aria-hidden="true" />
          <span>{{ publicationLoading ? '正在发布' : '确认发布' }}</span>
        </button>
      </template>
    </el-dialog>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import {
  ClipboardCheck,
  Eye,
  FileText,
  FileUp,
  Monitor,
  Play,
  RefreshCw,
  RotateCcw,
  Send,
  ShieldCheck,
  XCircle
} from 'lucide-vue-next';
import { apiAdapter } from '../api/adapters';

const POLL_DELAY_MS = 1000;
const candidateIdentityFields = [
  ['entry_identity', 'Entry'],
  ['document_identity', 'Document'],
  ['bundle_id', 'Candidate bundle'],
  ['bundle_item_id', 'Candidate bundle item'],
  ['editorial_source_revision', 'Candidate source revision'],
  ['bundle_sha256', 'Candidate bundle SHA-256'],
  ['bundle_item_sha256', 'Candidate item SHA-256'],
  ['input_sha256', 'Input SHA-256'],
  ['frozen_input_sha256', 'Frozen input SHA-256']
];

const bundles = ref([]);
const selectedBundleId = ref('');
const selectedBundle = ref(null);
const jobsById = ref({});
const loading = ref(false);
const listError = ref('');
const actionMessage = ref('');
const actionError = ref('');
const actionLoading = ref({});
const isDesktop = ref(true);
const isActive = ref(true);
const importDialogVisible = ref(false);
const manifestText = ref('');
const importLoading = ref(false);
const importError = ref('');
const selectedFilename = ref('');
const bundleFileInput = ref(null);
const selectedCandidateDetail = ref(null);
const selectedCandidateEligibility = ref(null);
const candidateLoading = ref({});
const publicationSelectionByCandidate = ref({});
const publicationConfirmationVisible = ref(false);
const publicationConfirmationAcknowledged = ref(false);
const activePublicationConfirmationId = ref(null);
const activePublicationConfirmationSelections = ref(null);
const publicationLoading = ref(false);
const publicationResults = ref(null);
let pollTimer = null;
let candidateSelectionEpoch = 0;

const bundleStateLabel = (state) =>
  ({
    received: '已接收',
    validating: '验证中',
    validated: '已验证',
    processing: '处理中',
    completed: '已完成',
    completed_with_rejections: '已完成，存在拒绝项'
  })[state] || state || '-';
const bundleStateTone = (state) => {
  if (state === 'completed') return 'success';
  if (state === 'completed_with_rejections') return 'danger';
  return 'neutral';
};
const operationLabel = (operation) =>
  ({
    create: '新建 (create)',
    replace: '替换计划 (replace)',
    no_op: '无变更 (no-op)',
    proposed_withdrawal: '撤回提案'
  })[operation] || operation || '-';
const itemStateLabel = (state) =>
  ({
    admitted: '已接纳',
    rejected: '已拒绝',
    no_op: '无变更',
    proposed_withdrawal: '撤回提案'
  })[state] || state || '-';
const itemStateTone = (state) => {
  if (state === 'admitted') return 'success';
  if (state === 'rejected') return 'danger';
  return 'neutral';
};
const jobStageLabel = (stage) =>
  ({
    queued: '排队中',
    parsing: '解析中',
    chunking: '分块中',
    indexing: '索引中'
  })[stage] || stage || '-';
const jobStatusLabel = (status, stage) =>
  ({
    candidate_ready: 'Candidate 已就绪',
    failed: '失败',
    canceled: '已取消',
    interrupted_retryable: '已中断，可重试',
    superseded: '已 supersede'
  })[status] || jobStageLabel(stage);
const nextActionLabel = (action) =>
  ({
    dispatch_candidate_build: '开始 Candidate Build',
    await_candidate_build: '等待构建',
    cancel_or_await_candidate_build: '等待或取消',
    await_candidate_inspection: '等待后续 inspection',
    await_candidate_acceptance: '等待 Candidate 验收',
    await_explicit_publication: '等待显式发布确认',
    published: '已发布',
    correct_item_in_new_bundle: '在新 Bundle 中修正',
    review_explicit_no_op: '复核无变更',
    requires_t04_publication_workflow: '等待后续 publication workflow',
    retry_fixed_inputs: '可用固定输入重试',
    reconcile_derived_data_then_retry: '先协调派生数据',
    reconcile_derived_data: '先协调派生数据',
    import_new_bundle: '需要新 Bundle',
    none: '无可执行操作'
  })[action] || action || '-';

const formatTime = (value) => {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date
    .toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    })
    .replaceAll('/', '-');
};

const structuredReasonMessage = (detail) => {
  if (!Array.isArray(detail?.reasons)) return '';
  return detail.reasons
    .slice(0, 4)
    .map((reason) => {
      if (!reason || typeof reason !== 'object') return '';
      const field = typeof reason.field === 'string' ? reason.field : '';
      const message = typeof reason.message === 'string' ? reason.message : '';
      return field && message ? `${field}: ${message}` : message || field;
    })
    .filter(Boolean)
    .join('；');
};

const friendlyError = (error, fallback) => {
  if (error?.status === 401) return '登录状态已失效，请重新登录。';
  if (error?.status === 403) return '当前账户无权管理 Reviewed Release Bundle。';
  const structuredReason = structuredReasonMessage(error?.detail);
  if (structuredReason) return structuredReason;
  if (error?.message && !/^Request failed with status code \d+$/.test(error.message)) return error.message;
  return fallback;
};

const jobFor = (item) => (item?.job_id ? jobsById.value[item.job_id] : null);
const nextActionFor = (item) => jobFor(item)?.allowed_next_action || item?.allowed_next_action;
const canDispatch = (item) =>
  jobFor(item)?.status === 'queued' && jobFor(item)?.allowed_next_action === 'dispatch_candidate_build';
const canRetry = (item) =>
  ['retry_fixed_inputs', 'reconcile_derived_data_then_retry'].includes(jobFor(item)?.allowed_next_action);
const canCancel = (item) => jobFor(item)?.allowed_next_action === 'cancel_or_await_candidate_build';
const hasActiveSelectedJob = computed(
  () => selectedBundle.value?.items?.some((item) => canCancel(item) && !canDispatch(item)) || false
);
const publicationSelections = computed(() => Object.values(publicationSelectionByCandidate.value));
const publicationConfirmationSelections = computed(
  () => activePublicationConfirmationSelections.value || publicationSelections.value
);
const formatStructured = (value) => JSON.stringify(value || {}, null, 2);

const mergeJob = (job) => {
  if (job?.job_id) jobsById.value = { ...jobsById.value, [job.job_id]: job };
};

const applyActionResult = (job, successMessage, failureFallback) => {
  mergeJob(job);
  if (job?.status === 'failed') {
    const reason = job.failure_reason;
    const detail = reason?.code && reason?.message ? `${reason.code}: ${reason.message}` : reason?.message || '';
    actionError.value = detail ? `${failureFallback} ${detail}` : failureFallback;
    return false;
  }
  actionMessage.value = successMessage;
  return true;
};

const beginCandidateSelection = () => {
  const selectionEpoch = ++candidateSelectionEpoch;
  selectedCandidateDetail.value = null;
  selectedCandidateEligibility.value = null;
  return selectionEpoch;
};

const loadCandidateSelection = async (candidateId, selectionEpoch) => {
  const detail = await apiAdapter.getReviewedCandidateInspection(candidateId);
  let eligibility = null;
  try {
    eligibility = await apiAdapter.getReviewedCandidatePublicationEligibility(candidateId);
  } catch (_error) {
    // Inspection remains useful when current eligibility cannot be read.
  }
  if (
    selectionEpoch !== candidateSelectionEpoch ||
    detail?.candidate?.candidate_id !== candidateId ||
    (eligibility && eligibility.candidate_id !== candidateId)
  ) {
    return false;
  }
  selectedCandidateDetail.value = detail;
  selectedCandidateEligibility.value = eligibility;
  return true;
};

const refreshSelectedBundle = async () => {
  const bundleId = selectedBundleId.value;
  if (!bundleId) return;
  const refreshed = await apiAdapter.getReviewedReleaseBundle(bundleId);
  if (selectedBundleId.value === bundleId) selectedBundle.value = refreshed;
};

const loadSelectedJobs = async () => {
  if (!selectedBundle.value?.items?.length) return;
  const jobs = await Promise.all(
    selectedBundle.value.items
      .map((item) => item.job_id)
      .filter(Boolean)
      .map((jobId) => apiAdapter.getReviewedBundleJob(jobId))
  );
  jobs.forEach(mergeJob);
  await refreshSelectedBundle();
};

const schedulePoll = () => {
  if (pollTimer) clearTimeout(pollTimer);
  if (!isActive.value || !isDesktop.value || !hasActiveSelectedJob.value) return;
  pollTimer = setTimeout(async () => {
    try {
      await loadSelectedJobs();
    } catch {
      // The next explicit refresh remains available if transient polling fails.
    } finally {
      schedulePoll();
    }
  }, POLL_DELAY_MS);
};

const selectBundle = async (bundleId) => {
  selectedBundleId.value = bundleId;
  actionError.value = '';
  beginCandidateSelection();
  try {
    selectedBundle.value = await apiAdapter.getReviewedReleaseBundle(bundleId);
    await loadSelectedJobs();
    schedulePoll();
  } catch (error) {
    actionError.value = friendlyError(error, '加载 Bundle 详情失败。');
  }
};

const loadBundles = async () => {
  if (!isDesktop.value) return;
  loading.value = true;
  listError.value = '';
  try {
    const data = await apiAdapter.listReviewedReleaseBundles();
    bundles.value = data?.items || [];
    const nextId = selectedBundleId.value || bundles.value[0]?.bundle_id || '';
    if (nextId && bundles.value.some((bundle) => bundle.bundle_id === nextId)) await selectBundle(nextId);
    else {
      selectedBundleId.value = '';
      selectedBundle.value = null;
      jobsById.value = {};
      selectedCandidateDetail.value = null;
      selectedCandidateEligibility.value = null;
      publicationSelectionByCandidate.value = {};
    }
  } catch (error) {
    listError.value = friendlyError(error, '加载 Reviewed Release Bundle 失败，请重新加载。');
  } finally {
    loading.value = false;
  }
};

const openImportDialog = () => {
  importError.value = '';
  importDialogVisible.value = true;
};

const resetImport = () => {
  manifestText.value = '';
  selectedFilename.value = '';
  importError.value = '';
  importLoading.value = false;
  if (bundleFileInput.value) bundleFileInput.value.value = '';
};

const readBundleFile = async (event) => {
  const [file] = event.target.files || [];
  if (!file) return;
  try {
    manifestText.value = await file.text();
    selectedFilename.value = file.name;
    importError.value = '';
  } catch {
    importError.value = '无法读取所选 Bundle 文件。';
  }
};

const importBundle = async () => {
  importError.value = '';
  actionError.value = '';
  actionMessage.value = '';
  let manifest;
  try {
    manifest = JSON.parse(manifestText.value);
  } catch {
    importError.value = 'Bundle manifest 不是有效 JSON。';
    return;
  }

  importLoading.value = true;
  try {
    const imported = await apiAdapter.importReviewedReleaseBundle(manifest);
    importDialogVisible.value = false;
    actionMessage.value = `已导入 Bundle ${imported.bundle_id}。`;
    await loadBundles();
    await selectBundle(imported.bundle_id);
  } catch (error) {
    importError.value = friendlyError(error, '导入 Bundle 失败。');
  } finally {
    importLoading.value = false;
  }
};

const dispatchJob = async (jobId) => {
  actionMessage.value = '';
  actionError.value = '';
  actionLoading.value[jobId] = true;
  try {
    const job = await apiAdapter.dispatchReviewedBundleJob(jobId);
    if (applyActionResult(job, `Candidate Build ${jobId} 已进入队列。`, `Candidate Build ${jobId} 未能进入队列。`)) {
      schedulePoll();
    }
  } catch (error) {
    actionError.value = friendlyError(error, `开始 Candidate Build ${jobId} 失败。`);
  } finally {
    delete actionLoading.value[jobId];
  }
};

const retryJob = async (jobId) => {
  actionMessage.value = '';
  actionError.value = '';
  actionLoading.value[jobId] = true;
  try {
    const job = await apiAdapter.retryReviewedBundleJob(jobId);
    if (applyActionResult(job, `Candidate Build ${jobId} 已进入重试队列。`, `Candidate Build ${jobId} 未能进入重试队列。`)) {
      schedulePoll();
    }
  } catch (error) {
    actionError.value = friendlyError(error, `重试 Candidate Build ${jobId} 失败。`);
  } finally {
    delete actionLoading.value[jobId];
  }
};

const cancelJob = async (jobId) => {
  actionMessage.value = '';
  actionError.value = '';
  actionLoading.value[jobId] = true;
  try {
    const job = await apiAdapter.cancelReviewedBundleJob(jobId);
    if (applyActionResult(job, `Candidate Build ${jobId} 的取消结果已由服务端确认。`, `Candidate Build ${jobId} 的取消未获服务端确认。`)) {
      schedulePoll();
    }
  } catch (error) {
    actionError.value = friendlyError(error, `取消 Candidate Build ${jobId} 失败。`);
  } finally {
    delete actionLoading.value[jobId];
  }
};

const viewCandidate = async (candidateId) => {
  actionMessage.value = '';
  actionError.value = '';
  const selectionEpoch = beginCandidateSelection();
  candidateLoading.value[candidateId] = true;
  try {
    await loadCandidateSelection(candidateId, selectionEpoch);
  } catch (error) {
    if (selectionEpoch === candidateSelectionEpoch) {
      actionError.value = friendlyError(error, `加载 Candidate ${candidateId} 失败。`);
    }
  } finally {
    delete candidateLoading.value[candidateId];
  }
};

const recordCandidateInspection = async (candidateId) => {
  actionMessage.value = '';
  actionError.value = '';
  const selectionEpoch = beginCandidateSelection();
  candidateLoading.value[candidateId] = true;
  try {
    await apiAdapter.inspectReviewedCandidate(candidateId);
    if (await loadCandidateSelection(candidateId, selectionEpoch)) {
      actionMessage.value = `Candidate ${candidateId} 的 inspection 已记录。`;
    }
  } catch (error) {
    if (selectionEpoch === candidateSelectionEpoch) {
      actionError.value = friendlyError(error, `记录 Candidate ${candidateId} 的 inspection 失败。`);
    }
  } finally {
    delete candidateLoading.value[candidateId];
  }
};

const acceptCandidate = async (candidateId) => {
  actionMessage.value = '';
  actionError.value = '';
  const selectionEpoch = beginCandidateSelection();
  candidateLoading.value[candidateId] = true;
  try {
    await apiAdapter.acceptReviewedCandidate(candidateId);
    if (await loadCandidateSelection(candidateId, selectionEpoch)) {
      actionMessage.value = `Candidate ${candidateId} 的 Supported 与 Boundary 验收已记录。`;
    }
  } catch (error) {
    if (selectionEpoch === candidateSelectionEpoch) {
      actionError.value = friendlyError(error, `执行 Candidate ${candidateId} 验收失败。`);
    }
  } finally {
    delete candidateLoading.value[candidateId];
  }
};

const isPublicationSelected = (candidateId) => Boolean(publicationSelectionByCandidate.value[candidateId]);

const togglePublicationSelection = (candidateId) => {
  if (isPublicationSelected(candidateId)) {
    const { [candidateId]: _removed, ...remaining } = publicationSelectionByCandidate.value;
    publicationSelectionByCandidate.value = remaining;
    return;
  }
  const eligibility = selectedCandidateEligibility.value;
  if (!eligibility?.eligible || eligibility.candidate_id !== candidateId) return;
  publicationSelectionByCandidate.value = {
    ...publicationSelectionByCandidate.value,
    [candidateId]: {
      candidate_id: candidateId,
      effect: eligibility.effect,
      current_published_knowledge_version: eligibility.current_published_knowledge_version?.identity || null,
      inspection_record_identity: eligibility.inspection_record_identity,
      acceptance_record_identity: eligibility.acceptance_record_identity
    }
  };
};

const openPublicationConfirmation = () => {
  publicationResults.value = null;
  publicationConfirmationAcknowledged.value = false;
  publicationConfirmationVisible.value = true;
};

const publicationConfirmationId = () => {
  if (typeof globalThis.crypto?.randomUUID === 'function') {
    return `candidate-publication-${globalThis.crypto.randomUUID()}`;
  }
  return `candidate-publication-${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

const shouldReplacePublicationConfirmation = (error) =>
  error?.status >= 400 &&
  error?.status < 500 &&
  !['PUBLICATION_CONFIRMATION_IN_PROGRESS', 'PUBLICATION_CONFIRMATION_LEASE_LOST'].includes(error?.code);

const confirmPublication = async () => {
  actionMessage.value = '';
  actionError.value = '';
  publicationLoading.value = true;
  try {
    if (!activePublicationConfirmationId.value) {
      activePublicationConfirmationId.value = publicationConfirmationId();
      activePublicationConfirmationSelections.value = publicationSelections.value.map((item) => ({ ...item }));
    }
    const selectedItems = activePublicationConfirmationSelections.value || [];
    const result = await apiAdapter.publishReviewedCandidateBatch({
      confirmation_id: activePublicationConfirmationId.value,
      selected_items: selectedItems
    });
    publicationResults.value = result;
    const retryableIds = new Set([
      ...result.failed.map((item) => item.candidate_id),
      ...result.skipped.map((item) => item.candidate_id)
    ]);
    publicationSelectionByCandidate.value = Object.fromEntries(
      selectedItems
        .filter((item) => retryableIds.has(item.candidate_id))
        .map((item) => [item.candidate_id, item])
    );
    activePublicationConfirmationId.value = null;
    activePublicationConfirmationSelections.value = null;
    if (selectedCandidateDetail.value) await viewCandidate(selectedCandidateDetail.value.candidate.candidate_id);
    try {
      await loadSelectedJobs();
    } catch (error) {
      actionError.value = friendlyError(error, '发布结果已保存，任务状态刷新失败。');
    }
    const batchState = result.batch_complete ? '批次发布完成' : '批次未全部发布';
    actionMessage.value = `${batchState}：${result.published.length} 项已发布，${result.failed.length} 项失败，${result.skipped.length} 项跳过。`;
  } catch (error) {
    if (shouldReplacePublicationConfirmation(error)) {
      activePublicationConfirmationId.value = null;
      activePublicationConfirmationSelections.value = null;
    }
    actionError.value = friendlyError(error, '批次发布失败。');
  } finally {
    publicationLoading.value = false;
  }
};

const updateViewportScope = () => {
  const wasDesktop = isDesktop.value;
  isDesktop.value = window.innerWidth >= 768;
  if (!isDesktop.value && pollTimer) clearTimeout(pollTimer);
  if (!wasDesktop && isDesktop.value) loadBundles();
};

onMounted(() => {
  updateViewportScope();
  window.addEventListener('resize', updateViewportScope);
  if (isDesktop.value) loadBundles();
});

onBeforeUnmount(() => {
  isActive.value = false;
  if (pollTimer) clearTimeout(pollTimer);
  window.removeEventListener('resize', updateViewportScope);
});
</script>

<style scoped>
.reviewed-bundles { max-width: 1440px; margin: 0 auto; }
.reviewed-bundles__header, .reviewed-bundles__section-header { display: flex; align-items: flex-start; justify-content: space-between; gap: var(--space-5); }
.reviewed-bundles__header { padding-bottom: var(--space-5); border-bottom: 1px solid var(--color-rule); }
.reviewed-bundles__eyebrow, .reviewed-bundles__detail-eyebrow { margin: 0 0 var(--space-2); color: var(--color-moss); font-size: 12px; font-weight: 600; }
.reviewed-bundles h1, .reviewed-bundles h2 { margin: 0; font-family: var(--font-display); font-weight: 600; }
.reviewed-bundles h1 { font-size: 26px; line-height: 1.3; }
.reviewed-bundles h2 { font-size: 18px; line-height: 1.4; }
.reviewed-bundles__description { max-width: 640px; margin: var(--space-2) 0 0; color: var(--color-ink-soft); font-size: 14px; line-height: 1.7; }
.reviewed-bundles__header-actions, .reviewed-bundles__import-tools, .reviewed-bundles__actions { display: flex; align-items: center; gap: var(--space-2); }
.reviewed-bundles__header-actions button, .reviewed-bundles__import-tools button, .reviewed-bundles__dialog-button, .reviewed-bundles__error button, .reviewed-bundles__actions button { display: inline-flex; min-height: 32px; align-items: center; justify-content: center; gap: 6px; border: 1px solid var(--color-rule); border-radius: var(--radius-control); background: var(--color-paper-raised); color: var(--color-ink); font: inherit; font-size: 13px; cursor: pointer; }
.reviewed-bundles__header-actions button, .reviewed-bundles__import-tools button, .reviewed-bundles__dialog-button { padding: 0 var(--space-3); }
.reviewed-bundles__header-actions button:disabled, .reviewed-bundles__import-tools button:disabled, .reviewed-bundles__dialog-button:disabled, .reviewed-bundles__actions button:disabled { cursor: wait; opacity: 0.65; }
.reviewed-bundles__header-actions button:not(:disabled):hover, .reviewed-bundles__import-tools button:not(:disabled):hover, .reviewed-bundles__dialog-button:not(:disabled):hover, .reviewed-bundles__actions button:not(:disabled):hover { border-color: var(--color-copper); color: var(--color-copper-strong); }
.reviewed-bundles__refresh-icon--spinning { animation: reviewed-bundles-spin 0.9s linear infinite; }
.reviewed-bundles__desktop-notice, .reviewed-bundles__error, .reviewed-bundles__success { display: flex; align-items: center; gap: var(--space-3); margin: var(--space-5) 0 0; padding: var(--space-3) var(--space-4); border-left: 3px solid var(--color-warning); background: var(--color-warning-soft); color: var(--color-warning); font-size: 13px; line-height: 1.5; }
.reviewed-bundles__desktop-notice p, .reviewed-bundles__error, .reviewed-bundles__success { margin: 0; }
.reviewed-bundles__error { border-left-color: var(--color-danger); background: var(--color-danger-soft); color: var(--color-danger); }
.reviewed-bundles__success { border-left-color: var(--color-moss); background: var(--color-moss-soft); color: var(--color-moss); }
.reviewed-bundles__error button { min-height: 28px; margin-left: auto; padding: 0 var(--space-2); border-color: currentColor; background: transparent; color: inherit; }
.reviewed-bundles__workspace { display: grid; grid-template-columns: minmax(310px, 0.8fr) minmax(0, 1.7fr); min-height: 540px; margin-top: var(--space-5); border-top: 1px solid var(--color-rule); border-bottom: 1px solid var(--color-rule); }
.reviewed-bundles__inventory { border-right: 1px solid var(--color-rule); background: var(--color-paper-muted); }
.reviewed-bundles__inventory, .reviewed-bundles__detail { min-width: 0; padding: var(--space-4); }
.reviewed-bundles__section-header { min-height: 42px; margin-bottom: var(--space-3); }
.reviewed-bundles__section-header > p { margin: 4px 0 0; color: var(--color-ink-soft); font-size: 12px; }
.reviewed-bundles__table-wrap, .reviewed-bundles__item-table-wrap { overflow: auto; border-top: 1px solid var(--color-rule); border-bottom: 1px solid var(--color-rule); background: var(--color-paper-raised); }
.reviewed-bundles table { width: 100%; border-collapse: collapse; table-layout: fixed; }
.reviewed-bundles__inventory table { min-width: 610px; }
.reviewed-bundles__item-table-wrap table { min-width: 1180px; }
.reviewed-bundles th, .reviewed-bundles td { padding: 11px 10px; border-bottom: 1px solid var(--color-rule); color: var(--color-ink); font-size: 12px; line-height: 1.5; text-align: left; vertical-align: top; }
.reviewed-bundles th { position: sticky; top: 0; z-index: 1; background: var(--color-paper-muted); color: var(--color-ink-soft); font-size: 11px; font-weight: 600; }
.reviewed-bundles tbody tr:last-child td { border-bottom: 0; }
.reviewed-bundles__row--selected td { background: var(--color-moss-soft); }
.reviewed-bundles__bundle-select { width: 100%; padding: 0; border: 0; background: transparent; color: var(--color-ink); font: inherit; font-family: var(--font-mono); font-size: 12px; text-align: left; cursor: pointer; overflow-wrap: anywhere; }
.reviewed-bundles__bundle-select:hover { color: var(--color-copper-strong); }
.reviewed-bundles__identifier { margin: 0; overflow-wrap: anywhere; font-family: var(--font-mono); font-size: 11px; }
.reviewed-bundles__timestamp, .reviewed-bundles__job-time { color: var(--color-ink-soft); font-size: 11px; }
.reviewed-bundles__entry, .reviewed-bundles__job-line, .reviewed-bundles__failure { margin: var(--space-1) 0 0; overflow-wrap: anywhere; }
.reviewed-bundles__entry, .reviewed-bundles__job-line { color: var(--color-ink-soft); font-size: 11px; }
.reviewed-bundles__failure { color: var(--color-danger); font-size: 11px; }
.reviewed-bundles__job-time { margin: var(--space-1) 0 0; }
.reviewed-bundles__status { display: inline-flex; min-width: 72px; justify-content: center; padding: 2px 5px; border: 1px solid currentColor; border-radius: 3px; font-size: 11px; white-space: nowrap; }
.reviewed-bundles__status--success { color: var(--color-moss); }
.reviewed-bundles__status--danger { color: var(--color-danger); }
.reviewed-bundles__status--neutral { color: var(--color-ink-soft); }
.reviewed-bundles__facts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0; margin: 0 0 var(--space-5); border-top: 1px solid var(--color-rule); border-bottom: 1px solid var(--color-rule); }
.reviewed-bundles__facts div { min-width: 0; padding: var(--space-3) var(--space-2); border-bottom: 1px solid var(--color-rule); }
.reviewed-bundles__facts div:nth-last-child(-n + 2) { border-bottom: 0; }
.reviewed-bundles__facts dt { color: var(--color-ink-soft); font-size: 11px; }
.reviewed-bundles__facts dd { margin: var(--space-1) 0 0; color: var(--color-ink); font-size: 12px; overflow-wrap: anywhere; }
.reviewed-bundles__next-action { color: var(--color-ink-soft); overflow-wrap: anywhere; }
.reviewed-bundles__actions { justify-content: flex-end; }
.reviewed-bundles__actions button { width: 30px; min-width: 30px; padding: 0; }
.reviewed-bundles__actions .reviewed-bundles__cancel { border-color: var(--color-danger); color: var(--color-danger); }
.reviewed-bundles__candidate-panel { margin-top: var(--space-5); padding-top: var(--space-4); border-top: 1px solid var(--color-rule); }
.reviewed-bundles__candidate-panel .reviewed-bundles__section-header > div:first-child { min-width: 0; overflow-wrap: anywhere; }
.reviewed-bundles__candidate-actions { flex-shrink: 0; }
.reviewed-bundles__record-detail { min-width: 0; margin: var(--space-3) 0; font-size: 12px; }
.reviewed-bundles__record-detail summary { cursor: pointer; color: var(--color-ink-soft); }
.reviewed-bundles__record-detail pre, .reviewed-bundles__replacement-content pre { max-height: 300px; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; font-family: var(--font-mono); font-size: 11px; line-height: 1.6; }
.reviewed-bundles__acceptance { padding: var(--space-4) 0; border-top: 1px solid var(--color-rule); }
.reviewed-bundles__replacement-content { grid-column: 1 / -1; min-width: 0; font-size: 12px; }
.reviewed-bundles__confirmation-bindings { grid-column: 1 / -1; min-width: 0; margin: 0; }
.reviewed-bundles__confirmation-bindings dd { margin: 4px 0 var(--space-2); overflow-wrap: anywhere; font-family: var(--font-mono); font-size: 11px; }
.reviewed-bundles__candidate-actions { display: flex; align-items: center; gap: var(--space-2); }
.reviewed-bundles__candidate-actions button { display: inline-flex; width: 30px; min-width: 30px; min-height: 30px; align-items: center; justify-content: center; padding: 0; border: 1px solid var(--color-rule); border-radius: var(--radius-control); background: var(--color-paper-raised); color: var(--color-ink); cursor: pointer; }
.reviewed-bundles__candidate-actions button:disabled { cursor: wait; opacity: 0.65; }
.reviewed-bundles__candidate-actions button:not(:disabled):hover { border-color: var(--color-copper); color: var(--color-copper-strong); }
.reviewed-bundles__candidate-facts { margin-bottom: var(--space-4); }
.reviewed-bundles__candidate-configuration { grid-column: 1 / -1; }
.reviewed-bundles__candidate-configuration pre { max-height: 180px; margin: var(--space-2) 0 0; overflow: auto; padding: var(--space-2); border-top: 1px solid var(--color-rule); background: var(--color-paper-muted); color: var(--color-ink); font-family: var(--font-mono); font-size: 10px; line-height: 1.55; white-space: pre-wrap; overflow-wrap: anywhere; }
.reviewed-bundles__replacement { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: var(--space-4); align-items: start; padding: var(--space-3) 0; border-top: 1px solid var(--color-rule); border-bottom: 1px solid var(--color-rule); }
.reviewed-bundles__replacement-title { margin: 0; color: var(--color-ink); font-size: 13px; font-weight: 600; }
.reviewed-bundles__replacement .reviewed-bundles__identifier { margin-top: var(--space-1); }
.reviewed-bundles__diff { display: grid; grid-template-columns: repeat(3, minmax(52px, 1fr)); margin: 0; border-left: 1px solid var(--color-rule); }
.reviewed-bundles__diff div { padding: 0 var(--space-3); text-align: center; }
.reviewed-bundles__diff dt { color: var(--color-ink-soft); font-size: 11px; white-space: nowrap; }
.reviewed-bundles__diff dd { margin: var(--space-1) 0 0; color: var(--color-ink); font-size: 16px; font-weight: 600; }
.reviewed-bundles__candidate-reasons { margin-top: var(--space-3); padding: var(--space-2) var(--space-3); border-left: 3px solid var(--color-warning); background: var(--color-warning-soft); color: var(--color-warning); font-size: 12px; line-height: 1.5; overflow-wrap: anywhere; }
.reviewed-bundles__publication-choice { display: inline-flex; align-items: center; gap: var(--space-2); margin-top: var(--space-3); color: var(--color-ink); font-size: 13px; cursor: pointer; }
.reviewed-bundles__publication-choice input { width: 16px; height: 16px; accent-color: var(--color-moss); }
.reviewed-bundles__candidate-chunks { margin-top: var(--space-4); }
.reviewed-bundles__candidate-chunk { margin-top: var(--space-2); border-top: 1px solid var(--color-rule); border-bottom: 1px solid var(--color-rule); background: var(--color-paper-muted); }
.reviewed-bundles__candidate-chunk header { display: flex; align-items: center; justify-content: space-between; gap: var(--space-3); padding: var(--space-2) var(--space-3); border-bottom: 1px solid var(--color-rule); color: var(--color-ink-soft); font-size: 11px; }
.reviewed-bundles__candidate-chunk code { overflow-wrap: anywhere; font-family: var(--font-mono); font-size: 10px; text-align: right; }
.reviewed-bundles__candidate-chunk pre { max-height: 240px; margin: 0; overflow: auto; padding: var(--space-3); color: var(--color-ink); font-family: var(--font-mono); font-size: 11px; line-height: 1.6; white-space: pre-wrap; overflow-wrap: anywhere; }
.reviewed-bundles__chunk-facts { display: grid; grid-template-columns: minmax(120px, 0.35fr) minmax(0, 1fr); gap: 0; margin: 0; border-bottom: 1px solid var(--color-rule); }
.reviewed-bundles__chunk-facts div { min-width: 0; padding: var(--space-2) var(--space-3); }
.reviewed-bundles__chunk-facts dt { color: var(--color-ink-soft); font-size: 10px; }
.reviewed-bundles__chunk-facts dd { margin: 2px 0 0; overflow-wrap: anywhere; color: var(--color-ink); font-family: var(--font-mono); font-size: 10px; }
.reviewed-bundles__candidate-chunk .reviewed-bundles__chunk-metadata { max-height: 220px; border-top: 1px solid var(--color-rule); background: var(--color-paper-raised); color: var(--color-ink-soft); }
.reviewed-bundles__publication-confirmation { display: grid; gap: var(--space-4); }
.reviewed-bundles__publication-confirmation ul { display: grid; gap: var(--space-2); max-height: 320px; margin: 0; padding: 0; overflow: auto; list-style: none; }
.reviewed-bundles__publication-confirmation li { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: var(--space-2) var(--space-3); align-items: center; padding: var(--space-3); border-top: 1px solid var(--color-rule); border-bottom: 1px solid var(--color-rule); background: var(--color-paper-muted); color: var(--color-ink-soft); font-size: 12px; }
.reviewed-bundles__publication-confirmation li code { min-width: 0; overflow-wrap: anywhere; font-family: var(--font-mono); font-size: 11px; }
.reviewed-bundles__publication-confirmation li code:last-child { grid-column: 1 / -1; }
.reviewed-bundles__publication-outcome { margin: 0; padding: var(--space-3); border-left: 3px solid var(--color-moss); background: var(--color-moss-soft); color: var(--color-moss); font-size: 13px; }
.reviewed-bundles__partial { border-left-color: var(--color-warning); background: var(--color-warning-soft); color: var(--color-warning); }
.reviewed-bundles__detail-empty, .reviewed-bundles__state { color: var(--color-ink-soft); font-size: 13px; text-align: center; }
.reviewed-bundles__detail-empty { margin: 150px 0; }
.reviewed-bundles__state { height: 180px; }
.reviewed-bundles__import-form { display: grid; gap: var(--space-4); }
.reviewed-bundles__selected-file { color: var(--color-ink-soft); font-family: var(--font-mono); font-size: 12px; overflow-wrap: anywhere; }
.reviewed-bundles__import-form label { display: grid; gap: var(--space-2); color: var(--color-ink); font-size: 13px; font-weight: 600; }
.reviewed-bundles__import-form textarea { width: 100%; min-height: 300px; resize: vertical; padding: var(--space-3); border: 1px solid var(--color-rule); border-radius: 3px; background: var(--color-paper-muted); color: var(--color-ink); font-family: var(--font-mono); font-size: 12px; line-height: 1.55; }
.reviewed-bundles__import-form textarea:focus-visible, .reviewed-bundles__bundle-select:focus-visible, .reviewed-bundles button:focus-visible { outline: 2px solid var(--color-focus); outline-offset: 2px; }
.reviewed-bundles__dialog-button--primary { border-color: var(--color-moss); background: var(--color-moss); color: var(--color-paper-raised); }
.reviewed-bundles__dialog-button--primary:not(:disabled):hover { border-color: var(--color-moss); background: var(--color-moss); color: var(--color-paper-raised); opacity: 0.88; }
@keyframes reviewed-bundles-spin { to { transform: rotate(360deg); } }
@media (max-width: 1180px) {
  .reviewed-bundles__workspace { grid-template-columns: minmax(290px, 0.72fr) minmax(0, 1.45fr); }
  .reviewed-bundles__inventory, .reviewed-bundles__detail { padding: var(--space-3); }
  .reviewed-bundles__replacement { grid-template-columns: 1fr; }
  .reviewed-bundles__diff { border-top: 1px solid var(--color-rule); border-left: 0; padding-top: var(--space-3); }
}
</style>
