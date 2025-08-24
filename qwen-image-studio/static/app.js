let ws;
let currentMode = 'generate';
let isSubmitting = false;
const jobs = new Map();
const actionLocks = new Set();
const expandedPrompts = new Set();
let jobSearchQuery = '';
const JOBS_BATCH = 20;
let jobsVisible = JOBS_BATCH;
let _observerInit = false;

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

async function copyText(t) {
    try {
        if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(t);
            return true;
        }
    } catch (_) { }
    const ta = document.createElement('textarea');
    ta.value = t; ta.setAttribute('readonly', '');
    ta.style.position = 'fixed'; ta.style.top = '-1000px';
    document.body.appendChild(ta); ta.select();
    let ok = false; try { ok = document.execCommand('copy'); } catch (_) { }
    document.body.removeChild(ta); return !!ok;
}

function togglePrompt(jobId, el) {
    if (expandedPrompts.has(jobId)) expandedPrompts.delete(jobId);
    else expandedPrompts.add(jobId);
    el.classList.toggle('expanded');
}

function outputThumb(absPath) {
    const url = `/api/file?path=${encodeURIComponent(absPath)}`;
    const name = absPath.split('/').pop();
    return `<figure style="margin:.5rem 0;">
    <img src="${url}" alt="${escapeHTML(name)}" style="max-width:100%;border-radius:8px"/>
    <figcaption class="muted" style="font-size:.8rem">${escapeHTML(name)}</figcaption>
  </figure>`;
}

function cancelAllJobs() {
    const actives = Array.from(jobs.values()).filter(j => j.status === 'queued' || j.status === 'processing');
    if (actives.length === 0) return;
    if (!confirm(`Cancel ${actives.length} active job(s)?`)) return;

    const btn = $('#cancelAllBtn'); if (btn) btn.disabled = true;

    // locally flip to "cancelling" and reuse existing single-cancel path
    for (const j of actives) {
        j.status = 'cancelling';
        j.stage = 'cancelling';
        jobs.set(j.id, j);
        if (ws?.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'cancel_job', job_id: j.id }));
        }
    }
    updateUI();

    // re-enable button shortly (WS updates will finalize states)
    setTimeout(() => { if (btn) btn.disabled = false; }, 1500);
}

function shouldShowError(job) {
    // Only show error if job has permanently failed (no more retries)
    // If status is 'failed', it means all retries are exhausted
    return job.error && job.status === 'failed';
}

function getThumbnailClass(job) {
    if (job.status === 'completed' && job.outputs?.length) return 'thumbnail-completed';
    if (job.status === 'failed') return 'thumbnail-failed';
    if (job.status === 'processing') return 'thumbnail-generating';
    return 'thumbnail-pending';
}

function getThumbnailClick(job) {
    if (job.status === 'completed' && job.outputs?.length) {
        const url = `/api/file?path=${encodeURIComponent(job.outputs[0])}`;
        return `openImage('${url}')`;
    }
    return '';
}

function getThumbnailContent(job) {
    if (job.status === 'completed' && job.outputs?.length) {
        const url = `/api/file?path=${encodeURIComponent(job.outputs[0])}`;
        return `<img src="${url}" alt="Generated image" />`;
    }
    if (job.status === 'failed') return '<span class="error-icon">✗</span>';
    if (job.status === 'processing') return '<div class="spinner"></div>';
    if (job.status === 'queued') return '<span class="queue-icon">⏳</span>';
    return '<span class="pending-icon">○</span>';
}

function openImage(url) {
    window.open(url, '_blank');
}

function saveJobsToStorage() { /* no-op: server is the source of truth */ }

async function loadJobsFromStorage() {
    try {
        const res = await fetch('/api/jobs');
        const data = await res.json();
        jobs.clear();
        for (const j of (data.jobs || [])) jobs.set(j.id, j);
        updateUI();
    } catch (e) {
        console.error('Failed to load jobs from server', e);
    }
}
async function deleteJob(jobId) {
    if (actionLocks.has(`delete:${jobId}`)) return;
    const j = jobs.get(jobId);
    if (!j) return;

    const running = (j.status === 'queued' || j.status === 'processing' || j.status === 'cancelling');
    const msg = running ? 'This job is running. Cancel and delete it?' : 'Delete this job?';
    if (!confirm(msg)) return;

    actionLocks.add(`delete:${jobId}`);
    updateUI();

    try {
        const res = await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        await loadJobsFromStorage(); // refresh from server
    } catch (e) {
        console.error('Delete failed', e);
        actionLocks.delete(`delete:${jobId}`);
        updateUI();
    }
}

function waitForCancellationThenDelete(jobId, tries = 120) {
    const t = setInterval(() => {
        const j = jobs.get(jobId);
        if (!j) { clearInterval(t); actionLocks.delete(`delete:${jobId}`); return; }
        if (j.status === 'cancelled') {
            clearInterval(t);
            jobs.delete(jobId);
            saveJobsToStorage();
            updateUI();
            actionLocks.delete(`delete:${jobId}`);
        } else if (--tries <= 0) {
            clearInterval(t);
            actionLocks.delete(`delete:${jobId}`);
        }
    }, 500);
}

function updateGallery() {
    const el = $('#gallery');
    if (!el) return;

    const cards = [];
    for (const j of jobs.values()) {
        if (j.status !== 'completed') continue;
        const outs = j.outputs || [];
        for (const absPath of outs) {
            cards.push(outputThumb(absPath));
        }
    }
    el.innerHTML = cards.length ? cards.join('') : '<p class="muted">No images yet.</p>';
}

function initWebSocket() {
    ws = new WebSocket(`ws://${window.location.host}/ws`);
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'init') {
            data.jobs.forEach(job => jobs.set(job.id, job));
            updateUI();
        } else if (data.type === 'job_update') {
            jobs.set(data.job.id, data.job);
            saveJobsToStorage();
            updateUI();
        }
        else if (data.type === 'gpu_stats') {
            updateGPUStats(data.stats);
        }
    };
    ws.onclose = () => setTimeout(initWebSocket, 3000);
}

function applyMode(mode) {
    currentMode = mode;
    $$('.tab-button').forEach(b => {
        const active = b.dataset.mode === mode;
        b.classList.toggle('active', active);
        b.setAttribute('aria-selected', active ? 'true' : 'false');
    });
    $('#editSection').classList.toggle('hidden', mode !== 'edit');
    $('#editSection').setAttribute('aria-hidden', mode === 'edit' ? 'false' : 'true');

    const resField = document.getElementById('resolution-field');
    if (resField) resField.style.display = (mode === 'generate') ? '' : 'none';

    const genLink = $('#generateModelLink');
    const editLink = $('#editModelLink');
    if (genLink) {
        genLink.classList.toggle('hidden', mode !== 'generate');
        genLink.setAttribute('aria-hidden', mode === 'generate' ? 'false' : 'true');
    }
    if (editLink) {
        editLink.classList.toggle('hidden', mode !== 'edit');
        editLink.setAttribute('aria-hidden', mode === 'edit' ? 'false' : 'true');
    }

    const submitBtnIcon = $('#submitBtnIcon');
    const submitBtnText = $('#submitBtnText');
    if (mode === 'edit') {
        submitBtnIcon.textContent = '✏️';
        submitBtnText.textContent = 'Edit Image';
    } else {
        submitBtnIcon.textContent = '✨';
        submitBtnText.textContent = 'Generate Image';
    }

    localStorage.setItem('preferredMode', mode);
}

/* ---------- Time formatting ---------- */
function formatDuration(seconds) {
    seconds = Math.max(0, Math.floor(seconds));
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    if (h > 0) return `${h}h ${m}m ${s}s`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
}

function updateUI() { updateQueue(); updateSubmitButton(); updateGallery(); }

// OPTIMIZED: Only update jobs that have actually changed
// Add this to track what was last rendered
let lastRenderedJobs = new Map();
let lastJobsVisible = 0;
let lastSearchQuery = '';

function updateQueue() {
    const queueSection = $('#jobQueue');
    const jobList = $('#jobList');

    let activeJobs = Array.from(jobs.values())
        .filter(j => ['queued', 'processing', 'failed', 'completed', 'cancelling', 'cancelled'].includes(j.status))
        .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

    // Search filtering
    if (jobSearchQuery.trim()) {
        const q = jobSearchQuery.toLowerCase();
        activeJobs = activeJobs.filter(j => (j.params?.prompt || '').toLowerCase().includes(q));
    }

    const total = activeJobs.length;
    if (total === 0) {
        queueSection.classList.add('hidden');
        jobList.innerHTML = '';
        lastRenderedJobs.clear();
        return;
    }

    queueSection.classList.remove('hidden');
    jobsVisible = Math.min(jobsVisible, total);
    const visible = activeJobs.slice(0, jobsVisible);
    window.__filteredJobsCount = total;

    // OPTIMIZATION: Only rebuild if something significant changed
    const needsFullRebuild = (
        lastJobsVisible !== jobsVisible ||
        lastSearchQuery !== jobSearchQuery ||
        lastRenderedJobs.size !== visible.length
    );

    if (needsFullRebuild) {
        // Full rebuild needed
        renderAllJobs(visible);
        lastRenderedJobs.clear();
        visible.forEach(job => lastRenderedJobs.set(job.id, { ...job }));
        lastJobsVisible = jobsVisible;
        lastSearchQuery = jobSearchQuery;
        return;
    }

    // OPTIMIZED: Only update jobs that changed
    visible.forEach(job => {
        const lastJob = lastRenderedJobs.get(job.id);
        if (!lastJob || jobHasChanged(job, lastJob)) {
            updateSingleJob(job);
            lastRenderedJobs.set(job.id, { ...job });
        }
    });
}

function jobHasChanged(current, previous) {
    // Always update running jobs (for timer)
    if (current.status === 'processing' || current.status === 'queued') {
        return true;
    }
    
    // Check the fields that matter for display
    return (
        current.status !== previous.status ||
        current.stage !== previous.stage ||
        (current.stages && JSON.stringify(current.stages) !== JSON.stringify(previous.stages)) ||
        current.retry_count !== previous.retry_count ||
        current.error !== previous.error ||
        current.completed_at !== previous.completed_at
    );
}

function updateSingleJob(job) {
    const existingCard = document.querySelector(`[data-job-id="${job.id}"]`);
    if (!existingCard) {
        // Job doesn't exist, need full rebuild
        const visible = Array.from(jobs.values())
            .filter(j => ['queued', 'processing', 'failed', 'completed', 'cancelling', 'cancelled'].includes(j.status))
            .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
            .slice(0, jobsVisible);
        renderAllJobs(visible);
        return;
    }

    // Update specific parts of the job card without touching animations
    updateJobStatus(existingCard, job);
    updateJobStages(existingCard, job);
    updateJobThumbnail(existingCard, job);
    updateJobActions(existingCard, job);
    updateJobError(existingCard, job);
}

function updateJobStatus(cardEl, job) {
    const statusPill = cardEl.querySelector('.status-pill');
    const timeInfo = cardEl.querySelector('.job-info small.muted');

    if (statusPill) {
        statusPill.className = `status-pill ${job.status}`;
        statusPill.textContent = job.status === 'processing' ? 'Generating'
            : job.status === 'queued' ? 'Queued'
                : job.status === 'cancelling' ? 'Cancelling'
                    : job.status === 'cancelled' ? 'Cancelled'
                        : job.status === 'failed' ? 'Failed'
                            : job.status === 'completed' ? 'Completed'
                                : job.status || '';
    }

    if (timeInfo) {
        const isTerminal = ['completed', 'failed', 'cancelled'].includes(job.status);
        const startTs = job.started_at ? new Date(job.started_at).getTime() : null;
        const endTs = isTerminal
            ? new Date(job.completed_at || job.updated_at || job.created_at || Date.now()).getTime()
            : Date.now();
        const elapsed = startTs ? Math.max(0, Math.floor((endTs - startTs) / 1000)) : 0;
        const duration = (job.completed_at && job.started_at)
            ? Math.max(0, Math.floor((new Date(job.completed_at).getTime() - startTs) / 1000))
            : elapsed;

        timeInfo.textContent = job.status === 'completed' ? `Completed in ${formatDuration(duration)}`
            : job.status === 'failed' ? `Failed after ${formatDuration(elapsed)} • Retries ${job.retry_count}/${job.max_retries}`
                : job.status === 'cancelled' ? `Stopped after ${formatDuration(elapsed)}`
                    : `Elapsed: ${formatDuration(elapsed)} • Retries ${job.retry_count}/${job.max_retries}`;
    }
}

function updateJobStages(cardEl, job) {
    const stagesContainer = cardEl.querySelector('.job-stages');
    if (!stagesContainer) return;

    const order = ['model_loading', 'pipeline_loading', 'lora_loading', 'generation'];
    const stagesHTML = job.stages ? Object.entries(job.stages)
        .sort(([a], [b]) => order.indexOf(a) - order.indexOf(b))
        .map(([stage, data]) => `
            <div class="stage-row">
              <span class="stage-label">${formatStageName(stage)}</span>
              <div class="stage-progress">
                <progress value="${Math.round((data.progress || 0) * 100)}" max="100"></progress>
                <span class="stage-status">${data.status === 'completed' ? '✓'
                : data.status === 'active' ? `${Math.round((data.progress || 0) * 100)}%`
                    : ''
            }</span>
              </div>
            </div>
        `).join('') : '';

    stagesContainer.innerHTML = stagesHTML;
}

function updateJobThumbnail(cardEl, job) {
    const thumbnail = cardEl.querySelector('.job-thumbnail:not(.source)');
    if (!thumbnail) return;

    const newClass = getThumbnailClass(job);
    const newContent = getThumbnailContent(job);
    const newClick = getThumbnailClick(job);

    // CRITICAL: Only update if the class changed to avoid restarting animations
    if (thumbnail.className !== `job-thumbnail ${newClass}`) {
        thumbnail.className = `job-thumbnail ${newClass}`;
    }

    // Only update content if it changed
    if (thumbnail.innerHTML !== newContent) {
        thumbnail.innerHTML = newContent;
    }

    // Update click handler
    thumbnail.setAttribute('onclick', newClick);
}

function updateJobActions(cardEl, job) {
    const footer = cardEl.querySelector('.job-actions');
    if (!footer) return;

    const actionsHTML = (job.status === 'cancelling')
        ? `<button type="button" class="secondary" disabled>⏳ Cancelling…</button>`
        : (job.status === 'queued' || job.status === 'processing')
            ? `<button type="button" class="secondary small" onclick="cancelJob('${job.id}')">❌ Cancel</button>`
            : `<button type="button" class="secondary small" onclick="restartJob('${job.id}')">🔄 Restart</button>`;

    footer.innerHTML = actionsHTML +
        `<button type="button" class="secondary small" onclick="deleteJob('${job.id}')">🗑️ Delete</button>`;
}

function updateJobError(cardEl, job) {
    let errorEl = cardEl.querySelector('.job-error');

    if (shouldShowError(job)) {
        if (!errorEl) {
            errorEl = document.createElement('p');
            errorEl.className = 'job-error';
            cardEl.querySelector('.job-content').after(errorEl);
        }
        errorEl.textContent = job.error;
    } else if (errorEl) {
        errorEl.remove();
    }
}

function renderAllJobs(visible) {
    const jobList = $('#jobList');
    const order = ['model_loading', 'pipeline_loading', 'lora_loading', 'generation'];

    jobList.innerHTML = visible.map(job => {
        const isTerminal = ['completed', 'failed', 'cancelled'].includes(job.status);
        const startTs = job.started_at ? new Date(job.started_at).getTime() : null;
        const endTs = isTerminal
            ? new Date(job.completed_at || job.updated_at || job.created_at || Date.now()).getTime()
            : Date.now();
        const elapsed = startTs ? Math.max(0, Math.floor((endTs - startTs) / 1000)) : 0;
        const duration = (job.completed_at && job.started_at)
            ? Math.max(0, Math.floor((new Date(job.completed_at).getTime() - startTs) / 1000))
            : elapsed;

        const stagesHTML = job.stages ? Object.entries(job.stages)
            .sort(([a], [b]) => order.indexOf(a) - order.indexOf(b))
            .map(([stage, data]) => `
                <div class="stage-row">
                  <span class="stage-label">${formatStageName(stage)}</span>
                  <div class="stage-progress">
                    <progress value="${Math.round((data.progress || 0) * 100)}" max="100"></progress>
                    <span class="stage-status">${data.status === 'completed' ? '✓'
                    : data.status === 'active' ? `${Math.round((data.progress || 0) * 100)}%`
                        : ''
                }</span>
                  </div>
                </div>
            `).join('') : '';

        return `
            <article class="job-card" data-job-id="${job.id}">
                <div class="job-content">
                    <div class="job-info">
                            <header><strong>Task: ${job.type === 'generate' ? '✨' : '✏️'} ${job.type === 'generate' ? 'Generate' : 'Edit'}</strong></header>                        <div class="job-prompt ${expandedPrompts.has(job.id) ? 'expanded' : ''}"
                            onclick="togglePrompt('${job.id}', this)" title="Click to expand">
                            <span class="job-prompt-text">${escapeHTML(job.params?.prompt || '')}</span>
                            <button type="button" class="copyPromptBtn" data-id="${job.id}" title="Copy">📋</button>
                        </div>
                        ${(() => {
                const p = job.params || {};
                const tags = [];
                if (p.ultra_fast === true || p.ultra_fast === "true") tags.push("Ultra Fast (4 steps)");
                else if (p.fast === true || p.fast === "true") tags.push("Fast (8 steps)");
                else if (p.steps) tags.push(`${p.steps} steps`);
                if (p.seed) tags.push(`Seed ${p.seed}`);
                if (job.type === "generate" && p.size) tags.push(p.size);
                return tags.length ? `<small class="muted job-params">${tags.join(" • ")}</small>` : "";
            })()}
                        <span class="status-pill ${job.status}">${job.status === 'processing' ? 'Generating'
                : job.status === 'queued' ? 'Queued'
                    : job.status === 'cancelling' ? 'Cancelling'
                        : job.status === 'cancelled' ? 'Cancelled'
                            : job.status === 'failed' ? 'Failed'
                                : job.status === 'completed' ? 'Completed'
                                    : escapeHTML(job.status || '')
            }</span><br/>
                        <small class="muted">${job.status === 'completed' ? `Completed in ${formatDuration(duration)}`
                : job.status === 'failed' ? `Failed after ${formatDuration(elapsed)} • Retries ${job.retry_count}/${job.max_retries}`
                    : job.status === 'cancelled' ? `Stopped after ${formatDuration(elapsed)}`
                        : `Elapsed: ${formatDuration(elapsed)} • Retries ${job.retry_count}/${job.max_retries}`
            }</small>
                    </div>
                    <div class="job-stages">${stagesHTML}</div>
                    <div class="job-thumbs">
                        ${job.type === 'edit' && job.params?.image_path ? `
                            <div class="job-thumbnail source" onclick="openImage('/api/file?path=${encodeURIComponent(job.params.image_path)}')">
                                <img src="/api/file?path=${encodeURIComponent(job.params.image_path)}" alt="Source image"/>
                            </div>
                            <div class="arrow-down">↓</div>` : ''
            }
                        <div class="job-thumbnail ${getThumbnailClass(job)}" onclick="${getThumbnailClick(job)}">
                            ${getThumbnailContent(job)}
                        </div>
                    </div>
                </div>
                ${shouldShowError(job) ? `<p class="job-error">${escapeHTML(job.error)}</p>` : ''}
                <footer class="job-actions">
                    ${(job.status === 'cancelling')
                ? `<button type="button" class="secondary" disabled>⏳ Cancelling…</button>`
                : (job.status === 'queued' || job.status === 'processing')
                    ? `<button type="button" class="secondary" onclick="cancelJob('${job.id}')">⌫ Cancel</button>`
                    : `<button type="button" class="contrast" onclick="restartJob('${job.id}')">🔄 Restart</button>`
            }
                    <button type="button" class="secondary" onclick="deleteJob('${job.id}')">🗑️ Delete</button>
                </footer>
            </article>
        `;
    }).join('');

    // Re-attach copy button event listeners
    $$('.copyPromptBtn').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const id = e.currentTarget.dataset.id;
            const text = (jobs.get(id)?.params?.prompt) || '';
            if (!text) return;
            const ok = await copyText(text);
            if (ok) {
                const old = btn.textContent;
                btn.textContent = '✓';
                setTimeout(() => btn.textContent = old, 900);
            }
        });
    });
}

setInterval(() => { if (document.hasFocus()) updateQueue(); }, 1000);

function updateSubmitButton() {
    const btn = $('#submitBtn');
    const icon = $('#submitBtnIcon');
    const text = $('#submitBtnText');

    btn.disabled = isSubmitting;

    if (isSubmitting) {
        icon.textContent = '⏳';
        text.textContent = currentMode === 'generate' ? 'Generating...' : 'Editing...';
    } else {
        icon.textContent = currentMode === 'generate' ? '✨' : '✏️';
        text.textContent = currentMode === 'generate' ? 'Generate Image' : 'Edit Image';
    }
}

/* GPU widget (unchanged) */
function updateGPUStats(stats) {
    const gpuStats = $('#gpuStats');
    if (!stats || !stats.gpu_name) return;
    gpuStats.style.display = 'flex';

    const utilBar = $('#gpuUtilBar');
    const utilText = $('#gpuUtilText');
    utilBar.style.width = `${stats.gpu_utilization}%`;
    utilText.textContent = `${stats.gpu_utilization}%`;
    utilBar.className = `gpu-stat-fill ${stats.gpu_utilization > 80 ? 'high' : stats.gpu_utilization > 50 ? 'medium' : 'low'}`;

    const vramBar = $('#vramBar');
    const vramText = $('#vramText');
    vramBar.style.width = `${stats.vram_used_percent}%`;
    const vramGB = stats.vram_total >= 1024
        ? `${(stats.vram_used / 1024).toFixed(1)}/${(stats.vram_total / 1024).toFixed(1)}GB`
        : `${stats.vram_used}/${stats.vram_total}MB`;
    vramText.textContent = vramGB;
    vramBar.className = `gpu-stat-fill ${stats.vram_used_percent > 85 ? 'high' : stats.vram_used_percent > 70 ? 'medium' : 'low'}`;

    const gttBar = $('#gttBar');
    const gttText = $('#gttText');
    gttBar.style.width = `${stats.gtt_used_percent}%`;
    const gttGB = stats.gtt_total >= 1024
        ? `${(stats.gtt_used / 1024).toFixed(1)}/${(stats.gtt_total / 1024).toFixed(1)}GB`
        : `${stats.gtt_used}/${stats.gtt_total}MB`;
    gttText.textContent = gttGB;
    gttBar.className = `gpu-stat-fill ${stats.gtt_used_percent > 85 ? 'high' : stats.gtt_used_percent > 70 ? 'medium' : 'low'}`;

    const tempText = $('#tempText');
    tempText.textContent = `${stats.gpu_temperature}°C`;
    tempText.style.color = stats.gpu_temperature > 80 ? '#ff4d4d' : stats.gpu_temperature > 60 ? '#ffad33' : 'var(--pico-muted-color)';
}

function formatStageName(stage) {
    const names = { 'model_loading': 'Model', 'pipeline_loading': 'Pipeline', 'lora_loading': 'LoRA', 'generation': 'Generate' };
    return names[stage] || stage;
}

function cancelJob(jobId) {
    const j = jobs.get(jobId);
    if (!j) return;
    if (!(j.status === 'queued' || j.status === 'processing')) return;
    if (!confirm('Cancel this job?')) return;

    j.status = 'cancelling';
    j.stage = 'cancelling';
    jobs.set(jobId, j);
    saveJobsToStorage();
    updateUI();

    if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'cancel_job', job_id: jobId }));
    }
}

function restartJob(jobId) { if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'restart_job', job_id: jobId })); }

function setupImageUpload() {
    const uploadArea = $('#uploadArea');
    const imageInput = $('#imageInput');
    const preview = $('#uploadPreview');

    uploadArea.addEventListener('click', () => imageInput.click());
    imageInput.onchange = (e) => {
        const file = e.target.files?.[0]; if (!file) return;
        const reader = new FileReader();
        reader.onload = ev => { preview.src = ev.target.result; preview.classList.remove('hidden'); uploadArea.querySelector('p').textContent = file.name; };
        reader.readAsDataURL(file);
    };
    uploadArea.ondragover = e => { e.preventDefault(); uploadArea.classList.add('dragover'); };
    uploadArea.ondragleave = () => uploadArea.classList.remove('dragover');
    uploadArea.ondrop = e => {
        e.preventDefault(); uploadArea.classList.remove('dragover');
        const file = e.dataTransfer.files?.[0]; if (file) { imageInput.files = e.dataTransfer.files; imageInput.onchange({ target: { files: e.dataTransfer.files } }); }
    };
}

async function submitForm(e) {
    e.preventDefault();
    if (isSubmitting) return;            // guard against double clicks
    isSubmitting = true;
    updateSubmitButton();

    const fd = new FormData();

    const prompt = e.target.prompt.value;
    const steps = e.target.steps.value || "50";
    const seed = e.target.seed.value || "";
    const fast = e.target.fast.checked;
    const ultra_fast = e.target.ultra_fast.checked;
    const batman = e.target.batman.checked;
    const size = e.target.size.value || "16:9";

    fd.append('prompt', prompt);
    if (!(fast || ultra_fast)) {
        fd.append('steps', steps);
    }
    fd.append('fast', String(fast));
    fd.append('ultra_fast', String(ultra_fast));
    fd.append('batman', String(batman));
    if (currentMode === 'generate') fd.append('size', size);

    if (seed) fd.append('seed', seed);

    const endpoint = currentMode === 'generate' ? '/api/generate' : '/api/edit';
    if (currentMode === 'edit') {
        const imageFile = $('#imageInput').files?.[0];
        if (!imageFile) {
            alert('Please upload an image for editing');
            isSubmitting = false;
            updateSubmitButton();
            return;
        }
        fd.append('image', imageFile);
    }

    try {
        const res = await fetch(endpoint, { method: 'POST', body: fd });
        if (!res.ok) throw new Error('Submission failed');
        await res.json();
        // Do NOT reset form — keep prompt, checkboxes, etc.
        // Only clear upload preview if in generate mode
        if (currentMode === 'generate') {
            $('#uploadPreview').classList.add('hidden');
        }
    } catch (err) {
        alert('Error: ' + err.message);
    } finally {
        isSubmitting = false;
        updateSubmitButton();
    }
}

function toggleTheme() {
    const html = document.documentElement;
    const current = html.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    $('#themeToggle').textContent = next === 'dark' ? '🌙' : '☀️';
}

function loadSettings() {
    const theme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', theme);
    $('#themeToggle').textContent = theme === 'dark' ? '🌙' : '☀️';
    const mode = localStorage.getItem('preferredMode') || 'generate';
    applyMode(mode);
}

function escapeHTML(s) {
    return String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;');
}

document.addEventListener('DOMContentLoaded', () => {
    loadSettings();
    loadJobsFromStorage();
    setupImageUpload();
    initWebSocket();

    // wire tabs
    $$('.tab-button').forEach(b => b.addEventListener('click', () => applyMode(b.dataset.mode)));

    $('#cancelAllBtn')?.addEventListener('click', cancelAllJobs);

    // mutually exclusive fast / ultra
    $$('#imageForm input[name="fast"], #imageForm input[name="ultra_fast"]').forEach(cb => {
        cb.addEventListener('change', (e) => {
            if (e.target.checked) {
                $$('#imageForm input[name="fast"], #imageForm input[name="ultra_fast"]').forEach(other => { if (other !== e.target) other.checked = false; });
            }
        });
    });

    const stepsInput = $('#imageForm input[name="steps"]');
    function syncStepsDisabled() {
        const fast = $('#imageForm input[name="fast"]').checked;
        const ultra = $('#imageForm input[name="ultra_fast"]').checked;
        stepsInput.disabled = (fast || ultra);
        stepsInput.title = (fast || ultra) ? 'Ignored in Fast/Ultra mode' : '';
    }
    $$('#imageForm input[name="fast"], #imageForm input[name="ultra_fast"]').forEach(cb => {
        cb.addEventListener('change', syncStepsDisabled);
    });
    syncStepsDisabled();

    $('#themeToggle').addEventListener('click', toggleTheme);

    $('#imageForm').addEventListener('submit', submitForm);

    const searchEl = $('#jobSearch');
    if (searchEl) {
        searchEl.addEventListener('input', (e) => {
            jobSearchQuery = e.target.value || '';
            jobsVisible = JOBS_BATCH;
            updateQueue();
        });
    }

    const sentinel = $('#jobInfiniteSentinel');
    if (sentinel && !_observerInit) {
        const io = new IntersectionObserver((entries) => {
            if (entries.some(e => e.isIntersecting)) {
                const total = window.__filteredJobsCount ?? 0;
                if (jobsVisible < total) {
                    jobsVisible += JOBS_BATCH;
                    updateQueue();
                }
            }
        }, { root: null, rootMargin: '200px', threshold: 0 });
        io.observe(sentinel);
        _observerInit = true;
    }
});
