const API_URL = '/trainweb/api'; // Proxied via Nginx to backend tunnel

// ==================== TAB NAVIGATION ====================
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        // Update buttons
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Update panels
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        document.getElementById(btn.dataset.tab).classList.add('active');

        // Redraw chart if metrics tab is selected
        if (btn.dataset.tab === 'metrics-tab') {
            setTimeout(() => {
                if (window.lastMetrics) drawChart(window.lastMetrics);
            }, 100);
        }
    });
});

// ==================== DATA FETCHING ====================
async function fetchData() {
    try {
        const [metricsRes, infoRes, logsRes] = await Promise.all([
            fetch(`${API_URL}/metrics`),
            fetch(`${API_URL}/info`),
            fetch(`${API_URL}/logs`)
        ]);

        const metricsData = await metricsRes.json();
        const infoData = await infoRes.json();
        const logsData = await logsRes.json();

        updateDashboard(metricsData.metrics, infoData.config, logsData.logs);
        document.getElementById('connection-status').innerText = 'ONLINE';
        document.getElementById('connection-status').style.color = '#00ff00';
    } catch (error) {
        console.error('Error fetching data:', error);
        document.getElementById('connection-status').innerText = 'ERROR - OFFLINE';
        document.getElementById('connection-status').style.color = 'red';
    }
}

function updateDashboard(metrics, config, logs) {
    // 1. Update Stats
    if (metrics && metrics.length > 0) {
        window.lastMetrics = metrics; // Store for redrawing
        const last = metrics[metrics.length - 1];
        document.getElementById('val-loss').innerText = last.loss.toFixed(4);
        document.getElementById('val-acc').innerText = (last.accuracy * 100).toFixed(1) + '%';
        document.getElementById('val-epoch').innerText = last.epoch;

        drawChart(metrics);
    }

    // 2. Update Config
    if (config) {
        const configHtml = Object.entries(config)
            .map(([k, v]) => `<div><span style="opacity:0.7">${k}:</span> ${v}</div>`)
            .join('');
        document.getElementById('config-display').innerHTML = configHtml;
    }

    // 3. Update Logs
    if (logs) {
        const logText = logs.join('');
        const logsDiv = document.getElementById('logs-display');
        logsDiv.innerText = logText;
        logsDiv.scrollTop = logsDiv.scrollHeight;
    }
}

// ==================== CHART DRAWING ====================
let chartDrawTimeout = null;
function drawChart(metrics) {
    // Debounce rapid redraws
    if (chartDrawTimeout) clearTimeout(chartDrawTimeout);
    chartDrawTimeout = setTimeout(() => _drawChartImmediate(metrics), 50);
}

function _drawChartImmediate(metrics) {
    const canvas = document.getElementById('metricsCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const parent = canvas.parentElement;

    // Use device pixel ratio for sharp rendering
    const dpr = window.devicePixelRatio || 1;
    const rect = parent.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    // Clear canvas with dark background
    ctx.fillStyle = '#0a1218';
    ctx.fillRect(0, 0, width, height);

    // Draw Grid Lines - use subtle cyan
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.12)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < width; i += 50) { ctx.moveTo(i, 0); ctx.lineTo(i, height); }
    for (let j = 0; j < height; j += 50) { ctx.moveTo(0, j); ctx.lineTo(width, j); }
    ctx.stroke();

    if (!metrics || metrics.length < 2) {
        ctx.fillStyle = '#00d4ff';
        ctx.font = '14px Courier New';
        ctx.fillText('WAITING FOR DATA...', width / 2 - 80, height / 2);
        return;
    }

    const padding = 50;

    // Filter valid metrics
    const validMetrics = metrics.filter(m => m.loss !== undefined && !isNaN(m.loss));
    if (validMetrics.length < 2) return;

    // Sample if too many points - reduced to 200 for faster rendering
    let displayMetrics = validMetrics;
    if (validMetrics.length > 200) {
        const step = Math.ceil(validMetrics.length / 200);
        displayMetrics = validMetrics.filter((_, i) => i % step === 0);
    }

    const losses = displayMetrics.map(d => d.loss);
    const maxLoss = Math.max(...losses) * 1.1;
    const minLoss = Math.min(...losses) * 0.9;
    const lossRange = maxLoss - minLoss || 1;

    // Draw loss line - vibrant green
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 2;
    ctx.shadowColor = '#00ff88';
    ctx.shadowBlur = 4;
    ctx.beginPath();

    displayMetrics.forEach((d, i) => {
        const x = padding + (i / (displayMetrics.length - 1)) * (width - 2 * padding);
        const y = height - padding - ((d.loss - minLoss) / lossRange) * (height - 2 * padding);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Draw dots at intervals - amber color
    ctx.fillStyle = '#ffaa00';
    const dotInterval = Math.max(1, Math.floor(displayMetrics.length / 20));
    displayMetrics.forEach((d, i) => {
        if (i % dotInterval === 0 || i === displayMetrics.length - 1) {
            const x = padding + (i / (displayMetrics.length - 1)) * (width - 2 * padding);
            const y = height - padding - ((d.loss - minLoss) / lossRange) * (height - 2 * padding);
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();
        }
    });

    // Draw axis labels - cyan
    ctx.fillStyle = '#00d4ff';
    ctx.font = '11px Courier New';
    ctx.fillText(`MAX: ${maxLoss.toFixed(2)}`, 5, 15);
    ctx.fillText(`MIN: ${minLoss.toFixed(2)}`, 5, height - 5);
    ctx.fillText(`STEPS: ${validMetrics.length}`, width - 100, 15);

    // Draw title - amber accent
    ctx.fillStyle = '#ffaa00';
    ctx.font = 'bold 12px Courier New';
    ctx.fillText('TRAINING LOSS', width / 2 - 50, 15);
}

// ==================== MODEL ARCHITECTURE ====================
async function fetchArchitecture() {
    try {
        const res = await fetch(`${API_URL}/architecture`);
        const data = await res.json();
        renderModelArchitecture(data.architecture, data.log_file);
    } catch (error) {
        console.error('Error fetching architecture:', error);
        renderModelArchitecture(null, null);
    }
}

// ==================== EVALUATION RESULTS ====================
async function fetchEvalResults() {
    try {
        const limit = document.getElementById('eval-limit-input')?.value || 3;
        const res = await fetch(`${API_URL}/eval?limit=${limit}`);
        const data = await res.json();
        renderEvalResults(data);
    } catch (error) {
        console.error('Error fetching eval results:', error);
        renderEvalResults(null);
    }
}

function renderEvalResults(data) {
    const container = document.getElementById('eval-display');
    if (!container) return;

    if (!data || !data.all_results || data.all_results.length === 0) {
        container.innerHTML = `
            <div style="text-align: center; padding: 50px; color: #ffaa00;">
                <div style="font-size: 1.5rem; margin-bottom: 10px;">NO EVAL RESULTS</div>
                <div style="opacity: 0.7;">No evaluation results found in outputs_eval_* directories</div>
            </div>
        `;
        return;
    }

    const byDataset = data.by_dataset || {};

    // Build HTML for each dataset
    let html = `
        <div style="text-align: center; margin-bottom: 20px; padding: 10px; background: rgba(0,212,255,0.1); border: 1px solid #00d4ff;">
            <span style="color: #00d4ff;">Found <span style="color: #00ff88; font-weight: bold;">${data.total_evals}</span> evaluation runs</span>
        </div>
    `;

    for (const [dataset, results] of Object.entries(byDataset)) {
        const bestResult = results[0]; // Already sorted by hits@1

        html += `
            <div class="eval-dataset-section">
                <div class="eval-dataset-header">📊 ${dataset.toUpperCase()}</div>
        `;

        results.forEach((result, idx) => {
            const isBest = idx === 0;
            const h1Pct = (result['hits@1'] * 100).toFixed(1);
            const h3Pct = (result['hits@3'] * 100).toFixed(1);
            const h5Pct = (result['hits@5'] * 100).toFixed(1);
            const h10Pct = (result['hits@10'] * 100).toFixed(1);
            const mrrPct = (result.mrr * 100).toFixed(1);

            const cardId = `eval-${dataset}-${idx}`;

            html += `
                <div class="eval-result-card ${isBest ? 'best' : ''}">
                    <div class="eval-result-header">
                        <span class="eval-result-name">
                            ${isBest ? '🏆 ' : ''}${result.eval_name}
                        </span>
                        <span class="eval-result-total">${result.total} samples</span>
                    </div>
                    
                    <div class="eval-metrics-grid">
                        <div class="eval-metric">
                            <div class="eval-metric-label">Hits@1</div>
                            <div class="eval-metric-value ${isBest ? 'highlight' : ''}">${h1Pct}%</div>
                        </div>
                        <div class="eval-metric">
                            <div class="eval-metric-label">Hits@3</div>
                            <div class="eval-metric-value">${h3Pct}%</div>
                        </div>
                        <div class="eval-metric">
                            <div class="eval-metric-label">Hits@5</div>
                            <div class="eval-metric-value">${h5Pct}%</div>
                        </div>
                        <div class="eval-metric">
                            <div class="eval-metric-label">Hits@10</div>
                            <div class="eval-metric-value">${h10Pct}%</div>
                        </div>
                        <div class="eval-metric">
                            <div class="eval-metric-label">MRR</div>
                            <div class="eval-metric-value">${mrrPct}%</div>
                        </div>
                    </div>
                    
                    <div class="eval-score-bar">
                        <div class="eval-score-bar-fill" style="width: ${h1Pct}%"></div>
                    </div>
                    
                    ${result.examples && result.examples.length > 0 ? `
                        <div class="eval-examples-toggle" onclick="toggleExamples('${cardId}')">
                            ▶ Show sample predictions (${result.examples.length})
                        </div>
                        <div class="eval-examples" id="${cardId}-examples">
                            ${result.examples.map(ex => `
                                <div class="eval-example">
                                    <div class="eval-example-question"><span style="color: #00d4ff;">Q:</span> ${ex.question}</div>
                                    <div class="eval-example-path">
                                        <div style="margin-bottom: 4px;"><span style="color: #ffaa00; opacity: 0.8;">PRED:</span> ${(ex.top_pred_path || []).join(' → ')}</div>
                                        ${ex.gt_path && ex.gt_path.length > 0 ?
                    `<div><span style="color: #00ff88; opacity: 0.8;">TRUE:</span> ${(ex.gt_path).join(' → ')}</div>`
                    : ''}
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    ` : ''}
                </div>
            `;
        });

        html += '</div>';
    }

    container.innerHTML = html;
}

function toggleExamples(cardId) {
    const examples = document.getElementById(`${cardId}-examples`);
    const toggle = examples.previousElementSibling;
    if (examples.style.display === 'block') {
        examples.style.display = 'none';
        toggle.textContent = toggle.textContent.replace('▼', '▶');
    } else {
        examples.style.display = 'block';
        toggle.textContent = toggle.textContent.replace('▶', '▼');
    }
}

function formatParams(count) {
    if (count >= 1_000_000_000) return (count / 1_000_000_000).toFixed(1) + 'B';
    if (count >= 1_000_000) return (count / 1_000_000).toFixed(1) + 'M';
    if (count >= 1_000) return (count / 1_000).toFixed(1) + 'K';
    return count.toString();
}


function renderModelArchitecture(arch, logFile) {
    const container = document.getElementById('model-diagram');
    if (!container) return;

    if (!arch || arch.error) {
        container.innerHTML = `
            <div style="text-align: center; padding: 50px; color: #ffaa00;">
                <div style="font-size: 1.5rem; margin-bottom: 10px;">NO MODEL DATA</div>
                <div style="opacity: 0.7;">Waiting for training log...</div>
            </div>
        `;
        return;
    }

    const modelName = arch.model_name || 'Unknown Model';
    const modelType = arch.model_type || 'unknown';
    const totalParams = arch.total_params || 0;
    const trainableParams = arch.trainable_params || 0;
    const frozenParams = arch.frozen_params || totalParams - trainableParams;
    const modules = arch.modules || [];
    const hiddenDim = arch.hidden_dim || 768;
    const diffusionSteps = arch.diffusion_steps || 10;
    const numHeads = arch.num_heads || 8;
    const dropout = arch.dropout || 0.1;
    const encoderName = arch.encoder_name || 'unknown';
    const batchSize = arch.batch_size || 0;
    const learningRate = arch.learning_rate || '0';
    const trainingConfig = arch.training_config || {};


    // Color mapping for module types
    const typeColors = {
        'encoder': { bg: 'rgba(76, 175, 80, 0.25)', border: '#4CAF50' },
        'attention': { bg: 'rgba(255, 152, 0, 0.25)', border: '#FF9800' },
        'diffusion': { bg: 'rgba(156, 39, 176, 0.25)', border: '#9C27B0' },
        'mlp': { bg: 'rgba(244, 67, 54, 0.25)', border: '#F44336' },
    };

    // Generate interactive module blocks
    let moduleBlocks = modules.map((mod, idx) => {
        const colors = typeColors[mod.type] || { bg: 'rgba(100,100,100,0.25)', border: '#888' };
        const frozenBadge = mod.frozen ? '<span style="color: #FF9800; font-size: 0.7rem;"> [FROZEN]</span>' : '';

        // Generate layer details
        const layerDetails = (mod.layers || []).map(layer => `
            <div style="display: flex; justify-content: space-between; padding: 4px 8px; border-bottom: 1px solid rgba(255,255,255,0.1);">
                <span style="color: #00d4ff;">${layer.name}</span>
                <span style="color: #aaa; font-size: 0.8rem;">${layer.spec}</span>
            </div>
        `).join('');

        return `
            <div class="module-block" data-module="${idx}" style="margin-bottom: 10px;">
                <div class="module-header" onclick="toggleModule(${idx})" style="
                    background: ${colors.bg};
                    border: 2px solid ${colors.border};
                    padding: 12px 15px;
                    cursor: pointer;
                    transition: all 0.2s;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                ">
                    <div>
                        <div style="font-weight: bold; font-size: 1rem; color: ${colors.border};">
                            ${mod.name}${frozenBadge}
                            <span style="float: right; font-size: 0.8rem; color: #00ff88; margin-left: 15px;">${mod.params}</span>
                        </div>
                        <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 3px;">${mod.description || ''}</div>
                    </div>
                    <div class="expand-icon" id="expand-${idx}" style="font-size: 1.2rem; color: ${colors.border};">+</div>
                </div>
                <div class="module-layers" id="layers-${idx}" style="
                    display: none;
                    background: rgba(0,0,0,0.3);
                    border: 1px solid ${colors.border};
                    border-top: none;
                    font-size: 0.85rem;
                    font-family: 'Courier New', monospace;
                ">
                    ${layerDetails}
                </div>
            </div>
            ${idx < modules.length - 1 ? '<div style="text-align: center; color: #00d4ff; font-size: 1.2rem; margin: 5px 0;">|</div>' : ''}
        `;
    }).join('');

    // Extra config for DCDR model
    const numLayers = arch.num_layers || null;
    const swapRate = arch.swap_rate || null;
    const plTemp = arch.pl_temperature || null;

    container.innerHTML = `
        <!-- Model Header -->
        <div style="text-align: center; margin-bottom: 20px; padding: 15px; border: 2px solid #00d4ff; background: rgba(0,212,255,0.1);">
            <div style="font-size: 1.4rem; font-weight: bold; color: #00d4ff;">${modelName}</div>
            <div style="font-size: 0.75rem; color: #888; margin-top: 3px;">${modelType.toUpperCase()}</div>
            <div style="font-size: 0.85rem; margin-top: 10px; display: flex; justify-content: center; gap: 15px; flex-wrap: wrap;">
                <span>Encoder: <span style="color: #00ff88;">${encoderName}</span></span>
            </div>
            <div style="font-size: 0.85rem; margin-top: 8px; display: flex; justify-content: center; gap: 15px; flex-wrap: wrap;">
                <span>Hidden: <span style="color: #00ff88;">${hiddenDim}</span></span>
                ${numLayers ? `<span>Layers: <span style="color: #00ff88;">${numLayers}</span></span>` : ''}
                <span>Steps: <span style="color: #00ff88;">${diffusionSteps}</span></span>
                <span>Batch: <span style="color: #00ff88;">${batchSize}</span></span>
                <span>LR: <span style="color: #00ff88;">${learningRate}</span></span>
            </div>
            ${swapRate ? `
            <div style="font-size: 0.8rem; margin-top: 6px; color: #9C27B0;">
                Swap Rate: ${swapRate} | PL Temp: ${plTemp || 1.0}
            </div>
            ` : ''}
        </div>
        
        <div style="font-size: 0.85rem; text-align: center; margin-bottom: 15px; color: #888;">
            Click on any module to expand/collapse layer details
        </div>
        
        <!-- Module Flow -->
        ${moduleBlocks}
        
        <!-- Summary Box -->
        <div style="margin-top: 20px; padding: 15px; border: 1px solid #00d4ff; background: rgba(0,212,255,0.05);">
            <div style="font-weight: bold; margin-bottom: 10px; color: #ffaa00;">MODEL SUMMARY</div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; font-size: 0.9rem;">
                <div>Total Parameters: <span style="color: #00ff88;">${formatParams(totalParams)}</span></div>
                <div>Trainable: <span style="color: #4CAF50;">${formatParams(trainableParams)}</span></div>
                <div>Frozen (Encoder): <span style="color: #FF9800;">${formatParams(frozenParams)}</span></div>
                <div>Modules: <span style="color: #00d4ff;">${modules.length}</span></div>
            </div>
            ${trainingConfig.gpus ? `
            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(0,212,255,0.3); font-size: 0.85rem;">
                <span style="color: #888;">Training: ${trainingConfig.per_gpu_batch || '?'} batch/GPU × ${trainingConfig.gpus || '?'} GPUs</span>
            </div>
            ` : ''}
        </div>
        
        <!-- Legend -->
        <div style="margin-top: 15px; display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; font-size: 0.8rem;">
            <span><span style="display: inline-block; width: 12px; height: 12px; background: #4CAF50; margin-right: 5px;"></span>Encoder</span>
            <span><span style="display: inline-block; width: 12px; height: 12px; background: #FF9800; margin-right: 5px;"></span>Attention</span>
            <span><span style="display: inline-block; width: 12px; height: 12px; background: #9C27B0; margin-right: 5px;"></span>Diffusion</span>
            <span><span style="display: inline-block; width: 12px; height: 12px; background: #F44336; margin-right: 5px;"></span>MLP</span>
        </div>
    `;
}

// Toggle module expansion
function toggleModule(idx) {
    const layers = document.getElementById(`layers-${idx}`);
    const icon = document.getElementById(`expand-${idx}`);
    if (layers.style.display === 'none') {
        layers.style.display = 'block';
        icon.textContent = '-';
    } else {
        layers.style.display = 'none';
        icon.textContent = '+';
    }
}

// Expand/collapse all modules
function expandAllModules() {
    document.querySelectorAll('.module-layers').forEach(el => {
        el.style.display = 'block';
    });
    document.querySelectorAll('.expand-icon').forEach(el => {
        el.textContent = '-';
    });
}

function collapseAllModules() {
    document.querySelectorAll('.module-layers').forEach(el => {
        el.style.display = 'none';
    });
    document.querySelectorAll('.expand-icon').forEach(el => {
        el.textContent = '+';
    });
}

// ==================== LOGS INTERACTIVITY ====================
let autoScrollEnabled = true;
let currentLogFilter = 'all';
let currentSearchTerm = '';
let rawLogs = [];

function updateLogsDisplay() {
    const logsDiv = document.getElementById('logs-display');
    if (!rawLogs.length) return;

    let filteredLogs = rawLogs;

    // Apply level filter
    if (currentLogFilter !== 'all') {
        filteredLogs = rawLogs.filter(line => {
            const lower = line.toLowerCase();
            if (currentLogFilter === 'error') return lower.includes('error') || lower.includes('exception');
            if (currentLogFilter === 'warning') return lower.includes('warning') || lower.includes('warn');
            if (currentLogFilter === 'info') return lower.includes('info');
            return true;
        });
    }

    // Apply search filter
    if (currentSearchTerm) {
        const term = currentSearchTerm.toLowerCase();
        filteredLogs = filteredLogs.filter(line => line.toLowerCase().includes(term));
    }

    // Render with highlighting
    const html = filteredLogs.map(line => {
        let className = '';
        const lower = line.toLowerCase();
        if (lower.includes('error') || lower.includes('exception')) className = 'log-line-error';
        else if (lower.includes('warning') || lower.includes('warn')) className = 'log-line-warning';

        // Highlight search term
        let displayLine = line;
        if (currentSearchTerm) {
            const regex = new RegExp(`(${currentSearchTerm})`, 'gi');
            displayLine = line.replace(regex, '<span class="log-highlight">$1</span>');
        }

        return `<div class="${className}">${displayLine}</div>`;
    }).join('');

    logsDiv.innerHTML = html;

    // Update counts
    document.getElementById('log-count').textContent = `${rawLogs.length} lines`;
    const filteredCount = document.getElementById('filtered-count');
    if (filteredLogs.length !== rawLogs.length) {
        filteredCount.textContent = `(showing ${filteredLogs.length})`;
    } else {
        filteredCount.textContent = '';
    }

    // Auto-scroll
    if (autoScrollEnabled) {
        logsDiv.scrollTop = logsDiv.scrollHeight;
    }
}

// Log search handler
document.getElementById('log-search')?.addEventListener('input', (e) => {
    currentSearchTerm = e.target.value;
    updateLogsDisplay();
});

// Clear search
document.getElementById('btn-clear-search')?.addEventListener('click', () => {
    document.getElementById('log-search').value = '';
    currentSearchTerm = '';
    updateLogsDisplay();
});

// Auto-scroll toggle
document.getElementById('btn-auto-scroll')?.addEventListener('click', (e) => {
    autoScrollEnabled = !autoScrollEnabled;
    e.target.classList.toggle('active', autoScrollEnabled);
});

// Log level filter
document.getElementById('log-level-filter')?.addEventListener('change', (e) => {
    currentLogFilter = e.target.value;
    updateLogsDisplay();
});

// ==================== CHART INTERACTIVITY ====================
let chartData = [];

function setupChartInteraction() {
    const canvas = document.getElementById('metricsCanvas');
    if (!canvas) return;

    const tooltip = document.getElementById('chart-tooltip');

    canvas.addEventListener('mousemove', (e) => {
        if (!chartData.length) return;

        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const padding = 50;
        const width = rect.width;

        if (x < padding || x > width - padding) {
            tooltip.style.display = 'none';
            return;
        }

        const ratio = (x - padding) / (width - 2 * padding);
        const idx = Math.min(Math.floor(ratio * chartData.length), chartData.length - 1);
        const data = chartData[idx];

        if (data) {
            document.getElementById('tooltip-epoch').textContent = data.epoch;
            document.getElementById('tooltip-loss').textContent = data.loss.toFixed(4);
            document.getElementById('tooltip-acc').textContent = (data.accuracy * 100).toFixed(1) + '%';

            tooltip.style.display = 'block';
            tooltip.style.left = (e.clientX + 15) + 'px';
            tooltip.style.top = (e.clientY - 10) + 'px';
        }
    });

    canvas.addEventListener('mouseleave', () => {
        tooltip.style.display = 'none';
    });
}

// ==================== EXPORT FUNCTIONALITY ====================
function exportMetrics() {
    if (!window.lastMetrics || !window.lastMetrics.length) {
        alert('No metrics data to export');
        return;
    }

    const csv = 'epoch,loss,accuracy\n' +
        window.lastMetrics.map(m => `${m.epoch},${m.loss},${m.accuracy}`).join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `training_metrics_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
}

// ==================== KEYBOARD SHORTCUTS ====================
function showShortcutsModal() {
    document.getElementById('shortcuts-modal').style.display = 'flex';
}

function closeShortcutsModal() {
    document.getElementById('shortcuts-modal').style.display = 'none';
}

document.addEventListener('keydown', (e) => {
    // Don't trigger if typing in input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        if (e.key === 'Escape') e.target.blur();
        return;
    }

    switch (e.key) {
        case '1':
            document.querySelector('[data-tab="metrics-tab"]').click();
            break;
        case '2':
            document.querySelector('[data-tab="architecture-tab"]').click();
            break;
        case '3':
            document.querySelector('[data-tab="eval-tab"]').click();
            break;
        case '4':
            document.querySelector('[data-tab="logs-tab"]').click();
            break;
        case 'r':
        case 'R':
            fetchData();
            fetchArchitecture();
            fetchEvalResults();
            break;
        case 'e':
        case 'E':
            exportMetrics();
            break;
        case 'a':
        case 'A':
            expandAllModules();
            break;
        case 'c':
        case 'C':
            collapseAllModules();
            break;
        case '/':
            e.preventDefault();
            document.getElementById('log-search')?.focus();
            break;
        case '?':
            showShortcutsModal();
            break;
        case 'Escape':
            closeShortcutsModal();
            break;
    }
});

// ==================== FOOTER BUTTONS ====================
document.getElementById('btn-refresh')?.addEventListener('click', () => {
    fetchData();
    fetchArchitecture();
    fetchEvalResults();
});

document.getElementById('btn-export')?.addEventListener('click', exportMetrics);
document.getElementById('btn-shortcuts')?.addEventListener('click', showShortcutsModal);

// ==================== INITIALIZATION ====================
// Initial load
fetchData();
fetchArchitecture();
fetchEvalResults();

// Polling interval - every 5 seconds
setInterval(fetchData, 5000);

// Refresh architecture every 30 seconds
setInterval(fetchArchitecture, 30000);

// Refresh eval every 60 seconds
setInterval(fetchEvalResults, 60000);

// Clock
setInterval(() => {
    document.getElementById('sys-time').innerText = new Date().toLocaleTimeString();
}, 1000);

// Handle window resize
window.addEventListener('resize', () => {
    if (window.lastMetrics) {
        drawChart(window.lastMetrics);
    }
});

// Setup chart interaction after DOM ready
setTimeout(setupChartInteraction, 500);

// Store logs for filtering
const originalUpdateDashboard = updateDashboard;
updateDashboard = function (metrics, config, logs) {
    if (logs) {
        rawLogs = logs;
        updateLogsDisplay();

        // Update last update timestamp
        const lastUpdate = document.getElementById('last-update');
        if (lastUpdate) {
            lastUpdate.textContent = `Last update: ${new Date().toLocaleTimeString()}`;
        }
    }

    if (metrics && metrics.length) {
        chartData = metrics;
    }

    // Call original but skip logs (we handle them separately)
    if (metrics && metrics.length > 0) {
        window.lastMetrics = metrics;
        const last = metrics[metrics.length - 1];
        document.getElementById('val-loss').innerText = last.loss.toFixed(4);
        document.getElementById('val-acc').innerText = (last.accuracy * 100).toFixed(1) + '%';
        document.getElementById('val-epoch').innerText = last.epoch;
        drawChart(metrics);
    }

    if (config) {
        const configHtml = Object.entries(config)
            .map(([k, v]) => `<div><span style="opacity:0.7">${k}:</span> ${v}</div>`)
            .join('');
        document.getElementById('config-display').innerHTML = configHtml;
    }
};
// ==================== EVAL CONTROLS ====================
document.getElementById('btn-refresh-eval')?.addEventListener('click', fetchEvalResults);
document.getElementById('eval-limit-input')?.addEventListener('change', fetchEvalResults);
