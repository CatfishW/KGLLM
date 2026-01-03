/**
 * PAPER READER - 1980s Electric Pixel Style
 * Frontend Application with Enhanced Mobile & Desktop Support
 */

// Configuration - detect environment and set API base URL
const isLocalDev = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const API_BASE = isLocalDev
    ? 'http://127.0.0.1:22222'
    : `${window.location.protocol}//${window.location.hostname}/paperreader`;

// State
let papers = [];
let sortOrder = 'desc';
let viewMode = 'grid';
let currentPaper = null;
let currentPaperIndex = -1;
let touchStartY = 0;
let isPulling = false;
let presenceInterval = null;

// DOM Elements
const papersGrid = document.getElementById('papers-grid');
const loading = document.getElementById('loading');
const emptyState = document.getElementById('empty-state');
const paperCount = document.getElementById('paper-count');
const searchInput = document.getElementById('search-input');
const clearSearch = document.getElementById('clear-search');
const sortBySelect = document.getElementById('sort-by');
const sortOrderBtn = document.getElementById('sort-order');
const viewGridBtn = document.getElementById('view-grid');
const viewListBtn = document.getElementById('view-list');
const refreshBtn = document.getElementById('refresh-btn');
const pdfModal = document.getElementById('pdf-modal');
const modalTitle = document.getElementById('modal-paper-title');
const pdfViewer = document.getElementById('pdf-viewer');
const btnDownload = document.getElementById('btn-download');
const btnFullscreen = document.getElementById('btn-fullscreen');
const btnCloseModal = document.getElementById('btn-close-modal');
const infoYear = document.getElementById('info-year');
const infoPages = document.getElementById('info-pages');
const infoSize = document.getElementById('info-size');

/**
 * Initialize the application
 */
async function init() {
    setupEventListeners();
    if (searchInput) {
        searchInput.value = '';
    }
    setupMobileInteractions();
    ThemeController.init();
    AuthController.init();
    ProjectManager.init();
    setupEditor(); // Initialize Editor
    ProjectManager.onProjectChange = (project) => {
        EditorController.handleProjectSwitch();
        TeamChatController.setProject(project);
    };
    ProjectManager.applyActiveProject();
    setupKeyboardShortcuts();
    setupButtonAnimations();
    showShortcutsHint();
    PresenceController.init(); // Initialize Presence tracking
    await loadPapers();
}

/**
 * Setup button click animations (ripple effect and feedback)
 */
function setupButtonAnimations() {
    // Add ripple effect to all buttons
    document.addEventListener('click', (e) => {
        const button = e.target.closest('button, .btn-view, .nav-tab, .file-item');
        if (!button) return;

        // Create ripple element
        const ripple = document.createElement('span');
        ripple.className = 'btn-ripple';

        const rect = button.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = e.clientX - rect.left - size / 2;
        const y = e.clientY - rect.top - size / 2;

        ripple.style.cssText = `
            position: absolute;
            width: ${size}px;
            height: ${size}px;
            left: ${x}px;
            top: ${y}px;
            background: radial-gradient(circle, rgba(255,255,255,0.4) 0%, transparent 70%);
            border-radius: 50%;
            transform: scale(0);
            animation: btn-ripple-anim 0.4s ease-out forwards;
            pointer-events: none;
        `;

        // Ensure button has relative positioning
        const computedStyle = window.getComputedStyle(button);
        if (computedStyle.position === 'static') {
            button.style.position = 'relative';
        }
        button.style.overflow = 'hidden';

        button.appendChild(ripple);

        // Clean up ripple after animation
        setTimeout(() => ripple.remove(), 400);
    });

    // Add ripple animation keyframes if not already present
    if (!document.getElementById('ripple-keyframes')) {
        const style = document.createElement('style');
        style.id = 'ripple-keyframes';
        style.textContent = `
            @keyframes btn-ripple-anim {
                0% { transform: scale(0); opacity: 1; }
                100% { transform: scale(2.5); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    }
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Search
    let searchTimeout;
    searchInput.addEventListener('input', (e) => {
        clearTimeout(searchTimeout);
        clearSearch.classList.toggle('visible', e.target.value.length > 0);
        searchTimeout = setTimeout(() => loadPapers(), 300);
    });

    clearSearch.addEventListener('click', () => {
        searchInput.value = '';
        clearSearch.classList.remove('visible');
        loadPapers();
    });

    // Sort
    sortBySelect.addEventListener('change', () => loadPapers());

    sortOrderBtn.addEventListener('click', () => {
        sortOrder = sortOrder === 'desc' ? 'asc' : 'desc';
        const icon = sortOrderBtn.querySelector('i');
        icon.className = sortOrder === 'desc' ? 'bi bi-sort-down' : 'bi bi-sort-up';
        loadPapers();
    });

    // View mode
    viewGridBtn.addEventListener('click', () => setViewMode('grid'));
    viewListBtn.addEventListener('click', () => setViewMode('list'));

    // Refresh
    refreshBtn.addEventListener('click', () => {
        refreshBtn.querySelector('i').style.animation = 'spin 0.5s linear';
        setTimeout(() => {
            refreshBtn.querySelector('i').style.animation = '';
        }, 500);
        loadPapers();
    });

    // Modal
    btnCloseModal.addEventListener('click', closeModal);
    pdfModal.querySelector('.modal-backdrop').addEventListener('click', closeModal);

    btnDownload.addEventListener('click', downloadCurrentPaper);

    btnFullscreen.addEventListener('click', toggleFullscreen);

    // Navigation buttons
    const btnPrev = document.getElementById('btn-prev');
    const btnNext = document.getElementById('btn-next');

    if (btnPrev) {
        btnPrev.addEventListener('click', () => navigatePaper(-1));
    }
    if (btnNext) {
        btnNext.addEventListener('click', () => navigatePaper(1));
    }
}

/**
 * Setup mobile touch interactions
 */
function setupMobileInteractions() {
    // Pull to refresh
    let pullIndicator = document.createElement('div');
    pullIndicator.className = 'pull-to-refresh';
    pullIndicator.innerHTML = '<i class="bi bi-arrow-down-circle"></i> <span>Pull to refresh</span>';
    document.body.insertBefore(pullIndicator, document.body.firstChild);

    document.addEventListener('touchstart', (e) => {
        if (window.scrollY === 0 && !pdfModal.classList.contains('active')) {
            touchStartY = e.touches[0].clientY;
        }
    }, { passive: true });

    document.addEventListener('touchmove', (e) => {
        if (touchStartY === 0 || pdfModal.classList.contains('active')) return;

        const touchY = e.touches[0].clientY;
        const diff = touchY - touchStartY;

        if (diff > 50 && window.scrollY === 0) {
            isPulling = true;
            pullIndicator.classList.add('active');
            pullIndicator.style.transform = `translateY(${Math.min(diff - 50, 60)}px)`;
        }
    }, { passive: true });

    document.addEventListener('touchend', () => {
        if (isPulling) {
            pullIndicator.classList.remove('active');
            pullIndicator.style.transform = '';
            loadPapers();
            showToast('Refreshing papers...');
        }
        touchStartY = 0;
        isPulling = false;
    });

    // Swipe to navigate in modal
    let modalTouchStartX = 0;

    pdfModal.addEventListener('touchstart', (e) => {
        modalTouchStartX = e.touches[0].clientX;
    }, { passive: true });

    pdfModal.addEventListener('touchend', (e) => {
        if (!pdfModal.classList.contains('active')) return;

        const touchEndX = e.changedTouches[0].clientX;
        const diff = touchEndX - modalTouchStartX;

        // Swipe threshold
        if (Math.abs(diff) > 100) {
            if (diff > 0) {
                // Swipe right - previous paper
                navigatePaper(-1);
            } else {
                // Swipe left - next paper
                navigatePaper(1);
            }
        }
    });

    // Double tap to zoom (for mobile PDF viewing)
    let lastTap = 0;
    pdfModal.addEventListener('touchend', (e) => {
        const currentTime = new Date().getTime();
        const tapLength = currentTime - lastTap;

        if (tapLength < 300 && tapLength > 0) {
            toggleFullscreen();
        }
        lastTap = currentTime;
    });
}

/**
 * Setup keyboard shortcuts
 */
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        const activeEl = document.activeElement;
        const tag = activeEl?.tagName?.toLowerCase();
        const isFormField = ['input', 'textarea', 'select', 'button'].includes(tag) || activeEl?.isContentEditable;
        const isSearchFocused = activeEl === searchInput;
        const inEditorView = EditorController?.elements?.editorView?.style.display !== 'none';

        // Global shortcuts
        if (e.key === 'Escape') {
            if (pdfModal.classList.contains('active')) {
                closeModal();
            } else if (AuthController?.elements?.modal?.classList.contains('active')) {
                AuthController.close();
            } else if (ProjectManager?.elements?.modal?.classList.contains('active')) {
                ProjectManager.closeModal();
            } else if (TeamChatController?.isOpen) {
                TeamChatController.close();
            } else if (isSearchFocused) {
                searchInput.blur();
            }
            return;
        }

        // Modal navigation shortcuts
        if (pdfModal.classList.contains('active')) {
            switch (e.key) {
                case 'ArrowLeft':
                case 'k':
                case 'K':
                    e.preventDefault();
                    navigatePaper(-1);
                    break;
                case 'ArrowRight':
                case 'j':
                case 'J':
                    e.preventDefault();
                    navigatePaper(1);
                    break;
                case 'd':
                case 'D':
                    e.preventDefault();
                    downloadCurrentPaper();
                    break;
                case 'f':
                case 'F':
                    e.preventDefault();
                    toggleFullscreen();
                    break;
            }
            return;
        }

        // Editor/library hotkeys (use non-conflicting combo)
        if (!isFormField) {
            if (e.ctrlKey && e.shiftKey && (e.key === 'e' || e.key === 'E')) {
                e.preventDefault();
                EditorController.switchTab('editor');
                EditorController.loadFiles();
                return;
            }

            if (e.ctrlKey && e.shiftKey && (e.key === 'l' || e.key === 'L')) {
                e.preventDefault();
                EditorController.switchTab('library');
                return;
            }

            if (inEditorView && e.ctrlKey && e.shiftKey && (e.key === 'd' || e.key === 'D')) {
                e.preventDefault();
                EditorController.downloadCompiled();
                return;
            }
        }

        // Don't trigger global shortcuts when typing in form fields or editor
        if (isFormField) return;

        switch (e.key) {
            // Search focus
            case '/':
                e.preventDefault();
                searchInput.focus();
                break;

            // Refresh
            case 'r':
            case 'R':
                e.preventDefault();
                refreshBtn.click();
                break;

            // Sort toggle
            case 's':
            case 'S':
                e.preventDefault();
                sortOrderBtn.click();
                break;

            // View mode toggle
            case 'v':
            case 'V':
                e.preventDefault();
                if (viewMode === 'grid') {
                    setViewMode('list');
                } else {
                    setViewMode('grid');
                }
                break;

            // Sort by cycling
            case '1':
                e.preventDefault();
                sortBySelect.value = 'year';
                loadPapers();
                showToast('Sorted by year');
                break;
            case '2':
                e.preventDefault();
                sortBySelect.value = 'title';
                loadPapers();
                showToast('Sorted by title');
                break;
            case '3':
                e.preventDefault();
                sortBySelect.value = 'size';
                loadPapers();
                showToast('Sorted by size');
                break;

            // Open first paper
            case 'Enter':
                e.preventDefault();
                if (papers.length > 0) {
                    openPaper(papers[0].id);
                }
                break;

            // Show help
            case '?':
            case 'h':
            case 'H':
                e.preventDefault();
                showHelpModal();
                break;

            // Navigate papers with number keys (1-9 opens that paper)
            case '0':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
                // Quick open papers 1-9
                const idx = parseInt(e.key) - 1;
                if (idx >= 0 && idx < papers.length && idx < 9) {
                    e.preventDefault();
                    openPaper(papers[idx].id);
                }
                break;
        }
    });
}

/**
 * Navigate to previous/next paper in modal
 */
function navigatePaper(direction) {
    if (!currentPaper || papers.length === 0) return;

    let newIndex = currentPaperIndex + direction;

    // Wrap around
    if (newIndex < 0) newIndex = papers.length - 1;
    if (newIndex >= papers.length) newIndex = 0;

    const newPaper = papers[newIndex];
    if (newPaper) {
        openPaper(newPaper.id);
        showToast(`Paper ${newIndex + 1} of ${papers.length}`);
    }
}

/**
 * Download current paper
 */
function downloadCurrentPaper() {
    if (currentPaper) {
        const link = document.createElement('a');
        link.href = `${API_BASE}${currentPaper.pdf_url}/download`;
        link.download = currentPaper.filename;
        link.click();
        showToast('Downloading...');
    }
}

/**
 * Toggle fullscreen
 */
function toggleFullscreen() {
    const container = pdfModal.querySelector('.modal-content');
    if (document.fullscreenElement) {
        document.exitFullscreen();
    } else {
        container.requestFullscreen().catch(() => {
            // Fallback for mobile
            pdfModal.classList.toggle('fullscreen-mode');
        });
    }
}

/**
 * Show toast notification
 */
function showToast(message) {
    // Remove existing toast
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    document.body.appendChild(toast);

    // Trigger animation
    setTimeout(() => toast.classList.add('show'), 10);

    // Remove after delay
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 2000);
}

/**
 * Show shortcuts hint on first load
 */
function showShortcutsHint() {
    const hasSeenHint = localStorage.getItem('paperreader-hint-seen');
    if (hasSeenHint) return;

    setTimeout(() => {
        showToast('Press ? for keyboard shortcuts');
        localStorage.setItem('paperreader-hint-seen', 'true');
    }, 2000);
}

/**
 * Show help modal with keyboard shortcuts
 */
function showHelpModal() {
    // Remove existing help
    const existing = document.querySelector('.help-modal');
    if (existing) {
        existing.remove();
        return;
    }

    const helpModal = document.createElement('div');
    helpModal.className = 'help-modal';
    helpModal.innerHTML = `
        <div class="help-backdrop"></div>
        <div class="help-content">
            <div class="help-header">
                <h2><i class="bi bi-keyboard"></i> KEYBOARD SHORTCUTS</h2>
                <button class="help-close"><i class="bi bi-x-lg"></i></button>
            </div>
            <div class="help-body">
                <div class="shortcut-group">
                    <h3>NAVIGATION</h3>
                    <div class="shortcut"><kbd>/</kbd> <span>Focus search</span></div>
                    <div class="shortcut"><kbd>ESC</kbd> <span>Close modal / Clear focus</span></div>
                    <div class="shortcut"><kbd>Enter</kbd> <span>Open first paper</span></div>
                </div>
                <div class="shortcut-group">
                    <h3>VIEW CONTROLS</h3>
                    <div class="shortcut"><kbd>R</kbd> <span>Refresh papers</span></div>
                    <div class="shortcut"><kbd>V</kbd> <span>Toggle grid/list view</span></div>
                    <div class="shortcut"><kbd>S</kbd> <span>Toggle sort order</span></div>
                    <div class="shortcut"><kbd>1</kbd> <span>Sort by year</span></div>
                    <div class="shortcut"><kbd>2</kbd> <span>Sort by title</span></div>
                    <div class="shortcut"><kbd>3</kbd> <span>Sort by size</span></div>
                </div>
                <div class="shortcut-group">
                    <h3>PDF VIEWER</h3>
                    <div class="shortcut"><kbd>←</kbd> / <kbd>K</kbd> <span>Previous paper</span></div>
                    <div class="shortcut"><kbd>→</kbd> / <kbd>J</kbd> <span>Next paper</span></div>
                    <div class="shortcut"><kbd>D</kbd> <span>Download PDF</span></div>
                    <div class="shortcut"><kbd>F</kbd> <span>Toggle fullscreen</span></div>
                </div>
                <div class="shortcut-group mobile-hint">
                    <h3>MOBILE GESTURES</h3>
                    <div class="shortcut"><span>Pull down</span> <span>Refresh papers</span></div>
                    <div class="shortcut"><span>Swipe L/R</span> <span>Navigate papers</span></div>
                    <div class="shortcut"><span>Double tap</span> <span>Toggle fullscreen</span></div>
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(helpModal);

    // Close handlers
    helpModal.querySelector('.help-backdrop').addEventListener('click', () => helpModal.remove());
    helpModal.querySelector('.help-close').addEventListener('click', () => helpModal.remove());

    // Animate in
    setTimeout(() => helpModal.classList.add('active'), 10);
}

/**
 * Load papers from API
 */
async function loadPapers() {
    showLoading(true);

    try {
        const params = new URLSearchParams({
            sort_by: sortBySelect.value,
            sort_order: sortOrder
        });

        const search = searchInput.value.trim();
        if (search) {
            params.append('search', search);
        }

        const response = await fetch(`${API_BASE}/api/papers?${params}`);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        papers = data.papers;
        paperCount.textContent = data.total;

        renderPapers();
    } catch (error) {
        console.error('Failed to load papers:', error);
        showError('Failed to connect to server');
    } finally {
        showLoading(false);
    }
}

/**
 * Render papers grid
 */
function renderPapers() {
    if (papers.length === 0) {
        papersGrid.innerHTML = '';
        emptyState.style.display = 'flex';
        return;
    }

    emptyState.style.display = 'none';

    papersGrid.innerHTML = papers.map((paper, index) => `
        <div class="paper-card" data-id="${paper.id}" data-index="${index}" onclick="openPaper('${paper.id}')">
            <div class="paper-thumbnail">
                <i class="bi bi-file-earmark-pdf loading-thumb"></i>
                ${paper.year ? `<span class="paper-year-badge">${paper.year}</span>` : ''}
            </div>
            <div class="paper-info">
                <h3 class="paper-title" title="${escapeHtml(paper.title)}">${escapeHtml(paper.title)}</h3>
                <div class="paper-meta">
                    <span><i class="bi bi-file-earmark"></i> ${paper.pages} pages</span>
                    <span><i class="bi bi-hdd"></i> ${paper.size_mb} MB</span>
                </div>
            </div>
        </div>
    `).join('');

    // Load thumbnails
    papers.forEach(paper => {
        loadThumbnail(paper);
    });
}

/**
 * Load thumbnail for a paper
 */
async function loadThumbnail(paper) {
    const card = document.querySelector(`.paper-card[data-id="${paper.id}"]`);
    if (!card) return;

    const container = card.querySelector('.paper-thumbnail');
    const img = new Image();

    img.onload = () => {
        container.innerHTML = `
            ${paper.year ? `<span class="paper-year-badge">${paper.year}</span>` : ''}
        `;
        container.insertBefore(img, container.firstChild);
    };

    img.onerror = () => {
        container.querySelector('.loading-thumb').className = 'bi bi-file-earmark-pdf';
        container.querySelector('.loading-thumb').style.animation = 'none';
    };

    img.src = `${API_BASE}${paper.thumbnail_url}`;
    img.alt = paper.title;
}

/**
 * Open paper in modal
 */
function openPaper(id) {
    const index = papers.findIndex(p => p.id === id);
    const paper = papers[index];
    if (!paper) return;

    currentPaper = paper;
    currentPaperIndex = index;

    modalTitle.textContent = paper.title;
    pdfViewer.src = `${API_BASE}${paper.pdf_url}`;

    infoYear.textContent = paper.year || '-';
    infoPages.textContent = paper.pages;
    infoSize.textContent = `${paper.size_mb} MB`;

    pdfModal.classList.add('active');
    document.body.style.overflow = 'hidden';

    // Hide screen effects for better PDF reading
    document.querySelector('.scanlines')?.classList.add('hidden');
    document.querySelector('.crt-flicker')?.classList.add('hidden');
}

/**
 * Close modal
 */
function closeModal() {
    pdfModal.classList.remove('active');
    pdfModal.classList.remove('fullscreen-mode');
    pdfViewer.src = '';
    currentPaper = null;
    currentPaperIndex = -1;
    document.body.style.overflow = '';

    if (document.fullscreenElement) {
        document.exitFullscreen();
    }

    // Restore screen effects
    document.querySelector('.scanlines')?.classList.remove('hidden');
    document.querySelector('.crt-flicker')?.classList.remove('hidden');
}

/**
 * Set view mode
 */
function setViewMode(mode) {
    viewMode = mode;

    viewGridBtn.classList.toggle('active', mode === 'grid');
    viewListBtn.classList.toggle('active', mode === 'list');

    papersGrid.classList.toggle('list-view', mode === 'list');

    showToast(`${mode.charAt(0).toUpperCase() + mode.slice(1)} view`);
}

/**
 * Show/hide loading state
 */
function showLoading(show) {
    loading.style.display = show ? 'flex' : 'none';
    papersGrid.style.display = show ? 'none' : 'grid';
}

/**
 * Show error message
 */
function showError(message) {
    emptyState.querySelector('h3').textContent = 'CONNECTION ERROR';
    emptyState.querySelector('p').textContent = message;
    emptyState.querySelector('i').className = 'bi bi-exclamation-triangle';
    emptyState.style.display = 'flex';
    papersGrid.innerHTML = '';
}

/**
 * Escape HTML
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Theme Controller - switch between UI themes
 */
const ThemeController = {
    themes: [
        { id: 'neon', label: 'NEON' },
        { id: 'light', label: 'LIGHT' },
        { id: 'dusk', label: 'DUSK' }
    ],
    currentTheme: 'neon',
    elements: {
        button: null,
        label: null
    },

    init() {
        this.elements.button = document.getElementById('btn-theme');
        this.elements.label = document.getElementById('theme-label');
        if (!this.elements.button || !this.elements.label) return;

        const saved = localStorage.getItem('paperreader_theme');
        const fallback = this.themes.find(item => item.id === saved) ? saved : 'neon';
        this.applyTheme(fallback);

        this.elements.button.addEventListener('click', () => this.cycleTheme());
    },

    cycleTheme() {
        const index = this.themes.findIndex(item => item.id === this.currentTheme);
        const nextIndex = (index + 1) % this.themes.length;
        this.applyTheme(this.themes[nextIndex].id);
    },

    applyTheme(themeId) {
        const theme = this.themes.find(item => item.id === themeId) || this.themes[0];
        this.currentTheme = theme.id;
        document.documentElement.dataset.theme = theme.id;
        localStorage.setItem('paperreader_theme', theme.id);
        if (this.elements.label) {
            this.elements.label.textContent = theme.label;
        }
    }
};

/**
 * Auth Controller - local account management
 */
const AuthController = {
    storageUsersKey: 'paperreader_auth_users',
    storageCurrentKey: 'paperreader_auth_current',
    users: [],
    currentUser: null,
    elements: {
        authBtn: null,
        authLabel: null,
        authSubtitle: null,
        modal: null,
        closeBtn: null,
        tabLogin: null,
        tabRegister: null,
        panelLogin: null,
        panelRegister: null,
        panelAccount: null,
        loginEmail: null,
        loginPassword: null,
        loginSubmit: null,
        registerName: null,
        registerEmail: null,
        registerPassword: null,
        registerSubmit: null,
        accountName: null,
        accountEmail: null,
        logoutBtn: null
    },

    init() {
        this.cacheElements();
        if (!this.elements.authBtn || !this.elements.modal) return;
        this.loadUsers();
        this.loadSession();
        this.bindEvents();
        this.updateHeader();
        this.updateAccountPanel();
    },

    cacheElements() {
        this.elements.authBtn = document.getElementById('btn-auth');
        this.elements.authLabel = document.getElementById('auth-label');
        this.elements.authSubtitle = document.getElementById('auth-subtitle');
        this.elements.modal = document.getElementById('auth-modal');
        this.elements.closeBtn = document.getElementById('auth-close');
        this.elements.tabLogin = document.getElementById('auth-tab-login');
        this.elements.tabRegister = document.getElementById('auth-tab-register');
        this.elements.panelLogin = document.getElementById('auth-panel-login');
        this.elements.panelRegister = document.getElementById('auth-panel-register');
        this.elements.panelAccount = document.getElementById('auth-panel-account');
        this.elements.loginEmail = document.getElementById('auth-login-email');
        this.elements.loginPassword = document.getElementById('auth-login-password');
        this.elements.loginSubmit = document.getElementById('auth-login-submit');
        this.elements.registerName = document.getElementById('auth-register-name');
        this.elements.registerEmail = document.getElementById('auth-register-email');
        this.elements.registerPassword = document.getElementById('auth-register-password');
        this.elements.registerSubmit = document.getElementById('auth-register-submit');
        this.elements.accountName = document.getElementById('auth-account-name');
        this.elements.accountEmail = document.getElementById('auth-account-email');
        this.elements.logoutBtn = document.getElementById('auth-logout');
    },

    bindEvents() {
        this.elements.authBtn?.addEventListener('click', () => this.open());
        this.elements.closeBtn?.addEventListener('click', () => this.close());
        this.elements.modal?.querySelector('.overlay-backdrop')?.addEventListener('click', () => this.close());

        this.elements.tabLogin?.addEventListener('click', () => this.showPanel('login'));
        this.elements.tabRegister?.addEventListener('click', () => this.showPanel('register'));

        this.elements.loginSubmit?.addEventListener('click', (e) => {
            e.preventDefault();
            this.login();
        });
        this.elements.registerSubmit?.addEventListener('click', (e) => {
            e.preventDefault();
            this.register();
        });
        this.elements.logoutBtn?.addEventListener('click', () => this.logout());
    },

    loadUsers() {
        try {
            const saved = localStorage.getItem(this.storageUsersKey);
            this.users = saved ? JSON.parse(saved) : [];
        } catch (e) {
            this.users = [];
        }
    },

    saveUsers() {
        try {
            localStorage.setItem(this.storageUsersKey, JSON.stringify(this.users));
        } catch (e) {
            console.warn('Failed to save users');
        }
    },

    loadSession() {
        const currentId = localStorage.getItem(this.storageCurrentKey);
        if (!currentId) {
            this.currentUser = null;
            return;
        }
        this.currentUser = this.users.find(user => user.id === currentId) || null;
    },

    setCurrentUser(user) {
        this.currentUser = user || null;
        if (user) {
            localStorage.setItem(this.storageCurrentKey, user.id);
            localStorage.setItem('paperreader-user-name', user.name);
            if (user.avatarSeed !== undefined) {
                localStorage.setItem('paperreader-user-avatar', user.avatarSeed.toString());
            }
        } else {
            localStorage.removeItem(this.storageCurrentKey);
            localStorage.removeItem('paperreader-user-name');
            localStorage.removeItem('paperreader-user-avatar');
        }
        this.updateHeader();
        this.updateAccountPanel();
        this.refreshPresenceAndChat();
    },

    updateHeader() {
        if (!this.elements.authLabel || !this.elements.authSubtitle || !this.elements.authBtn) return;
        if (this.currentUser) {
            this.elements.authLabel.textContent = this.truncateLabel(this.currentUser.name, 14).toUpperCase();
            this.elements.authSubtitle.textContent = 'SIGNED IN';
            this.elements.authBtn.classList.add('signed-in');
        } else {
            this.elements.authLabel.textContent = 'SIGN IN';
            this.elements.authSubtitle.textContent = 'GUEST';
            this.elements.authBtn.classList.remove('signed-in');
        }
    },

    updateAccountPanel() {
        if (!this.elements.accountName || !this.elements.accountEmail) return;
        if (this.currentUser) {
            this.elements.accountName.textContent = this.currentUser.name;
            this.elements.accountEmail.textContent = this.currentUser.email;
        } else {
            this.elements.accountName.textContent = 'Guest';
            this.elements.accountEmail.textContent = 'Not signed in';
        }
    },

    open() {
        if (!this.elements.modal) return;
        this.elements.modal.style.display = 'flex';
        setTimeout(() => this.elements.modal.classList.add('active'), 10);
        if (this.currentUser) {
            this.showPanel('account');
        } else {
            this.showPanel('login');
            if (this.elements.loginPassword) this.elements.loginPassword.value = '';
        }
    },

    close() {
        if (!this.elements.modal) return;
        this.elements.modal.classList.remove('active');
        setTimeout(() => {
            this.elements.modal.style.display = 'none';
        }, 200);
    },

    toggleTabs(visible) {
        const display = visible ? '' : 'none';
        if (this.elements.tabLogin) this.elements.tabLogin.style.display = display;
        if (this.elements.tabRegister) this.elements.tabRegister.style.display = display;
    },

    showPanel(panel) {
        this.elements.panelLogin?.classList.toggle('active', panel === 'login');
        this.elements.panelRegister?.classList.toggle('active', panel === 'register');
        this.elements.panelAccount?.classList.toggle('active', panel === 'account');
        this.elements.tabLogin?.classList.toggle('active', panel === 'login');
        this.elements.tabRegister?.classList.toggle('active', panel === 'register');
        this.toggleTabs(panel !== 'account');
    },

    login() {
        const email = (this.elements.loginEmail?.value || '').trim().toLowerCase();
        const password = this.elements.loginPassword?.value || '';
        if (!email || !password) {
            showToast('Enter your email and password');
            return;
        }
        const user = this.users.find(u => u.email.toLowerCase() === email && u.password === password);
        if (!user) {
            showToast('Invalid credentials');
            return;
        }
        this.setCurrentUser(user);
        showToast(`Welcome back, ${user.name}`);
        this.close();
    },

    register() {
        const name = (this.elements.registerName?.value || '').trim();
        const email = (this.elements.registerEmail?.value || '').trim().toLowerCase();
        const password = this.elements.registerPassword?.value || '';
        if (!name || !email || !password) {
            showToast('Complete all fields');
            return;
        }
        if (this.users.find(u => u.email.toLowerCase() === email)) {
            showToast('Email already registered');
            return;
        }
        const user = {
            id: `user_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`,
            name,
            email,
            password,
            avatarSeed: Math.floor(Math.random() * 1000)
        };
        this.users.push(user);
        this.saveUsers();
        this.setCurrentUser(user);
        showToast(`Account created for ${name}`);
        this.close();
    },

    logout() {
        this.setCurrentUser(null);
        showToast('Signed out');
        this.showPanel('login');
    },

    getPresenceProfile() {
        if (!this.currentUser) return null;
        const seed = this.currentUser.avatarSeed ?? this.currentUser.id;
        return {
            id: this.currentUser.id,
            name: this.currentUser.name,
            avatar: `https://api.dicebear.com/7.x/avataaars/svg?seed=${seed}&backgroundColor=b6e3f4,c0aede,d1d4f9`
        };
    },

    refreshPresenceAndChat() {
        if (typeof PresenceController !== 'undefined' && PresenceController.refreshUserProfile) {
            PresenceController.refreshUserProfile();
        }
        if (typeof TeamChatController !== 'undefined' && TeamChatController.refreshUser) {
            TeamChatController.refreshUser();
        }
    },

    truncateLabel(text, maxLen) {
        if (!text) return '';
        if (text.length <= maxLen) return text;
        return `${text.slice(0, maxLen - 3)}...`;
    }
};

/**
 * Project Manager - multi-project support (local + server)
 */
const ProjectManager = {
    storageKey: 'paperreader_projects',
    activeKey: 'paperreader_active_project',
    projects: [],
    activeProject: null,
    onProjectChange: null,
    elements: {
        btnProjects: null,
        modal: null,
        closeBtn: null,
        list: null,
        nameInput: null,
        typeSelect: null,
        createBtn: null,
        activeName: null
    },

    init() {
        this.cacheElements();
        this.loadProjects();
        this.bindEvents();
        this.renderProjects();
        this.updateHeader();
    },

    cacheElements() {
        this.elements.btnProjects = document.getElementById('btn-projects');
        this.elements.modal = document.getElementById('project-modal');
        this.elements.closeBtn = document.getElementById('project-close');
        this.elements.list = document.getElementById('project-list');
        this.elements.nameInput = document.getElementById('project-name-input');
        this.elements.typeSelect = document.getElementById('project-type-select');
        this.elements.createBtn = document.getElementById('project-create');
        this.elements.activeName = document.getElementById('active-project-name');
    },

    bindEvents() {
        this.elements.btnProjects?.addEventListener('click', () => this.openModal());
        this.elements.closeBtn?.addEventListener('click', () => this.closeModal());
        this.elements.modal?.querySelector('.overlay-backdrop')?.addEventListener('click', () => this.closeModal());
        this.elements.createBtn?.addEventListener('click', () => this.createFromForm());
        this.elements.nameInput?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this.createFromForm();
            }
        });
        this.elements.list?.addEventListener('click', (e) => this.handleListAction(e));
    },

    loadProjects() {
        try {
            const saved = localStorage.getItem(this.storageKey);
            this.projects = saved ? JSON.parse(saved) : [];
        } catch (e) {
            this.projects = [];
        }

        if (!Array.isArray(this.projects)) {
            this.projects = [];
        }

        if (!this.projects.find(project => project.id === 'server')) {
            this.projects.unshift(this.createRemoteProject('Server Workspace', 'server'));
        }

        const urlProject = new URLSearchParams(window.location.search).get('project');
        if (urlProject && !this.projects.find(project => project.id === urlProject)) {
            this.projects.push(this.createRemoteProject(`Server Project ${urlProject}`, urlProject));
        }

        const activeId = urlProject || localStorage.getItem(this.activeKey) || this.projects[0]?.id;
        this.activeProject = this.projects.find(project => project.id === activeId) || this.projects[0] || null;
    },

    saveProjects() {
        try {
            localStorage.setItem(this.storageKey, JSON.stringify(this.projects));
        } catch (e) {
            console.warn('Failed to save projects');
        }
    },

    applyActiveProject() {
        if (this.activeProject?.id) {
            localStorage.setItem(this.activeKey, this.activeProject.id);
        }
        this.updateHeader();
        this.renderProjects();
        this.notifyChange();
    },

    notifyChange() {
        if (typeof this.onProjectChange === 'function') {
            this.onProjectChange(this.activeProject);
        }
    },

    openModal() {
        if (!this.elements.modal) return;
        this.renderProjects();
        this.elements.modal.style.display = 'flex';
        setTimeout(() => this.elements.modal.classList.add('active'), 10);
    },

    closeModal() {
        if (!this.elements.modal) return;
        this.elements.modal.classList.remove('active');
        setTimeout(() => {
            this.elements.modal.style.display = 'none';
        }, 200);
    },

    updateHeader() {
        if (!this.elements.activeName) return;
        const name = this.activeProject?.name || 'SERVER';
        this.elements.activeName.textContent = this.truncateLabel(name.toUpperCase(), 16);
    },

    setActiveProject(id, options = {}) {
        const project = this.projects.find(item => item.id === id);
        if (!project) return;
        this.activeProject = project;
        localStorage.setItem(this.activeKey, project.id);
        this.touchProject(project.id);
        this.updateHeader();
        this.renderProjects();
        if (!options.silent) {
            this.closeModal();
        }
        this.notifyChange();
    },

    touchProject(id) {
        const project = this.projects.find(item => item.id === id);
        if (project) {
            project.updatedAt = Date.now();
            this.saveProjects();
        }
    },

    createFromForm() {
        const name = this.sanitizeName(this.elements.nameInput?.value || '');
        if (!name) {
            showToast('Project name required');
            return;
        }
        if (this.projects.some(project => project.name.toLowerCase() === name.toLowerCase())) {
            showToast('Project name already exists');
            return;
        }
        const type = this.elements.typeSelect?.value === 'remote' ? 'remote' : 'local';
        const id = this.generateProjectId(name, type);
        const project = {
            id,
            name,
            type,
            updatedAt: Date.now(),
            files: type === 'local' ? this.createTemplateFiles(name) : []
        };
        this.projects.unshift(project);
        this.saveProjects();
        if (this.elements.nameInput) this.elements.nameInput.value = '';
        this.setActiveProject(id);
    },

    handleListAction(event) {
        const btn = event.target.closest('button');
        const item = event.target.closest('.project-item');
        if (!btn || !item) return;
        const id = item.dataset.id;
        const action = btn.dataset.action;
        if (!id || !action) return;

        if (action === 'open') {
            this.setActiveProject(id);
            return;
        }
        if (action === 'rename') {
            this.renameProject(id);
            return;
        }
        if (action === 'duplicate') {
            this.duplicateProject(id);
            return;
        }
        if (action === 'delete') {
            this.deleteProject(id);
        }
    },

    renameProject(id) {
        const project = this.projects.find(item => item.id === id);
        if (!project) return;
        const name = prompt('Rename project:', project.name);
        const clean = this.sanitizeName(name || '');
        if (!clean) return;
        project.name = clean;
        project.updatedAt = Date.now();
        this.saveProjects();
        this.updateHeader();
        this.renderProjects();
    },

    duplicateProject(id) {
        const project = this.projects.find(item => item.id === id);
        if (!project) return;
        if (project.type !== 'local') {
            showToast('Duplicate is only available for local projects');
            return;
        }
        const clone = {
            id: this.generateProjectId(`${project.name} Copy`, 'local'),
            name: `${project.name} Copy`,
            type: 'local',
            updatedAt: Date.now(),
            files: (project.files || []).map(file => ({ ...file }))
        };
        this.projects.unshift(clone);
        this.saveProjects();
        this.renderProjects();
        showToast('Project duplicated');
    },

    deleteProject(id) {
        const project = this.projects.find(item => item.id === id);
        if (!project) return;
        if (project.id === 'server') {
            showToast('Server workspace cannot be deleted');
            return;
        }
        if (!confirm(`Delete project "${project.name}"?`)) return;
        this.projects = this.projects.filter(item => item.id !== id);
        if (this.activeProject?.id === id) {
            this.activeProject = this.projects[0] || null;
            if (this.activeProject) {
                localStorage.setItem(this.activeKey, this.activeProject.id);
            } else {
                localStorage.removeItem(this.activeKey);
            }
        }
        this.saveProjects();
        this.updateHeader();
        this.renderProjects();
        this.notifyChange();
    },

    renderProjects() {
        if (!this.elements.list) return;
        if (!this.projects.length) {
            this.elements.list.innerHTML = '<div class="outline-empty">No projects yet</div>';
            return;
        }
        this.elements.list.innerHTML = this.projects.map(project => {
            const isActive = this.activeProject?.id === project.id;
            const updated = project.updatedAt ? new Date(project.updatedAt).toLocaleString() : 'Never';
            const fileCount = project.type === 'local' ? `${project.files?.length || 0} files` : 'Server';
            const tagClass = project.type === 'local' ? 'project-tag local' : 'project-tag';
            const typeLabel = project.type === 'local' ? 'local' : 'server';
            const disableDelete = project.id === 'server';
            return `
                <div class="project-item ${isActive ? 'active' : ''}" data-id="${project.id}">
                    <div class="project-info">
                        <div class="project-name">${escapeHtml(project.name)}</div>
                        <div class="project-meta">
                            <span class="${tagClass}">${typeLabel}</span>
                            <span>${fileCount}</span>
                            <span>Updated ${updated}</span>
                        </div>
                    </div>
                    <div class="project-actions">
                        <button class="btn-tool btn-small" data-action="open">${isActive ? 'ACTIVE' : 'OPEN'}</button>
                        <button class="btn-tool btn-small" data-action="rename">RENAME</button>
                        <button class="btn-tool btn-small" data-action="duplicate">DUPLICATE</button>
                        <button class="btn-tool btn-small btn-danger" data-action="delete" ${disableDelete ? 'disabled' : ''}>DELETE</button>
                    </div>
                </div>
            `;
        }).join('');
    },

    createRemoteProject(name, id) {
        return {
            id,
            name,
            type: 'remote',
            updatedAt: Date.now(),
            files: []
        };
    },

    createTemplateFiles(projectName) {
        const safeName = projectName.replace(/[^a-zA-Z0-9 ]+/g, '').trim() || 'Rowan 308 Lab Workspace';
        const mainTex = `% ${safeName}\n\\documentclass{article}\n\\usepackage[margin=1in]{geometry}\n\\usepackage{graphicx}\n\\title{${safeName}}\n\\author{Your Name}\n\\date{\\today}\n\\begin{document}\n\\maketitle\n\\begin{abstract}\nWrite your abstract here.\n\\end{abstract}\n\\section{Introduction}\nStart writing your paper...\n\\end{document}\n`;
        const bib = `@article{sample2024,\n  title={Sample Reference},\n  author={Doe, Jane},\n  journal={Journal of Retro Research},\n  year={2024}\n}\n`;
        return [
            { name: 'main.tex', content: mainTex, updatedAt: Date.now() },
            { name: 'refs.bib', content: bib, updatedAt: Date.now() }
        ];
    },

    generateProjectId(name, type) {
        const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'project';
        return `${type}_${slug}_${Date.now().toString(36)}`;
    },

    sanitizeName(name) {
        const trimmed = (name || '').trim();
        if (!trimmed) return null;
        return trimmed.slice(0, 60);
    },

    truncateLabel(text, maxLen) {
        if (!text) return '';
        if (text.length <= maxLen) return text;
        return `${text.slice(0, maxLen - 3)}...`;
    },

    isLocalProject() {
        return this.activeProject?.type === 'local';
    },

    getLocalFilesList() {
        if (!this.isLocalProject()) return [];
        return (this.activeProject.files || []).map(file => ({
            name: file.name,
            type: this.inferFileType(file.name),
            size: (file.content || '').length
        }));
    },

    getLocalFileContent(filename) {
        if (!this.isLocalProject()) return null;
        const entry = (this.activeProject.files || []).find(file => file.name === filename);
        return entry ? entry.content : null;
    },

    saveLocalFile(filename, content) {
        if (!this.isLocalProject()) return false;
        const files = this.activeProject.files || [];
        const existing = files.find(file => file.name === filename);
        if (existing) {
            existing.content = content;
            existing.updatedAt = Date.now();
        } else {
            files.push({ name: filename, content, updatedAt: Date.now() });
        }
        this.activeProject.files = files;
        this.touchProject(this.activeProject.id);
        return true;
    },

    deleteLocalFile(filename) {
        if (!this.isLocalProject()) return false;
        this.activeProject.files = (this.activeProject.files || []).filter(file => file.name !== filename);
        this.touchProject(this.activeProject.id);
        return true;
    },

    renameLocalFile(oldName, newName) {
        if (!this.isLocalProject()) return false;
        const files = this.activeProject.files || [];
        const entry = files.find(file => file.name === oldName);
        if (!entry) return false;
        entry.name = newName;
        entry.updatedAt = Date.now();
        this.touchProject(this.activeProject.id);
        return true;
    },

    inferFileType(filename) {
        if (filename.endsWith('.tex')) return 'tex';
        if (filename.endsWith('.bib')) return 'bib';
        return 'file';
    }
};

/**
 * Real-time Presence Controller
 */
const PresenceController = {
    ws: null,
    user: null,
    activeUsers: [],

    init() {
        this.user = this.generateUser();
        this.connect();
    },

    generateUser() {
        const authProfile = AuthController.getPresenceProfile();
        if (authProfile) return authProfile;

        // Generate a fun random name if not in local storage
        let name = localStorage.getItem('paperreader-user-name');
        if (!name) {
            const adjectives = ['Electric', 'Pixel', 'Neon', 'Retro', 'Cyber', 'Vintage', 'Turbo', 'Digital', 'Arcade', 'Vector'];
            const nouns = ['User', 'Editor', 'Hacker', 'Researcher', 'Scholar', 'Writer', 'Reader', 'Pilot', 'Wizard', 'Nova'];
            name = `${adjectives[Math.floor(Math.random() * adjectives.length)]} ${nouns[Math.floor(Math.random() * nouns.length)]} #${Math.floor(Math.random() * 900 + 100)}`;
            localStorage.setItem('paperreader-user-name', name);
        }

        let avatarSeed = localStorage.getItem('paperreader-user-avatar');
        if (!avatarSeed) {
            avatarSeed = Math.floor(Math.random() * 1000);
            localStorage.setItem('paperreader-user-avatar', avatarSeed);
        }

        return {
            name: name,
            avatar: `https://api.dicebear.com/7.x/avataaars/svg?seed=${avatarSeed}&backgroundColor=b6e3f4,c0aede,d1d4f9`
        };
    },

    refreshUserProfile() {
        this.user = this.generateUser();
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            try {
                this.ws.send(JSON.stringify({ type: 'join', user: this.user }));
            } catch (e) {
                console.warn('Failed to refresh presence user');
            }
        } else {
            try {
                this.ws?.close();
            } catch (e) {
                console.warn('Failed to close websocket');
            }
            this.connect();
        }
    },

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        let wsUrl;

        // Use relative path for better proxy compatibility
        const host = window.location.host;
        const path = window.location.pathname.replace(/\/$/, ''); // Remove trailing slash

        if (isLocalDev) {
            wsUrl = `${protocol}//127.0.0.1:22222/ws/presence`;
        } else {
            // Path-aware websocket URL
            if (path.includes('/paperreader')) {
                // If we are at /paperreader/ or /paperreader, use /paperreader/ws/presence
                wsUrl = `${protocol}//${host}/paperreader/ws/presence`;
            } else {
                wsUrl = `${protocol}//${host}/ws/presence`;
            }
        }

        console.log('Connecting to presence websocket...', wsUrl);

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                console.log('Presence websocket connected');
                this.ws.send(JSON.stringify({
                    type: 'join',
                    user: this.user
                }));
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'presence') {
                        this.updatePresence(data.users, data.count);
                    }
                } catch (e) {
                    console.error('Error parsing presence message', e);
                }
            };

            this.ws.onclose = (event) => {
                console.log('Presence websocket closed', event.code, event.reason);
                // Don't reconnect if it was a normal closure and we have a clean exit
                if (event.code !== 1000) {
                    setTimeout(() => this.connect(), 5000);
                }
            };

            this.ws.onerror = (error) => {
                console.error('Presence websocket error', error);
                // Reconnect will be handled by onclose
            };
        } catch (error) {
            console.error('Failed to create WebSocket', error);
            setTimeout(() => this.connect(), 5000);
        }
    },

    updatePresence(users, count) {
        this.activeUsers = users;
        const listContainer = document.getElementById('presence-list');
        const userCountSpan = document.getElementById('user-count');

        if (userCountSpan) userCountSpan.textContent = count;

        if (listContainer) {
            // Re-render the user avatars
            listContainer.innerHTML = users.map(user => `
                <div class="user-avatar" title="${escapeHtml(user.name)}">
                    <img src="${user.avatar}" alt="${escapeHtml(user.name)}" onerror="this.src='https://ui-avatars.com/api/?name=${encodeURIComponent(user.name)}&background=random'">
                    <div class="user-tooltip">${escapeHtml(user.name)}</div>
                </div>
            `).join('');
        }

        if (typeof TeamChatController !== 'undefined' && TeamChatController.updateUsersFromPresence) {
            TeamChatController.updateUsersFromPresence(users);
        }
    }
};

// Initialize
document.addEventListener('DOMContentLoaded', init);

/**
 * ==========================================
 * EDITOR & OVERLEAF LOGIC
 * ==========================================
 */

const EditorController = {
    state: {
        files: [],
        currentFile: null,
        citationMap: {},
        citationKeys: [],
        pdfDoc: null,
        lastCompiledPdfUrl: null,
        pdfScale: 1.2,
        editHistory: [],
        findMatches: [],
        currentFindIndex: -1,
        fileFilter: '',
        fileContents: {},
        dirtyFiles: {},
        autoSaveEnabled: false,
        autoCompileEnabled: false,
        autoSaveTimer: null,
        lastAutoSaveAt: 0,
        outlineItems: [],
        outlineTimer: null,
        comments: [],
        commentSelection: null,
        versionHistory: [],
        versionThrottle: {},
        selectedVersionId: null,
        historyFileSelected: null,
        lastCompileLog: '',
        lastCompileSummary: { errors: 0, warnings: 0 },
        lastCompileAt: 0,
        currentPreviewPage: 1,
        pdfScrollHandler: null,
        editorMode: 'code',
        wrapEnabled: true,
        visualSyncTimer: null,
        visualSyncLock: false,
        highlightTimer: null,
        autocomplete: {
            visible: false,
            mode: null,
            items: [],
            activeIndex: -1,
            context: null
        },
        autocompleteMirror: null
    },

    elements: {
        tabLibrary: document.getElementById('tab-library'),
        tabEditor: document.getElementById('tab-editor'),
        libraryView: document.getElementById('library-view'),
        editorView: document.getElementById('editor-view'),
        fileList: document.getElementById('file-list'),
        currentFileName: document.getElementById('current-file-name'),
        btnCompile: document.getElementById('btn-compile'),
        btnSave: document.getElementById('btn-save'),
        btnUpload: document.getElementById('btn-upload'),
        fileUploadInput: document.getElementById('file-upload-input'),
        btnDelete: document.getElementById('btn-delete'),
        codeEditor: document.getElementById('code-editor'),
        visualEditor: document.getElementById('visual-editor'),
        visualToolbar: document.getElementById('visual-toolbar'),
        btnModeCode: document.getElementById('btn-mode-code'),
        btnModeVisual: document.getElementById('btn-mode-visual'),
        btnLineWrap: document.getElementById('btn-line-wrap'),
        codeHighlight: document.getElementById('code-highlight'),
        codeHighlightContent: document.getElementById('code-highlight-content'),
        latexAutocomplete: document.getElementById('latex-autocomplete'),
        latexAutocompleteHeader: document.getElementById('latex-autocomplete-header'),
        latexAutocompleteList: document.getElementById('latex-autocomplete-list'),
        compileStatus: document.getElementById('compile-status'),
        pdfContainer: document.getElementById('pdf-viewer-container'),
        btnDownloadCompiled: document.getElementById('btn-download-compiled'),
        // New elements
        btnFindTex: document.getElementById('btn-find-tex'),
        btnFindPdf: document.getElementById('btn-find-pdf'),
        btnHistory: document.getElementById('btn-history'),
        btnAutoSave: document.getElementById('btn-auto-save'),
        btnAutoCompile: document.getElementById('btn-auto-compile'),
        btnShare: document.getElementById('btn-share'),
        editorStats: document.getElementById('editor-stats'),
        btnZoomIn: document.getElementById('btn-zoom-in'),
        btnZoomOut: document.getElementById('btn-zoom-out'),
        btnSyncPdf: document.getElementById('btn-sync-pdf'),
        btnLogs: document.getElementById('btn-logs'),
        findBar: document.getElementById('find-bar'),
        findInput: document.getElementById('find-input'),
        findCount: document.getElementById('find-count'),
        findPrev: document.getElementById('find-prev'),
        findNext: document.getElementById('find-next'),
        findClose: document.getElementById('find-close'),
        pdfFindBar: document.getElementById('pdf-find-bar'),
        pdfFindInput: document.getElementById('pdf-find-input'),
        pdfFindCount: document.getElementById('pdf-find-count'),
        pdfFindPrev: document.getElementById('pdf-find-prev'),
        pdfFindNext: document.getElementById('pdf-find-next'),
        pdfFindClose: document.getElementById('pdf-find-close'),
        historyModal: document.getElementById('history-modal'),
        historyList: document.getElementById('history-list'),
        historyClose: document.getElementById('history-close'),
        historyTabActivity: document.getElementById('history-tab-activity'),
        historyTabVersions: document.getElementById('history-tab-versions'),
        historyPanelActivity: document.getElementById('history-panel-activity'),
        historyPanelVersions: document.getElementById('history-panel-versions'),
        historyFileSelect: document.getElementById('history-file-select'),
        historyRefresh: document.getElementById('history-refresh'),
        historyVersionList: document.getElementById('history-version-list'),
        historyPreviewTitle: document.getElementById('history-preview-title'),
        historyPreviewContent: document.getElementById('history-preview-content'),
        historyRestoreVersion: document.getElementById('history-restore-version'),
        historyCopyVersion: document.getElementById('history-copy-version'),
        // Line numbers and resizer
        lineNumbers: document.getElementById('line-numbers'),
        btnLineNumbers: document.getElementById('btn-line-numbers'),
        sourceView: document.getElementById('source-view'),
        previewView: document.getElementById('preview-view'),
        editorSidebar: document.querySelector('.editor-sidebar'),
        btnToggleSidebar: document.getElementById('btn-toggle-sidebar'),
        panelResizer: document.getElementById('panel-resizer'),
        btnToggleCitationHighlights: document.getElementById('btn-toggle-citation-highlights'),
        pageCountDisplay: document.getElementById('page-count-display'),
        compileLogPanel: document.getElementById('compile-log-panel'),
        compileLogFilter: document.getElementById('compile-log-filter'),
        compileLogContent: document.getElementById('compile-log-content'),
        compileLogCopy: document.getElementById('compile-log-copy'),
        compileLogClose: document.getElementById('compile-log-close'),
        compileLogMeta: document.getElementById('compile-log-meta'),
        sidebarTabFiles: document.getElementById('sidebar-tab-files'),
        sidebarTabOutline: document.getElementById('sidebar-tab-outline'),
        sidebarTabComments: document.getElementById('sidebar-tab-comments'),
        sidebarPanelFiles: document.getElementById('sidebar-panel-files'),
        sidebarPanelOutline: document.getElementById('sidebar-panel-outline'),
        sidebarPanelComments: document.getElementById('sidebar-panel-comments'),
        sidebarBackdrop: null,
        btnNewFile: document.getElementById('btn-new-file'),
        btnRenameFile: document.getElementById('btn-rename-file'),
        btnDuplicateFile: document.getElementById('btn-duplicate-file'),
        fileSearchInput: document.getElementById('file-search-input'),
        outlineList: document.getElementById('outline-list'),
        commentContext: document.getElementById('comment-context'),
        commentInput: document.getElementById('comment-input'),
        btnAddComment: document.getElementById('btn-add-comment'),
        commentList: document.getElementById('comment-list'),
        // Mobile tabs
        btnShowSource: document.getElementById('btn-show-source'),
        btnShowPreview: document.getElementById('btn-show-preview'),
        editorContentArea: document.querySelector('.editor-content-area')
    },

    latexCommandSuggestions: [
        { label: '\\cite{}', insertText: '\\cite{}', cursorOffset: -1, meta: 'cite' },
        { label: '\\citet{}', insertText: '\\citet{}', cursorOffset: -1, meta: 'cite' },
        { label: '\\citep{}', insertText: '\\citep{}', cursorOffset: -1, meta: 'cite' },
        { label: '\\citeauthor{}', insertText: '\\citeauthor{}', cursorOffset: -1, meta: 'cite' },
        { label: '\\citeyear{}', insertText: '\\citeyear{}', cursorOffset: -1, meta: 'cite' },
        { label: '\\citeyearpar{}', insertText: '\\citeyearpar{}', cursorOffset: -1, meta: 'cite' },
        { label: '\\citealt{}', insertText: '\\citealt{}', cursorOffset: -1, meta: 'cite' },
        { label: '\\citealp{}', insertText: '\\citealp{}', cursorOffset: -1, meta: 'cite' },
        { label: '\\citenum{}', insertText: '\\citenum{}', cursorOffset: -1, meta: 'cite' },
        { label: '\\citeonline{}', insertText: '\\citeonline{}', cursorOffset: -1, meta: 'cite' },
        { label: '\\nocite{}', insertText: '\\nocite{}', cursorOffset: -1, meta: 'cite' },
        { label: '\\bibitem{}', insertText: '\\bibitem{}', cursorOffset: -1, meta: 'cite' },
        { label: '\\bibliographystyle{}', insertText: '\\bibliographystyle{}', cursorOffset: -1, meta: 'cite' },
        { label: '\\bibliography{}', insertText: '\\bibliography{}', cursorOffset: -1, meta: 'cite' },
        { label: '\\usepackage{}', insertText: '\\usepackage{}', cursorOffset: -1, meta: 'pkg' },
        { label: '\\usepackage[]{}', insertText: '\\usepackage[]{}', cursorOffset: -2, meta: 'pkg' },
        { label: '\\documentclass{}', insertText: '\\documentclass{}', cursorOffset: -1, meta: 'pkg' },
        { label: '\\documentclass[]{}', insertText: '\\documentclass[]{}', cursorOffset: -2, meta: 'pkg' },
        { label: '\\begin{}', insertText: '\\begin{}', cursorOffset: -1, meta: 'env' },
        { label: '\\end{}', insertText: '\\end{}', cursorOffset: -1, meta: 'env' },
        { label: '\\begin{itemize}', insertText: '\\begin{itemize}\n\\item \n\\end{itemize}', cursorOffset: -12, meta: 'env' },
        { label: '\\begin{enumerate}', insertText: '\\begin{enumerate}\n\\item \n\\end{enumerate}', cursorOffset: -13, meta: 'env' },
        { label: '\\begin{description}', insertText: '\\begin{description}\n\\item[] \n\\end{description}', cursorOffset: -15, meta: 'env' },
        { label: '\\begin{figure}', insertText: '\\begin{figure}\n\\centering\n\\includegraphics{}\n\\caption{}\n\\label{}\n\\end{figure}', cursorOffset: -47, meta: 'env' },
        { label: '\\begin{table}', insertText: '\\begin{table}\n\\centering\n\\caption{}\n\\label{}\n\\begin{tabular}{}\n\\end{tabular}\n\\end{table}', cursorOffset: -47, meta: 'env' },
        { label: '\\begin{tabular}{}', insertText: '\\begin{tabular}{}\n\n\\end{tabular}', cursorOffset: -15, meta: 'env' },
        { label: '\\begin{align}', insertText: '\\begin{align}\n\n\\end{align}', cursorOffset: -12, meta: 'env' },
        { label: '\\begin{equation}', insertText: '\\begin{equation}\n\n\\end{equation}', cursorOffset: -14, meta: 'env' },
        { label: '\\begin{theorem}', insertText: '\\begin{theorem}\n\n\\end{theorem}', cursorOffset: -14, meta: 'env' },
        { label: '\\begin{lemma}', insertText: '\\begin{lemma}\n\n\\end{lemma}', cursorOffset: -12, meta: 'env' },
        { label: '\\begin{proof}', insertText: '\\begin{proof}\n\n\\end{proof}', cursorOffset: -12, meta: 'env' },
        { label: '\\section{}', insertText: '\\section{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\subsection{}', insertText: '\\subsection{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\subsubsection{}', insertText: '\\subsubsection{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\paragraph{}', insertText: '\\paragraph{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\subparagraph{}', insertText: '\\subparagraph{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\chapter{}', insertText: '\\chapter{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\title{}', insertText: '\\title{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\author{}', insertText: '\\author{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\date{}', insertText: '\\date{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\maketitle', insertText: '\\maketitle', cursorOffset: 0, meta: 'cmd' },
        { label: '\\tableofcontents', insertText: '\\tableofcontents', cursorOffset: 0, meta: 'cmd' },
        { label: '\\listoffigures', insertText: '\\listoffigures', cursorOffset: 0, meta: 'cmd' },
        { label: '\\listoftables', insertText: '\\listoftables', cursorOffset: 0, meta: 'cmd' },
        { label: '\\item', insertText: '\\item ', cursorOffset: 0, meta: 'cmd' },
        { label: '\\label{}', insertText: '\\label{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\ref{}', insertText: '\\ref{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\eqref{}', insertText: '\\eqref{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\pageref{}', insertText: '\\pageref{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\url{}', insertText: '\\url{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\href{}{}', insertText: '\\href{}{}', cursorOffset: -3, meta: 'cmd' },
        { label: '\\footnote{}', insertText: '\\footnote{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\textbf{}', insertText: '\\textbf{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\textit{}', insertText: '\\textit{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\texttt{}', insertText: '\\texttt{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\emph{}', insertText: '\\emph{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\underline{}', insertText: '\\underline{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\textcolor{}{}', insertText: '\\textcolor{}{}', cursorOffset: -3, meta: 'cmd' },
        { label: '\\color{}', insertText: '\\color{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\includegraphics{}', insertText: '\\includegraphics{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\includegraphics[]{}', insertText: '\\includegraphics[]{}', cursorOffset: -2, meta: 'cmd' },
        { label: '\\caption{}', insertText: '\\caption{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\centering', insertText: '\\centering', cursorOffset: 0, meta: 'cmd' },
        { label: '\\raggedright', insertText: '\\raggedright', cursorOffset: 0, meta: 'cmd' },
        { label: '\\raggedleft', insertText: '\\raggedleft', cursorOffset: 0, meta: 'cmd' },
        { label: '\\itemsep', insertText: '\\itemsep', cursorOffset: 0, meta: 'cmd' },
        { label: '\\setlength{}{}', insertText: '\\setlength{}{}', cursorOffset: -3, meta: 'cmd' },
        { label: '\\newcommand{}{}', insertText: '\\newcommand{}{}', cursorOffset: -3, meta: 'cmd' },
        { label: '\\renewcommand{}{}', insertText: '\\renewcommand{}{}', cursorOffset: -3, meta: 'cmd' },
        { label: '\\input{}', insertText: '\\input{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\include{}', insertText: '\\include{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\clearpage', insertText: '\\clearpage', cursorOffset: 0, meta: 'cmd' },
        { label: '\\newpage', insertText: '\\newpage', cursorOffset: 0, meta: 'cmd' },
        { label: '\\pagebreak', insertText: '\\pagebreak', cursorOffset: 0, meta: 'cmd' },
        { label: '\\noindent', insertText: '\\noindent', cursorOffset: 0, meta: 'cmd' },
        { label: '\\hfill', insertText: '\\hfill', cursorOffset: 0, meta: 'cmd' },
        { label: '\\vfill', insertText: '\\vfill', cursorOffset: 0, meta: 'cmd' },
        { label: '\\mbox{}', insertText: '\\mbox{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\phantom{}', insertText: '\\phantom{}', cursorOffset: -1, meta: 'cmd' },
        { label: '\\mathrm{}', insertText: '\\mathrm{}', cursorOffset: -1, meta: 'math' },
        { label: '\\mathit{}', insertText: '\\mathit{}', cursorOffset: -1, meta: 'math' },
        { label: '\\mathbf{}', insertText: '\\mathbf{}', cursorOffset: -1, meta: 'math' },
        { label: '\\mathcal{}', insertText: '\\mathcal{}', cursorOffset: -1, meta: 'math' },
        { label: '\\mathbb{}', insertText: '\\mathbb{}', cursorOffset: -1, meta: 'math' },
        { label: '\\frac{}{}', insertText: '\\frac{}{}', cursorOffset: -3, meta: 'math' },
        { label: '\\sqrt{}', insertText: '\\sqrt{}', cursorOffset: -1, meta: 'math' },
        { label: '\\sum', insertText: '\\sum', cursorOffset: 0, meta: 'math' },
        { label: '\\prod', insertText: '\\prod', cursorOffset: 0, meta: 'math' },
        { label: '\\int', insertText: '\\int', cursorOffset: 0, meta: 'math' },
        { label: '\\lim', insertText: '\\lim', cursorOffset: 0, meta: 'math' },
        { label: '\\log', insertText: '\\log', cursorOffset: 0, meta: 'math' },
        { label: '\\ln', insertText: '\\ln', cursorOffset: 0, meta: 'math' },
        { label: '\\sin', insertText: '\\sin', cursorOffset: 0, meta: 'math' },
        { label: '\\cos', insertText: '\\cos', cursorOffset: 0, meta: 'math' },
        { label: '\\tan', insertText: '\\tan', cursorOffset: 0, meta: 'math' },
        { label: '\\left(', insertText: '\\left(', cursorOffset: 0, meta: 'math' },
        { label: '\\right)', insertText: '\\right)', cursorOffset: 0, meta: 'math' },
        { label: '\\left[', insertText: '\\left[', cursorOffset: 0, meta: 'math' },
        { label: '\\right]', insertText: '\\right]', cursorOffset: 0, meta: 'math' },
        { label: '\\left\\{', insertText: '\\left\\{', cursorOffset: 0, meta: 'math' },
        { label: '\\right\\}', insertText: '\\right\\}', cursorOffset: 0, meta: 'math' },
        { label: '\\leq', insertText: '\\leq', cursorOffset: 0, meta: 'math' },
        { label: '\\geq', insertText: '\\geq', cursorOffset: 0, meta: 'math' },
        { label: '\\neq', insertText: '\\neq', cursorOffset: 0, meta: 'math' },
        { label: '\\approx', insertText: '\\approx', cursorOffset: 0, meta: 'math' },
        { label: '\\times', insertText: '\\times', cursorOffset: 0, meta: 'math' },
        { label: '\\cdot', insertText: '\\cdot', cursorOffset: 0, meta: 'math' },
        { label: '\\pm', insertText: '\\pm', cursorOffset: 0, meta: 'math' },
        { label: '\\to', insertText: '\\to', cursorOffset: 0, meta: 'math' },
        { label: '\\rightarrow', insertText: '\\rightarrow', cursorOffset: 0, meta: 'math' },
        { label: '\\Rightarrow', insertText: '\\Rightarrow', cursorOffset: 0, meta: 'math' },
        { label: '\\in', insertText: '\\in', cursorOffset: 0, meta: 'math' },
        { label: '\\subset', insertText: '\\subset', cursorOffset: 0, meta: 'math' },
        { label: '\\subseteq', insertText: '\\subseteq', cursorOffset: 0, meta: 'math' },
        { label: '\\forall', insertText: '\\forall', cursorOffset: 0, meta: 'math' },
        { label: '\\exists', insertText: '\\exists', cursorOffset: 0, meta: 'math' },
        { label: '\\nabla', insertText: '\\nabla', cursorOffset: 0, meta: 'math' },
        { label: '\\partial', insertText: '\\partial', cursorOffset: 0, meta: 'math' },
        { label: '\\alpha', insertText: '\\alpha', cursorOffset: 0, meta: 'math' },
        { label: '\\beta', insertText: '\\beta', cursorOffset: 0, meta: 'math' },
        { label: '\\gamma', insertText: '\\gamma', cursorOffset: 0, meta: 'math' },
        { label: '\\delta', insertText: '\\delta', cursorOffset: 0, meta: 'math' },
        { label: '\\epsilon', insertText: '\\epsilon', cursorOffset: 0, meta: 'math' },
        { label: '\\theta', insertText: '\\theta', cursorOffset: 0, meta: 'math' },
        { label: '\\lambda', insertText: '\\lambda', cursorOffset: 0, meta: 'math' },
        { label: '\\mu', insertText: '\\mu', cursorOffset: 0, meta: 'math' },
        { label: '\\pi', insertText: '\\pi', cursorOffset: 0, meta: 'math' },
        { label: '\\sigma', insertText: '\\sigma', cursorOffset: 0, meta: 'math' },
        { label: '\\phi', insertText: '\\phi', cursorOffset: 0, meta: 'math' },
        { label: '\\omega', insertText: '\\omega', cursorOffset: 0, meta: 'math' },
        { label: '\\leftarrow', insertText: '\\leftarrow', cursorOffset: 0, meta: 'math' },
        { label: '\\uparrow', insertText: '\\uparrow', cursorOffset: 0, meta: 'math' },
        { label: '\\downarrow', insertText: '\\downarrow', cursorOffset: 0, meta: 'math' },
        { label: '\\cdots', insertText: '\\cdots', cursorOffset: 0, meta: 'math' },
        { label: '\\ldots', insertText: '\\ldots', cursorOffset: 0, meta: 'math' },
        { label: '\\dots', insertText: '\\dots', cursorOffset: 0, meta: 'math' },
        { label: '\\hline', insertText: '\\hline', cursorOffset: 0, meta: 'table' },
        { label: '\\cline{}', insertText: '\\cline{}', cursorOffset: -1, meta: 'table' },
        { label: '\\multicolumn{}{}{}', insertText: '\\multicolumn{}{}{}', cursorOffset: -5, meta: 'table' },
        { label: '\\multirow{}{}{}', insertText: '\\multirow{}{}{}', cursorOffset: -5, meta: 'table' },
        { label: '\\toprule', insertText: '\\toprule', cursorOffset: 0, meta: 'table' },
        { label: '\\midrule', insertText: '\\midrule', cursorOffset: 0, meta: 'table' },
        { label: '\\bottomrule', insertText: '\\bottomrule', cursorOffset: 0, meta: 'table' },
        { label: '\\cmidrule{}', insertText: '\\cmidrule{}', cursorOffset: -1, meta: 'table' },
        { label: '\\textwidth', insertText: '\\textwidth', cursorOffset: 0, meta: 'len' },
        { label: '\\linewidth', insertText: '\\linewidth', cursorOffset: 0, meta: 'len' },
        { label: '\\columnwidth', insertText: '\\columnwidth', cursorOffset: 0, meta: 'len' },
        { label: '\\onecolumn', insertText: '\\onecolumn', cursorOffset: 0, meta: 'layout' },
        { label: '\\twocolumn', insertText: '\\twocolumn', cursorOffset: 0, meta: 'layout' },
        { label: '\\appendix', insertText: '\\appendix', cursorOffset: 0, meta: 'layout' },
        { label: '\\abstract', insertText: '\\begin{abstract}\n\n\\end{abstract}', cursorOffset: -12, meta: 'env' },
        { label: '\\IEEEauthorblockN{}', insertText: '\\IEEEauthorblockN{}', cursorOffset: -1, meta: 'ieee' },
        { label: '\\IEEEauthorblockA{}', insertText: '\\IEEEauthorblockA{}', cursorOffset: -1, meta: 'ieee' },
        { label: '\\IEEEpubid{}', insertText: '\\IEEEpubid{}', cursorOffset: -1, meta: 'ieee' },
        { label: '\\IEEEpubidadjcol', insertText: '\\IEEEpubidadjcol', cursorOffset: 0, meta: 'ieee' },
        { label: '\\IEEEeqnarray', insertText: '\\IEEEeqnarray', cursorOffset: 0, meta: 'ieee' }
    ],

    async init() {
        // Tab switching
        this.elements.tabLibrary.addEventListener('click', () => this.switchTab('library'));
        this.elements.tabEditor.addEventListener('click', () => this.switchTab('editor'));

        // Compile button
        // Actions
        this.elements.btnCompile.addEventListener('click', () => this.compileCurrent());
        this.elements.btnDownloadCompiled.addEventListener('click', () => this.downloadCompiled());
        this.elements.btnSave.addEventListener('click', () => this.saveCurrent());
        this.elements.btnUpload.addEventListener('click', () => this.elements.fileUploadInput.click());
        this.elements.fileUploadInput.addEventListener('change', (e) => this.handleUpload(e));
        this.elements.btnDelete.addEventListener('click', () => this.deleteCurrent());

        // Sidebar tabs
        this.elements.sidebarTabFiles?.addEventListener('click', () => this.switchSidebarTab('files'));
        this.elements.sidebarTabOutline?.addEventListener('click', () => this.switchSidebarTab('outline'));
        this.elements.sidebarTabComments?.addEventListener('click', () => this.switchSidebarTab('comments'));

        // File actions
        this.elements.btnNewFile?.addEventListener('click', () => this.createNewFile());
        this.elements.btnRenameFile?.addEventListener('click', () => this.renameCurrentFile());
        this.elements.btnDuplicateFile?.addEventListener('click', () => this.duplicateCurrentFile());
        this.elements.fileSearchInput?.addEventListener('input', (e) => {
            this.state.fileFilter = e.target.value.toLowerCase();
            this.renderFileList();
        });

        // Default disabled until first successful compile
        this.elements.btnDownloadCompiled.disabled = true;

        // Find in TeX
        this.elements.btnFindTex?.addEventListener('click', () => this.toggleFindBar());
        this.elements.findClose?.addEventListener('click', () => this.closeFindBar());
        this.elements.findInput?.addEventListener('input', () => this.performFind());
        this.elements.findInput?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                if (e.shiftKey) {
                    this.findPrev();
                } else {
                    this.findNext();
                }
            }
        });
        this.elements.findNext?.addEventListener('click', () => this.findNext());
        this.elements.findPrev?.addEventListener('click', () => this.findPrev());

        // Find in PDF
        this.elements.btnFindPdf?.addEventListener('click', () => this.togglePdfFindBar());
        this.elements.pdfFindClose?.addEventListener('click', () => this.closePdfFindBar());
        this.elements.pdfFindInput?.addEventListener('input', () => this.performPdfFind());
        this.elements.pdfFindNext?.addEventListener('click', () => this.pdfFindNext());
        this.elements.pdfFindPrev?.addEventListener('click', () => this.pdfFindPrev());

        // History
        this.elements.btnHistory?.addEventListener('click', () => this.showHistory());
        this.elements.historyClose?.addEventListener('click', () => this.closeHistory());
        this.elements.historyModal?.querySelector('.history-backdrop')?.addEventListener('click', () => this.closeHistory());
        this.elements.historyTabActivity?.addEventListener('click', () => this.switchHistoryTab('activity'));
        this.elements.historyTabVersions?.addEventListener('click', () => this.switchHistoryTab('versions'));
        this.elements.historyFileSelect?.addEventListener('change', () => this.renderVersionHistory());
        this.elements.historyRefresh?.addEventListener('click', () => this.renderVersionHistory());
        this.elements.historyRestoreVersion?.addEventListener('click', () => this.restoreSelectedVersion());
        this.elements.historyCopyVersion?.addEventListener('click', () => this.copySelectedVersion());

        // Zoom
        this.elements.btnZoomIn?.addEventListener('click', () => this.zoomIn());
        this.elements.btnZoomOut?.addEventListener('click', () => this.zoomOut());
        this.elements.btnToggleCitationHighlights?.addEventListener('click', () => {
            this.elements.previewView.classList.toggle('highlight-citations');
            const isActive = this.elements.previewView.classList.contains('highlight-citations');
            this.elements.btnToggleCitationHighlights.classList.toggle('active', isActive);
            showToast(isActive ? 'Citation highlights enabled' : 'Citation highlights disabled');
        });

        // Auto save/compile + share
        this.elements.btnAutoSave?.addEventListener('click', () => this.toggleAutoSave());
        this.elements.btnAutoCompile?.addEventListener('click', () => this.toggleAutoCompile());
        this.elements.btnShare?.addEventListener('click', () => this.copyShareLink());

        // Sync + compile log
        this.elements.btnSyncPdf?.addEventListener('click', () => this.syncSourceToPdf());
        this.elements.btnLogs?.addEventListener('click', () => this.toggleCompileLog());
        this.elements.compileLogClose?.addEventListener('click', () => this.hideCompileLog());
        this.elements.compileLogCopy?.addEventListener('click', () => this.copyCompileLog());
        this.elements.compileLogFilter?.addEventListener('change', () => this.renderCompileLog());

        // Comments
        this.elements.btnAddComment?.addEventListener('click', () => this.addComment());
        this.elements.commentInput?.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                this.addComment();
            }
        });
        this.elements.commentList?.addEventListener('click', (e) => this.handleCommentAction(e));

        // Line numbers toggle
        this.elements.btnLineNumbers?.addEventListener('click', () => this.toggleLineNumbers());
        this.elements.btnLineWrap?.addEventListener('click', () => this.toggleLineWrap());
        this.elements.btnModeCode?.addEventListener('click', () => this.setEditorMode('code'));
        this.elements.btnModeVisual?.addEventListener('click', () => this.setEditorMode('visual'));
        this.elements.visualToolbar?.addEventListener('click', (e) => this.handleVisualToolbar(e));
        this.elements.visualEditor?.addEventListener('input', () => this.handleVisualInput());
        this.elements.visualEditor?.addEventListener('keydown', (e) => this.handleVisualKeydown(e));
        this.setupLatexAutocomplete();

        // Panel resizer
        this.setupPanelResizer();

        // Sidebar toggle
        this.setupSidebarOverlay();
        this.elements.btnToggleSidebar?.addEventListener('click', () => this.toggleSidebar());

        // Mobile Tab switching
        this.elements.btnShowSource?.addEventListener('click', () => this.switchMobileView('source'));
        this.elements.btnShowPreview?.addEventListener('click', () => this.switchMobileView('preview'));

        // Initialize mobile view
        if (window.innerWidth <= 768) {
            this.switchMobileView('source');
        }

        if (this.isMobileEditorView()) {
            this.setSidebarCollapsed(true);
        }

        // Update line numbers on editor scroll/input
        this.elements.codeEditor?.addEventListener('scroll', () => {
            this.syncLineNumberScroll();
            this.syncHighlightScroll();
            this.positionLatexAutocomplete();
        });
        this.elements.codeEditor?.addEventListener('input', () => this.handleEditorInput());
        this.elements.codeEditor?.addEventListener('keyup', () => {
            this.updateCurrentLine();
            this.captureCommentSelection();
            this.updateLatexAutocomplete();
        });
        this.elements.codeEditor?.addEventListener('click', () => {
            this.updateCurrentLine();
            this.captureCommentSelection();
            this.updateLatexAutocomplete();
        });
        this.elements.codeEditor?.addEventListener('mouseup', () => {
            this.captureCommentSelection();
            this.updateLatexAutocomplete();
        });

        // Editor shortcuts
        document.addEventListener('keydown', (e) => {
            if (this.elements.editorView.style.display !== 'none') {
                if ((e.ctrlKey || e.metaKey) && e.key === 's') {
                    e.preventDefault();
                    this.saveAndCompile(); // Auto-compile on save
                }
                if ((e.ctrlKey || e.metaKey) && e.key === 'f' && !e.shiftKey) {
                    e.preventDefault();
                    this.toggleFindBar();
                }
                if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'F') {
                    e.preventDefault();
                    this.togglePdfFindBar();
                }
            }
        });

        // Load history from localStorage
        this.loadHistory();
        this.loadVersionHistory();
        this.loadComments();
        this.loadAutoPreferences();

        // Load line numbers preference
        this.loadLineNumbersPreference();
        this.applyLineWrap();
        this.applyEditorMode();
        this.loadEditorPreferences();

        // Load citation map
        await this.loadCitationMap();

        this.updateEditorStats();
        this.updateCommentContext();

        window.addEventListener('resize', () => this.handleEditorResize());
    },

    async switchTab(tab) {
        if (tab === 'library') {
            this.elements.libraryView.style.display = 'block';
            this.elements.editorView.style.display = 'none';
            this.elements.tabLibrary.classList.add('active');
            this.elements.tabEditor.classList.remove('active');
            loadPapers();
            this.hideLatexAutocomplete();
            // Show library tools
            document.querySelector('.toolbar').style.display = 'block';
            this.hideCompileLog();
            if (this.isMobileEditorView()) {
                this.setSidebarCollapsed(true);
            } else {
                this.elements.sidebarBackdrop?.classList.remove('visible');
            }
            if (this.state.autoSaveTimer) {
                clearTimeout(this.state.autoSaveTimer);
                this.state.autoSaveTimer = null;
            }
        } else {
            this.elements.libraryView.style.display = 'none';
            this.elements.editorView.style.display = 'flex';
            this.elements.tabLibrary.classList.remove('active');
            this.elements.tabEditor.classList.add('active');
            this.loadFiles();
            // Hide library tools
            document.querySelector('.toolbar').style.display = 'none';

            // On mobile, default to source view when entering editor
            if (window.innerWidth <= 768) {
                this.switchMobileView('source');
            }
            if (this.isMobileEditorView()) {
                this.setSidebarCollapsed(true);
            }
        }
    },

    switchMobileView(view) {
        if (view === 'source') {
            this.elements.editorContentArea.classList.add('show-source');
            this.elements.editorContentArea.classList.remove('show-preview');
            this.elements.btnShowSource.classList.add('active');
            this.elements.btnShowPreview.classList.remove('active');
        } else {
            this.elements.editorContentArea.classList.remove('show-source');
            this.elements.editorContentArea.classList.add('show-preview');
            this.elements.btnShowSource.classList.remove('active');
            this.elements.btnShowPreview.classList.add('active');
            this.hideLatexAutocomplete();

            // If viewing preview, make sure PDF is rendered
            if (this.state.lastCompiledPdfUrl) {
                this.loadPDF(this.state.lastCompiledPdfUrl);
            }
        }
    },

    switchSidebarTab(tab) {
        const tabs = {
            files: this.elements.sidebarTabFiles,
            outline: this.elements.sidebarTabOutline,
            comments: this.elements.sidebarTabComments
        };

        const panels = {
            files: this.elements.sidebarPanelFiles,
            outline: this.elements.sidebarPanelOutline,
            comments: this.elements.sidebarPanelComments
        };

        Object.keys(tabs).forEach(key => {
            tabs[key]?.classList.toggle('active', key === tab);
            panels[key]?.classList.toggle('active', key === tab);
        });

        if (this.isMobileEditorView()) {
            this.setSidebarCollapsed(false);
        }
    },

    switchHistoryTab(tab) {
        this.elements.historyTabActivity?.classList.toggle('active', tab === 'activity');
        this.elements.historyTabVersions?.classList.toggle('active', tab === 'versions');
        this.elements.historyPanelActivity?.classList.toggle('active', tab === 'activity');
        this.elements.historyPanelVersions?.classList.toggle('active', tab === 'versions');
    },

    toggleAutoSave() {
        this.state.autoSaveEnabled = !this.state.autoSaveEnabled;
        localStorage.setItem('paperreader_autosave', this.state.autoSaveEnabled ? 'true' : 'false');
        this.updateAutoButtons();
        showToast(this.state.autoSaveEnabled ? 'Auto save enabled' : 'Auto save disabled');
    },

    toggleAutoCompile() {
        this.state.autoCompileEnabled = !this.state.autoCompileEnabled;
        localStorage.setItem('paperreader_autocompile', this.state.autoCompileEnabled ? 'true' : 'false');
        this.updateAutoButtons();
        showToast(this.state.autoCompileEnabled ? 'Auto compile enabled' : 'Auto compile disabled');
    },

    loadAutoPreferences() {
        const autoSave = localStorage.getItem('paperreader_autosave');
        const autoCompile = localStorage.getItem('paperreader_autocompile');
        this.state.autoSaveEnabled = autoSave === 'true';
        this.state.autoCompileEnabled = autoCompile === 'true';
        this.updateAutoButtons();
    },

    updateAutoButtons() {
        this.elements.btnAutoSave?.classList.toggle('active', this.state.autoSaveEnabled);
        this.elements.btnAutoCompile?.classList.toggle('active', this.state.autoCompileEnabled);
    },

    copyShareLink() {
        const project = this.getActiveProject();
        if (project?.type === 'local') {
            showToast('Share link is only available for server projects');
            return;
        }
        let url = window.location.href;
        if (project?.id && project.id !== 'server') {
            const params = new URLSearchParams(window.location.search);
            params.set('project', project.id);
            url = `${window.location.origin}${window.location.pathname}?${params.toString()}`;
        }
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(url).then(() => {
                showToast('Share link copied');
            }).catch(() => {
                prompt('Copy share link:', url);
            });
        } else {
            prompt('Copy share link:', url);
        }
    },

    setupSidebarOverlay() {
        if (this.elements.sidebarBackdrop) return;
        const backdrop = document.createElement('div');
        backdrop.className = 'editor-sidebar-backdrop';
        backdrop.addEventListener('click', () => this.setSidebarCollapsed(true));
        document.body.appendChild(backdrop);
        this.elements.sidebarBackdrop = backdrop;
    },

    toggleSidebar() {
        const isCollapsed = this.elements.editorSidebar?.classList.contains('collapsed');
        this.setSidebarCollapsed(!isCollapsed);
    },

    setSidebarCollapsed(collapsed) {
        if (!this.elements.editorSidebar) return;
        this.elements.editorSidebar.classList.toggle('collapsed', collapsed);
        this.elements.btnToggleSidebar?.classList.toggle('active', !collapsed);
        if (this.isMobileEditorView()) {
            this.elements.sidebarBackdrop?.classList.toggle('visible', !collapsed);
        } else {
            this.elements.sidebarBackdrop?.classList.remove('visible');
        }
    },

    handleEditorResize() {
        if (this.isMobileEditorView()) {
            if (!this.elements.editorSidebar?.classList.contains('collapsed')) {
                this.elements.sidebarBackdrop?.classList.add('visible');
            }
        } else {
            this.elements.sidebarBackdrop?.classList.remove('visible');
        }
        this.positionLatexAutocomplete();
    },

    isMobileEditorView() {
        return window.matchMedia('(max-width: 900px)').matches;
    },

    getActiveProject() {
        return ProjectManager.activeProject || null;
    },

    isLocalProject() {
        return ProjectManager.isLocalProject();
    },

    getProjectStorageKey(baseKey) {
        const projectId = this.getActiveProject()?.id || 'default';
        return `${baseKey}_${projectId}`;
    },

    getCompiledPdfUrl(filename) {
        return this.withProjectParam(`${API_BASE}/api/project/file/${filename}`);
    },

    saveLastCompiledPdf(filename) {
        if (!filename) return;
        try {
            const key = this.getProjectStorageKey('paperreader_last_compiled_pdf');
            localStorage.setItem(key, JSON.stringify({ filename, savedAt: Date.now() }));
        } catch (e) {
            console.warn('Failed to save last compiled PDF', e);
        }
    },

    loadLastCompiledPdf() {
        try {
            const key = this.getProjectStorageKey('paperreader_last_compiled_pdf');
            const raw = localStorage.getItem(key);
            if (!raw) return null;
            const data = JSON.parse(raw);
            if (!data?.filename) return null;
            return data;
        } catch (e) {
            console.warn('Failed to load last compiled PDF', e);
            return null;
        }
    },

    restoreLastCompiledPdf() {
        if (this.isLocalProject()) return;
        if (this.state.lastCompiledPdfUrl) return;
        const saved = this.loadLastCompiledPdf();
        if (!saved) return;
        const url = this.getCompiledPdfUrl(saved.filename);
        this.state.lastCompiledPdfUrl = url;
        this.elements.btnDownloadCompiled.disabled = false;
        this.loadPDF(saved.filename);
    },

    withProjectParam(url) {
        const project = this.getActiveProject();
        if (!project || project.type !== 'remote' || !project.id || project.id === 'server') return url;
        const joiner = url.includes('?') ? '&' : '?';
        return `${url}${joiner}project_id=${encodeURIComponent(project.id)}`;
    },

    handleProjectSwitch() {
        this.state.currentFile = null;
        this.state.files = [];
        this.state.fileContents = {};
        this.state.dirtyFiles = {};
        this.state.commentSelection = null;
        this.state.outlineItems = [];
        this.state.editHistory = [];
        this.state.versionHistory = [];
        this.state.selectedVersionId = null;
        this.state.historyFileSelected = null;
        this.state.lastCompiledPdfUrl = null;
        this.state.fileFilter = '';

        if (this.elements.fileSearchInput) {
            this.elements.fileSearchInput.value = '';
        }

        if (this.elements.currentFileName) {
            this.elements.currentFileName.textContent = 'Select a file...';
        }
        if (this.elements.codeEditor) {
            this.elements.codeEditor.value = '';
            this.elements.codeEditor.disabled = true;
        }
        if (this.elements.codeHighlightContent) {
            this.elements.codeHighlightContent.innerHTML = '';
        }
        if (this.elements.visualEditor) {
            this.elements.visualEditor.innerHTML = '';
            this.setVisualEditorEnabled(false);
        }
        this.elements.btnSave.disabled = true;
        this.elements.btnDelete.style.display = 'none';
        this.elements.btnDownloadCompiled.disabled = true;
        if (this.elements.btnCompile) {
            const isLocal = this.isLocalProject();
            this.elements.btnCompile.disabled = isLocal;
            this.elements.btnCompile.title = isLocal ? 'Compile requires a server project' : 'Recompile';
        }
        if (this.elements.btnAutoCompile) {
            const isLocal = this.isLocalProject();
            this.elements.btnAutoCompile.disabled = isLocal;
            this.elements.btnAutoCompile.title = isLocal ? 'Auto compile requires a server project' : 'Toggle auto compile';
        }

        this.loadHistory();
        this.loadVersionHistory();
        this.loadComments();
        this.loadCitationMap();
        this.renderComments();
        this.updateEditorStats();
        this.buildOutlineFromContent('');
        this.loadFiles();
        this.renderFileList();
    },

    async loadFiles() {
        try {
            if (this.isLocalProject()) {
                this.state.files = ProjectManager.getLocalFilesList();
            } else {
                const res = await fetch(this.withProjectParam(`${API_BASE}/api/project/files`));
                if (!res.ok) throw new Error("Failed to load files");
                const data = await res.json();
                this.state.files = data.files || [];
            }
            this.renderFileList();
            this.updateHistoryFileOptions();
            this.restoreLastCompiledPdf();

            if (this.state.currentFile) {
                const exists = this.state.files.find(f => f.name === this.state.currentFile);
                if (!exists) {
                    this.state.currentFile = null;
                    this.elements.currentFileName.textContent = "Select a file...";
                    this.elements.codeEditor.value = "";
                    this.elements.codeEditor.disabled = true;
                    this.elements.btnSave.disabled = true;
                    this.elements.btnDelete.style.display = 'none';
                    this.state.commentSelection = null;
                    this.updateCommentContext();
                    this.renderComments();
                    this.buildOutlineFromContent('');
                    this.updateEditorStats();
                }
            }
        } catch (e) {
            console.error("Failed to load files", e);
            showToast("Failed to load files");
        }
    },

    renderFileList() {
        this.elements.fileList.innerHTML = '';
        const filter = this.state.fileFilter || '';
        const files = this.state.files.filter(file => file.name.toLowerCase().includes(filter));

        if (files.length === 0) {
            const message = filter ? 'No files match the filter' : 'No files yet';
            this.elements.fileList.innerHTML = `<div class="outline-empty">${message}</div>`;
            return;
        }

        files.forEach(file => {
            const el = document.createElement('div');
            el.className = `file-item ${this.state.currentFile === file.name ? 'active' : ''}`;

            let icon = 'bi-file-earmark';
            if (file.type === 'tex') icon = 'bi-file-earmark-code';
            if (file.type === 'bib') icon = 'bi-file-earmark-text';

            const dirty = this.state.dirtyFiles[file.name];
            const size = this.formatFileSize(file.size);
            el.innerHTML = `
                <i class="bi ${icon}"></i>
                <span class="file-name">${escapeHtml(file.name)}</span>
                <span class="file-meta">
                    ${dirty ? '<span class="file-dirty">*</span>' : ''}
                    ${size}
                </span>
            `;
            el.title = file.name;
            el.onclick = () => this.selectFile(file);
            this.elements.fileList.appendChild(el);
        });
    },

    async selectFile(file) {
        if (this.state.autoSaveTimer) {
            clearTimeout(this.state.autoSaveTimer);
            this.state.autoSaveTimer = null;
        }
        if (this.state.currentFile && this.isDirty(this.state.currentFile)) {
            const shouldSave = confirm('You have unsaved changes. Save before switching?');
            if (shouldSave) {
                const saved = await this.saveCurrent();
                if (!saved) return;
            } else {
                const discard = confirm('Discard unsaved changes?');
                if (!discard) return;
            }
        }

        this.state.currentFile = file.name;
        this.updateCurrentFileLabel();
        this.renderFileList();
        if (this.isMobileEditorView()) {
            this.setSidebarCollapsed(true);
        }

        // Enable/Disable controls
        this.elements.btnSave.disabled = false;
        this.elements.codeEditor.disabled = this.state.editorMode === 'visual';
        this.setVisualEditorEnabled(true);
        this.elements.btnDelete.style.display = 'inline-block';

        // Load content
        try {
            this.elements.codeEditor.value = "Loading...";
            if (this.isLocalProject()) {
                const content = ProjectManager.getLocalFileContent(file.name) || '';
                this.elements.codeEditor.value = content;
                this.state.fileContents[file.name] = content;
                if (file.name.toLowerCase().endsWith('.bib')) {
                    this.updateCitationKeysFromBibContent(content);
                }
                this.state.dirtyFiles[file.name] = false;
                this.state.commentSelection = null;
                this.updateCommentContext();
                this.updateCurrentFileLabel();
                this.updateLineNumbers();
                this.updateCurrentLine();
                this.updateEditorStats();
                this.buildOutlineFromContent(content);
                this.renderComments();
                this.updateCodeHighlight();
                if (this.state.editorMode === 'visual') {
                    this.syncCodeToVisual();
                }
                return;
            }

            const res = await fetch(this.withProjectParam(`${API_BASE}/api/project/content/${file.name}`));
            if (!res.ok) throw new Error("Failed to load content");
            const data = await res.json();
            this.elements.codeEditor.value = data.content;
            this.state.fileContents[file.name] = data.content;
            if (file.name.toLowerCase().endsWith('.bib')) {
                this.updateCitationKeysFromBibContent(data.content);
            }
            this.state.dirtyFiles[file.name] = false;
            this.state.commentSelection = null;
            this.updateCommentContext();
            this.updateCurrentFileLabel();

            // Update line numbers after content loads
            this.updateLineNumbers();
            this.updateCurrentLine();
            this.updateEditorStats();
            this.buildOutlineFromContent(data.content);
            this.renderComments();
            this.updateCodeHighlight();
            if (this.state.editorMode === 'visual') {
                this.syncCodeToVisual();
            }
        } catch (e) {
            console.error(e);
            this.elements.codeEditor.value = "// Error loading content or binary file";
            this.updateLineNumbers();
            this.updateEditorStats();
        }
    },

    handleEditorInput() {
        this.updateLineNumbers();
        this.updateCurrentLine();
        this.updateEditorStats();
        this.updateDirtyState();
        this.scheduleOutlineUpdate();
        this.scheduleAutoSave();
        this.updateLatexAutocomplete();
        this.scheduleHighlightUpdate();
    },

    updateDirtyState() {
        if (!this.state.currentFile) return;
        const content = this.elements.codeEditor.value;
        const savedContent = this.state.fileContents[this.state.currentFile] ?? '';
        const isDirty = content !== savedContent;
        this.state.dirtyFiles[this.state.currentFile] = isDirty;
        this.updateCurrentFileLabel();
        this.renderFileList();
    },

    isDirty(filename) {
        return !!this.state.dirtyFiles[filename];
    },

    updateCurrentFileLabel() {
        if (!this.state.currentFile || !this.elements.currentFileName) return;
        const dirty = this.isDirty(this.state.currentFile) ? ' *' : '';
        this.elements.currentFileName.textContent = `${this.state.currentFile}${dirty}`;
    },

    updateEditorStats() {
        const statsEl = this.elements.editorStats;
        const editor = this.elements.codeEditor;
        if (!statsEl || !editor || !this.state.currentFile) {
            if (statsEl) statsEl.textContent = 'WORDS 0 • LINES 0 • CHARS 0';
            return;
        }

        const content = editor.value || '';
        const lines = content.length ? content.split('\n').length : 0;
        const words = content.trim() ? content.trim().split(/\s+/).length : 0;
        const chars = content.length;
        statsEl.textContent = `WORDS ${words} • LINES ${lines} • CHARS ${chars}`;
    },

    setupLatexAutocomplete() {
        const editor = this.elements.codeEditor;
        const box = this.elements.latexAutocomplete;
        const list = this.elements.latexAutocompleteList;
        if (!editor || !box || !list) return;

        editor.addEventListener('keydown', (e) => this.handleAutocompleteKeydown(e));
        box.addEventListener('mousedown', (e) => e.preventDefault());

        list.addEventListener('click', (e) => {
            const item = e.target.closest('.autocomplete-item');
            if (!item) return;
            const index = Number(item.dataset.index);
            const selected = this.state.autocomplete.items[index];
            if (!selected || selected.disabled) return;
            this.applyAutocompleteItem(selected);
        });

        list.addEventListener('mousemove', (e) => {
            const item = e.target.closest('.autocomplete-item');
            if (!item) return;
            const index = Number(item.dataset.index);
            if (Number.isNaN(index)) return;
            this.state.autocomplete.activeIndex = index;
            this.updateAutocompleteActiveItem();
        });

        document.addEventListener('mousedown', (e) => {
            if (box.contains(e.target) || e.target === editor) return;
            this.hideLatexAutocomplete();
        });
    },

    updateLatexAutocomplete() {
        const editor = this.elements.codeEditor;
        if (!editor || editor.disabled || !this.state.currentFile) {
            this.hideLatexAutocomplete();
            return;
        }
        if (!this.isSourceViewVisible()) {
            this.hideLatexAutocomplete();
            return;
        }
        if (editor.selectionStart === null || editor.selectionStart !== editor.selectionEnd) {
            this.hideLatexAutocomplete();
            return;
        }

        const context = this.getAutocompleteContext(editor.value, editor.selectionStart);
        if (!context) {
            this.hideLatexAutocomplete();
            return;
        }

        let items = [];
        if (context.type === 'command') {
            items = this.getLatexCommandSuggestions(context.query);
        } else if (context.type === 'citekey') {
            items = this.getCitationKeySuggestions(context.query);
        }

        if (!items.length) {
            this.hideLatexAutocomplete();
            return;
        }

        const firstEnabled = items.findIndex(item => !item.disabled);
        if (firstEnabled < 0) {
            this.hideLatexAutocomplete();
            return;
        }

        this.state.autocomplete.visible = true;
        this.state.autocomplete.mode = context.type;
        this.state.autocomplete.items = items;
        this.state.autocomplete.activeIndex = firstEnabled;
        this.state.autocomplete.context = context;
        this.renderLatexAutocomplete();
        this.positionLatexAutocomplete();
    },

    renderLatexAutocomplete() {
        const box = this.elements.latexAutocomplete;
        const list = this.elements.latexAutocompleteList;
        const header = this.elements.latexAutocompleteHeader;
        if (!box || !list || !header) return;

        const { items, mode, activeIndex } = this.state.autocomplete;
        header.textContent = mode === 'citekey' ? 'CITATION KEYS' : 'LATEX COMMANDS';

        list.innerHTML = items.map((item, index) => {
            const disabled = item.disabled ? ' disabled' : '';
            const active = index === activeIndex ? ' active' : '';
            const label = this.escapeHtml(item.label);
            const meta = item.meta ? `<span class="item-meta">${this.escapeHtml(item.meta)}</span>` : '';
            return `
                <div class="autocomplete-item${disabled}${active}" data-index="${index}" role="option" aria-disabled="${item.disabled ? 'true' : 'false'}">
                    <span class="item-label">${label}</span>
                    ${meta}
                </div>
            `;
        }).join('');

        box.classList.add('visible');
        box.setAttribute('aria-hidden', 'false');
    },

    updateAutocompleteActiveItem() {
        const list = this.elements.latexAutocompleteList;
        if (!list) return;
        const items = list.querySelectorAll('.autocomplete-item');
        items.forEach((item, index) => {
            item.classList.toggle('active', index === this.state.autocomplete.activeIndex);
        });
    },

    hideLatexAutocomplete() {
        const box = this.elements.latexAutocomplete;
        if (!box) return;
        box.classList.remove('visible');
        box.setAttribute('aria-hidden', 'true');
        this.state.autocomplete.visible = false;
        this.state.autocomplete.items = [];
        this.state.autocomplete.activeIndex = -1;
        this.state.autocomplete.context = null;
    },

    handleAutocompleteKeydown(e) {
        if (!this.state.autocomplete.visible) return;

        if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            e.preventDefault();
            const direction = e.key === 'ArrowDown' ? 1 : -1;
            this.moveAutocompleteSelection(direction);
            return;
        }

        if (e.key === 'Enter' || e.key === 'Tab') {
            const selected = this.state.autocomplete.items[this.state.autocomplete.activeIndex];
            if (selected && !selected.disabled) {
                e.preventDefault();
                this.applyAutocompleteItem(selected);
            }
            return;
        }

        if (e.key === 'Escape') {
            e.preventDefault();
            this.hideLatexAutocomplete();
        }
    },

    moveAutocompleteSelection(direction) {
        const items = this.state.autocomplete.items;
        if (!items.length) return;
        let index = this.state.autocomplete.activeIndex;
        for (let i = 0; i < items.length; i += 1) {
            index = (index + direction + items.length) % items.length;
            if (!items[index].disabled) {
                this.state.autocomplete.activeIndex = index;
                this.updateAutocompleteActiveItem();
                return;
            }
        }
    },

    applyAutocompleteItem(item) {
        const editor = this.elements.codeEditor;
        if (!editor) return;
        const cursor = editor.selectionStart;
        if (cursor === null) return;

        const context = this.getAutocompleteContext(editor.value, cursor);
        if (!context) return;

        const text = editor.value;
        let insertText = item.insertText || item.label;

        if (context.type === 'command') {
            const start = context.replaceStart;
            const end = context.replaceEnd;
            const updated = text.slice(0, start) + insertText + text.slice(end);
            editor.value = updated;
            const cursorPos = start + insertText.length + (item.cursorOffset || 0);
            editor.focus();
            editor.setSelectionRange(cursorPos, cursorPos);
        } else if (context.type === 'citekey') {
            const start = context.replaceStart;
            const end = context.replaceEnd;
            const updated = text.slice(0, start) + insertText + text.slice(end);
            editor.value = updated;
            const cursorPos = start + insertText.length;
            editor.focus();
            editor.setSelectionRange(cursorPos, cursorPos);
        }

        this.hideLatexAutocomplete();
        this.handleEditorInput();
    },

    positionLatexAutocomplete() {
        const box = this.elements.latexAutocomplete;
        const editor = this.elements.codeEditor;
        if (!box || !editor || !this.state.autocomplete.visible) return;
        if (!this.elements.sourceView) return;

        const coords = this.getCaretCoordinates(editor, editor.selectionStart || 0);
        const sourceRect = this.elements.sourceView.getBoundingClientRect();
        const editorRect = editor.getBoundingClientRect();
        let left = editorRect.left - sourceRect.left + coords.left + 6;
        let top = editorRect.top - sourceRect.top + coords.top + coords.height + 6;

        const maxLeft = this.elements.sourceView.clientWidth - box.offsetWidth - 12;
        if (maxLeft > 0) {
            left = Math.min(Math.max(left, 8), maxLeft);
        }

        const belowMax = this.elements.sourceView.clientHeight - box.offsetHeight - 8;
        if (top > belowMax && coords.top > box.offsetHeight) {
            top = editorRect.top - sourceRect.top + coords.top - box.offsetHeight - 8;
        }
        if (belowMax > 0) {
            top = Math.min(Math.max(top, 8), belowMax);
        }

        box.style.left = `${left}px`;
        box.style.top = `${top}px`;
    },

    getCaretCoordinates(textarea, position) {
        if (!this.state.autocompleteMirror) {
            const mirror = document.createElement('div');
            mirror.style.position = 'absolute';
            mirror.style.visibility = 'hidden';
            mirror.style.whiteSpace = 'pre-wrap';
            mirror.style.wordWrap = 'break-word';
            mirror.style.overflow = 'hidden';
            mirror.style.top = '-9999px';
            mirror.style.left = '-9999px';
            document.body.appendChild(mirror);
            this.state.autocompleteMirror = mirror;
        }

        const mirror = this.state.autocompleteMirror;
        const style = window.getComputedStyle(textarea);
        const props = [
            'boxSizing',
            'fontFamily',
            'fontSize',
            'fontWeight',
            'fontStyle',
            'letterSpacing',
            'textTransform',
            'textIndent',
            'lineHeight',
            'paddingTop',
            'paddingRight',
            'paddingBottom',
            'paddingLeft',
            'borderTopWidth',
            'borderRightWidth',
            'borderBottomWidth',
            'borderLeftWidth',
            'tabSize'
        ];

        props.forEach((prop) => {
            mirror.style[prop] = style[prop];
        });

        mirror.style.width = style.width;
        mirror.textContent = textarea.value.substring(0, position);
        const span = document.createElement('span');
        span.textContent = '\u200b';
        mirror.appendChild(span);

        const top = span.offsetTop - textarea.scrollTop;
        const left = span.offsetLeft - textarea.scrollLeft;
        const height = parseFloat(style.lineHeight) || span.offsetHeight || 16;

        mirror.innerHTML = '';
        return { top, left, height };
    },

    getAutocompleteContext(text, cursor) {
        const before = text.slice(0, cursor);
        const citeMatch = before.match(/\\cite[a-zA-Z*]*\{([^}]*)$/);
        if (citeMatch) {
            const matchText = citeMatch[0];
            const matchStart = before.length - matchText.length;
            const braceIndex = matchText.lastIndexOf('{');
            const insideStart = matchStart + braceIndex + 1;
            const insideValue = citeMatch[1];
            const lastComma = insideValue.lastIndexOf(',');
            let replaceStart = insideStart;
            if (lastComma >= 0) {
                replaceStart = insideStart + lastComma + 1;
                const afterComma = insideValue.slice(lastComma + 1);
                const leadingSpaces = afterComma.match(/^\s*/)?.[0]?.length || 0;
                replaceStart += leadingSpaces;
            }
            const query = insideValue.slice(lastComma + 1).trim();
            return {
                type: 'citekey',
                query,
                replaceStart,
                replaceEnd: cursor
            };
        }

        const commandMatch = before.match(/\\[a-zA-Z]*$/);
        if (commandMatch) {
            const matchText = commandMatch[0];
            const matchStart = before.length - matchText.length;
            return {
                type: 'command',
                query: matchText.slice(1),
                replaceStart: matchStart,
                replaceEnd: cursor
            };
        }

        return null;
    },

    getLatexCommandSuggestions(query) {
        const q = (query || '').toLowerCase();
        return this.latexCommandSuggestions
            .filter(item => item.label.toLowerCase().includes(q))
            .slice(0, 10);
    },

    getCitationKeySuggestions(query) {
        if (!this.state.citationKeys.length) {
            const bibContent = Object.entries(this.state.fileContents)
                .filter(([name]) => name.toLowerCase().endsWith('.bib'))
                .map(([, content]) => content)
                .join('\n');
            if (bibContent) {
                this.updateCitationKeysFromBibContent(bibContent);
            }
        }

        const keys = this.state.citationKeys;
        if (!keys.length) return [];

        const q = (query || '').toLowerCase();
        return keys
            .filter((key) => key.toLowerCase().includes(q))
            .slice(0, 12)
            .map((key) => ({
                label: key,
                insertText: key,
                meta: 'bib'
            }));
    },

    updateCitationKeysFromBibContent(content) {
        const keys = [];
        const regex = /@\w+\s*\{\s*([^,\s]+)\s*,/g;
        let match;
        while ((match = regex.exec(content)) !== null) {
            keys.push(match[1]);
        }
        if (!keys.length) return;
        const existing = new Set(this.state.citationKeys);
        keys.forEach(key => existing.add(key));
        this.state.citationKeys = Array.from(existing).sort();
    },

    isSourceViewVisible() {
        if (this.state.editorMode === 'visual') {
            return false;
        }
        if (window.innerWidth <= 768) {
            return this.elements.editorContentArea?.classList.contains('show-source');
        }
        return true;
    },

    formatFileSize(bytes) {
        if (bytes === undefined || bytes === null) return '';
        if (bytes < 1024) return `${bytes} B`;
        const kb = bytes / 1024;
        if (kb < 1024) return `${kb.toFixed(1)} KB`;
        const mb = kb / 1024;
        return `${mb.toFixed(2)} MB`;
    },

    sanitizeFilename(name) {
        const trimmed = (name || '').trim();
        if (!trimmed) return null;
        if (trimmed.includes('..') || trimmed.includes('/') || trimmed.includes('\\')) return null;
        return trimmed;
    },

    async createNewFile() {
        let name = prompt('Enter new filename (e.g. section.tex):');
        name = this.sanitizeFilename(name);
        if (!name) return;

        if (!name.includes('.')) {
            const addExt = confirm('No extension detected. Add .tex?');
            if (addExt) name += '.tex';
        }

        if (this.state.files.find(f => f.name === name)) {
            alert('A file with that name already exists.');
            return;
        }

        const ok = await this.saveFileContent(name, '');
        if (ok) {
            this.saveToHistory('Created file', name);
            this.state.fileFilter = '';
            if (this.elements.fileSearchInput) {
                this.elements.fileSearchInput.value = '';
            }
            await this.loadFiles();
            const file = this.state.files.find(f => f.name === name);
            if (file) this.selectFile(file);
        }
    },

    async renameCurrentFile() {
        if (!this.state.currentFile) return;

        const oldName = this.state.currentFile;
        let name = prompt('Rename file:', oldName);
        name = this.sanitizeFilename(name);
        if (!name || name === oldName) return;

        if (this.state.files.find(f => f.name === name)) {
            alert('A file with that name already exists.');
            return;
        }

        const content = this.elements.codeEditor.value || '';
        const ok = await this.saveFileContent(name, content);
        if (!ok) return;

        if (this.isLocalProject()) {
            ProjectManager.renameLocalFile(oldName, name);
        } else {
            try {
                await fetch(this.withProjectParam(`${API_BASE}/api/project/file/${oldName}`), { method: 'DELETE' });
            } catch (e) {
                console.warn('Failed to delete old file after rename', e);
            }
        }

        this.state.currentFile = name;
        this.state.fileContents[name] = content;
        this.state.dirtyFiles[name] = false;
        delete this.state.fileContents[oldName];
        delete this.state.dirtyFiles[oldName];
        this.state.fileFilter = '';
        if (this.elements.fileSearchInput) {
            this.elements.fileSearchInput.value = '';
        }

        this.state.comments = this.state.comments.map(comment => {
            if (comment.filename === oldName) {
                return { ...comment, filename: name };
            }
            return comment;
        });
        this.saveComments();

        this.state.versionHistory = this.state.versionHistory.map(entry => {
            if (entry.filename === oldName) {
                return { ...entry, filename: name };
            }
            return entry;
        });
        try {
            localStorage.setItem(this.getProjectStorageKey('paperreader_versions'), JSON.stringify(this.state.versionHistory));
        } catch (e) {
            console.warn('Failed to update version history for rename');
        }

        await this.loadFiles();
        this.updateCurrentFileLabel();
        this.renderComments();
        this.saveToHistory(`Renamed ${oldName}`, name);
        showToast('File renamed');
    },

    async duplicateCurrentFile() {
        if (!this.state.currentFile) return;
        const base = this.state.currentFile.replace(/(\.[^.]*)$/, '');
        const ext = this.state.currentFile.includes('.') ? this.state.currentFile.split('.').pop() : '';
        const suggested = ext ? `${base}_copy.${ext}` : `${base}_copy`;
        let name = prompt('Duplicate file as:', suggested);
        name = this.sanitizeFilename(name);
        if (!name) return;

        if (this.state.files.find(f => f.name === name)) {
            alert('A file with that name already exists.');
            return;
        }

        const content = this.elements.codeEditor.value || '';
        const ok = await this.saveFileContent(name, content);
        if (ok) {
            this.saveToHistory('Duplicated file', name);
            await this.loadFiles();
            showToast('File duplicated');
        }
    },

    async saveFileContent(filename, content) {
        try {
            if (this.isLocalProject()) {
                const ok = ProjectManager.saveLocalFile(filename, content);
                if (!ok) throw new Error('Save failed');
                this.saveVersionSnapshot(filename, content, 'manual');
                if (filename.toLowerCase().endsWith('.bib')) {
                    this.updateCitationKeysFromBibContent(content);
                }
                if (this.getActiveProject()?.id) {
                    ProjectManager.touchProject(this.getActiveProject().id);
                }
                return true;
            }

            const res = await fetch(this.withProjectParam(`${API_BASE}/api/project/save`), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename, content })
            });
            if (!res.ok) throw new Error('Save failed');
            this.saveVersionSnapshot(filename, content, 'manual');
            if (filename.toLowerCase().endsWith('.bib')) {
                this.updateCitationKeysFromBibContent(content);
            }
            if (this.getActiveProject()?.id) {
                ProjectManager.touchProject(this.getActiveProject().id);
            }
            return true;
        } catch (e) {
            console.error('Save failed', e);
            alert('Failed to save file');
            return false;
        }
    },

    scheduleOutlineUpdate() {
        if (this.state.outlineTimer) {
            clearTimeout(this.state.outlineTimer);
        }

        this.state.outlineTimer = setTimeout(() => {
            const content = this.elements.codeEditor.value || '';
            this.buildOutlineFromContent(content);
        }, 300);
    },

    buildOutlineFromContent(content) {
        if (!this.elements.outlineList) return;
        if (!this.state.currentFile || !this.state.currentFile.endsWith('.tex')) {
            this.state.outlineItems = [];
            this.renderOutline();
            return;
        }

        const lines = content.split('\n');
        const items = [];
        const pattern = /\\(chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?(?:\[[^\]]*\])?\{([^}]+)\}/;

        lines.forEach((line, index) => {
            const match = line.match(pattern);
            if (!match) return;
            const levelMap = {
                chapter: 1,
                section: 1,
                subsection: 2,
                subsubsection: 3,
                paragraph: 4,
                subparagraph: 5
            };
            items.push({
                title: match[2],
                line: index + 1,
                level: levelMap[match[1]] || 1
            });
        });

        this.state.outlineItems = items;
        this.renderOutline();
    },

    renderOutline() {
        const container = this.elements.outlineList;
        if (!container) return;

        if (!this.state.currentFile || this.state.outlineItems.length === 0) {
            container.innerHTML = '<div class="outline-empty">No outline items</div>';
            return;
        }

        container.innerHTML = this.state.outlineItems.map(item => `
            <div class="outline-item level-${item.level}" data-line="${item.line}">
                <i class="bi bi-chevron-right"></i>
                <span>${escapeHtml(item.title)}</span>
            </div>
        `).join('');

        container.querySelectorAll('.outline-item').forEach(el => {
            el.addEventListener('click', () => {
                const line = parseInt(el.dataset.line, 10);
                if (line) this.gotoLine(line);
            });
        });
    },

    loadComments() {
        try {
            const saved = localStorage.getItem(this.getProjectStorageKey('paperreader_comments'));
            this.state.comments = saved ? JSON.parse(saved) : [];
        } catch (e) {
            this.state.comments = [];
        }
        this.renderComments();
    },

    saveComments() {
        try {
            localStorage.setItem(this.getProjectStorageKey('paperreader_comments'), JSON.stringify(this.state.comments));
        } catch (e) {
            console.warn('Failed to save comments');
        }
    },

    captureCommentSelection() {
        const editor = this.elements.codeEditor;
        if (!editor || editor.disabled || !this.state.currentFile) return;

        const start = editor.selectionStart;
        const end = editor.selectionEnd;
        if (start === end) {
            this.state.commentSelection = null;
            this.updateCommentContext();
            return;
        }

        const fullText = editor.value || '';
        const text = fullText.slice(start, end).trim();
        if (!text) return;

        const startLine = fullText.slice(0, start).split('\n').length;
        const endLine = fullText.slice(0, end).split('\n').length;
        this.state.commentSelection = { text, startLine, endLine };
        this.updateCommentContext();
    },

    updateCommentContext() {
        const el = this.elements.commentContext;
        if (!el) return;

        if (!this.state.commentSelection) {
            el.textContent = 'Select text to comment';
            return;
        }

        const { text, startLine, endLine } = this.state.commentSelection;
        const preview = text.length > 80 ? `${text.slice(0, 80)}...` : text;
        el.textContent = `Lines ${startLine}-${endLine}: ${preview}`;
    },

    addComment() {
        if (!this.state.currentFile) return;

        const text = (this.elements.commentInput?.value || '').trim();
        if (!text) {
            showToast('Comment text is empty');
            return;
        }

        if (!this.state.commentSelection) {
            showToast('Select text to comment');
            return;
        }

        const entry = {
            id: Date.now(),
            filename: this.state.currentFile,
            text,
            excerpt: this.state.commentSelection.text,
            lineStart: this.state.commentSelection.startLine,
            lineEnd: this.state.commentSelection.endLine,
            resolved: false,
            timestamp: new Date().toISOString()
        };

        this.state.comments.unshift(entry);
        this.saveComments();
        if (this.elements.commentInput) {
            this.elements.commentInput.value = '';
        }
        this.state.commentSelection = null;
        this.updateCommentContext();
        this.renderComments();
        showToast('Comment added');
    },

    renderComments() {
        const container = this.elements.commentList;
        if (!container) return;

        if (!this.state.currentFile) {
            container.innerHTML = '<div class="outline-empty">Select a file to view comments</div>';
            return;
        }

        const comments = this.state.comments.filter(c => c.filename === this.state.currentFile);
        if (comments.length === 0) {
            container.innerHTML = '<div class="outline-empty">No comments yet</div>';
            return;
        }

        container.innerHTML = comments.map(comment => {
            const date = new Date(comment.timestamp);
            const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            const lineLabel = `Lines ${comment.lineStart}-${comment.lineEnd}`;
            return `
                <div class="comment-item ${comment.resolved ? 'resolved' : ''}" data-id="${comment.id}">
                    <div class="comment-meta">
                        <span>${lineLabel}</span>
                        <span>${timeStr}</span>
                    </div>
                    <div class="comment-text">${escapeHtml(comment.text)}</div>
                    <div class="comment-actions">
                        <button class="btn-tool btn-small" data-action="goto">GO TO</button>
                        <button class="btn-tool btn-small" data-action="resolve">
                            ${comment.resolved ? 'REOPEN' : 'RESOLVE'}
                        </button>
                        <button class="btn-tool btn-small btn-danger" data-action="delete">DELETE</button>
                    </div>
                </div>
            `;
        }).join('');
    },

    handleCommentAction(event) {
        const target = event.target.closest('button');
        if (!target) return;
        const item = event.target.closest('.comment-item');
        if (!item) return;

        const id = parseInt(item.dataset.id, 10);
        const action = target.dataset.action;
        const comment = this.state.comments.find(c => c.id === id);
        if (!comment) return;

        if (action === 'goto') {
            this.gotoLine(comment.lineStart);
            return;
        }

        if (action === 'resolve') {
            comment.resolved = !comment.resolved;
            this.saveComments();
            this.renderComments();
            return;
        }

        if (action === 'delete') {
            if (!confirm('Delete this comment?')) return;
            this.state.comments = this.state.comments.filter(c => c.id !== id);
            this.saveComments();
            this.renderComments();
        }
    },

    loadVersionHistory() {
        try {
            const saved = localStorage.getItem(this.getProjectStorageKey('paperreader_versions'));
            this.state.versionHistory = saved ? JSON.parse(saved) : [];
        } catch (e) {
            this.state.versionHistory = [];
        }
        this.renderVersionHistory();
    },

    saveVersionSnapshot(filename, content, source = 'manual') {
        if (!filename) return;
        const existing = this.state.versionHistory.find(v => v.filename === filename);
        if (existing && existing.content === content) return;

        const now = Date.now();
        const lastTime = this.state.versionThrottle[filename] || 0;
        if (source === 'auto' && now - lastTime < 60000) {
            return;
        }
        this.state.versionThrottle[filename] = now;

        const entry = {
            id: now,
            filename,
            timestamp: new Date().toISOString(),
            content,
            source
        };

        this.state.versionHistory.unshift(entry);

        // Keep last 120 entries total
        if (this.state.versionHistory.length > 120) {
            this.state.versionHistory = this.state.versionHistory.slice(0, 120);
        }

        try {
            localStorage.setItem(this.getProjectStorageKey('paperreader_versions'), JSON.stringify(this.state.versionHistory));
        } catch (e) {
            console.warn('Failed to save version history');
        }

        if (this.elements.historyModal?.style.display === 'flex') {
            this.renderVersionHistory();
        }
    },

    updateHistoryFileOptions() {
        const select = this.elements.historyFileSelect;
        if (!select) return;

        const current = select.value;
        select.innerHTML = '';
        this.state.files.forEach(file => {
            const option = document.createElement('option');
            option.value = file.name;
            option.textContent = file.name;
            select.appendChild(option);
        });

        if (current && this.state.files.find(f => f.name === current)) {
            select.value = current;
        } else if (this.state.currentFile) {
            select.value = this.state.currentFile;
        }
    },

    renderVersionHistory() {
        const list = this.elements.historyVersionList;
        const previewTitle = this.elements.historyPreviewTitle;
        const previewContent = this.elements.historyPreviewContent;
        if (!list || !previewTitle || !previewContent) return;

        const filename = this.elements.historyFileSelect?.value || this.state.currentFile;
        if (!filename) {
            list.innerHTML = '<div class="outline-empty">Select a file</div>';
            previewTitle.textContent = 'Select a version to preview';
            previewContent.textContent = '';
            previewContent.classList.remove('diff-view');
            if (this.elements.historyRestoreVersion) this.elements.historyRestoreVersion.disabled = true;
            if (this.elements.historyCopyVersion) this.elements.historyCopyVersion.disabled = true;
            return;
        }

        if (this.state.historyFileSelected !== filename) {
            this.state.selectedVersionId = null;
            this.state.historyFileSelected = filename;
        }

        const versions = this.state.versionHistory.filter(v => v.filename === filename);
        if (versions.length === 0) {
            list.innerHTML = '<div class="outline-empty">No versions yet</div>';
            previewTitle.textContent = 'Select a version to preview';
            previewContent.textContent = '';
            previewContent.classList.remove('diff-view');
            if (this.elements.historyRestoreVersion) this.elements.historyRestoreVersion.disabled = true;
            if (this.elements.historyCopyVersion) this.elements.historyCopyVersion.disabled = true;
            return;
        }

        if (this.state.selectedVersionId && !versions.some(v => v.id === this.state.selectedVersionId)) {
            this.state.selectedVersionId = null;
        }

        list.innerHTML = versions.map(v => {
            const date = new Date(v.timestamp);
            const label = date.toLocaleString();
            const sourceLabel = v.source === 'auto' ? 'Auto' : 'Manual';
            return `
                <div class="history-version-item ${this.state.selectedVersionId === v.id ? 'active' : ''}" data-id="${v.id}">
                    <div class="version-title">${label}</div>
                    <div class="version-meta">${sourceLabel}</div>
                </div>
            `;
        }).join('');

        list.querySelectorAll('.history-version-item').forEach(item => {
            item.addEventListener('click', () => {
                const id = parseInt(item.dataset.id, 10);
                this.selectVersionPreview(id);
            });
        });

        const hasSelection = !!this.state.selectedVersionId;
        if (!hasSelection) {
            previewTitle.textContent = 'Select a version to preview';
            previewContent.textContent = '';
            previewContent.classList.remove('diff-view');
        }
        if (this.elements.historyRestoreVersion) this.elements.historyRestoreVersion.disabled = !hasSelection;
        if (this.elements.historyCopyVersion) this.elements.historyCopyVersion.disabled = !hasSelection;
    },

    selectVersionPreview(id) {
        const entry = this.state.versionHistory.find(v => v.id === id);
        if (!entry) return;

        this.state.selectedVersionId = id;
        const date = new Date(entry.timestamp);
        const versions = this.state.versionHistory.filter(v => v.filename === entry.filename);
        const index = versions.findIndex(v => v.id === entry.id);
        const baseEntry = index >= 0 ? versions[index + 1] : null;
        const baseLabel = baseEntry ? 'diff vs previous' : 'diff vs current';

        if (this.elements.historyPreviewTitle) {
            this.elements.historyPreviewTitle.textContent = `${entry.filename} - ${date.toLocaleString()} (${baseLabel})`;
        }
        if (this.elements.historyPreviewContent) {
            const baseContent = baseEntry ? (baseEntry.content || '') : this.getDiffBaseContent(entry.filename);
            const diffHtml = this.renderDiffPreview(baseContent, entry.content || '');
            this.elements.historyPreviewContent.innerHTML = diffHtml;
            this.elements.historyPreviewContent.classList.add('diff-view');
        }
        if (this.elements.historyRestoreVersion) this.elements.historyRestoreVersion.disabled = false;
        if (this.elements.historyCopyVersion) this.elements.historyCopyVersion.disabled = false;
        this.renderVersionHistory();
    },

    getDiffBaseContent(filename) {
        if (this.state.currentFile === filename && this.elements.codeEditor) {
            return this.elements.codeEditor.value || '';
        }
        if (this.state.fileContents[filename]) {
            return this.state.fileContents[filename];
        }
        if (this.isLocalProject()) {
            return ProjectManager.getLocalFileContent(filename) || '';
        }
        const latest = this.state.versionHistory.find(v => v.filename === filename);
        return latest?.content || '';
    },

    renderDiffPreview(oldText, newText) {
        const oldLines = (oldText || '').split('\n');
        const newLines = (newText || '').split('\n');
        const diff = this.computeLineDiff(oldLines, newLines);

        return diff.map((item) => {
            const safeLine = item.line === '' ? '&nbsp;' : this.escapeHtml(item.line);
            const prefix = item.type === 'add' ? '+' : item.type === 'del' ? '-' : ' ';
            const cls = item.type === 'add' ? 'diff-add' : item.type === 'del' ? 'diff-del' : 'diff-eq';
            return `<div class="diff-line ${cls}"><span class="diff-prefix">${prefix}</span><span class="diff-text">${safeLine}</span></div>`;
        }).join('');
    },

    computeLineDiff(oldLines, newLines) {
        const n = oldLines.length;
        const m = newLines.length;
        const max = n + m;
        const size = 2 * max + 1;
        let v = new Array(size).fill(0);
        const trace = [];

        for (let d = 0; d <= max; d += 1) {
            trace.push(v.slice());
            for (let k = -d; k <= d; k += 2) {
                const idx = k + max;
                let x;
                if (k === -d || (k !== d && v[idx - 1] < v[idx + 1])) {
                    x = v[idx + 1];
                } else {
                    x = v[idx - 1] + 1;
                }
                let y = x - k;
                while (x < n && y < m && oldLines[x] === newLines[y]) {
                    x += 1;
                    y += 1;
                }
                v[idx] = x;
                if (x >= n && y >= m) {
                    return this.backtrackDiff(trace, oldLines, newLines, max, x, y, d);
                }
            }
        }
        return [];
    },

    backtrackDiff(trace, oldLines, newLines, max, x, y, d) {
        const diff = [];
        for (let depth = d; depth >= 0; depth -= 1) {
            const v = trace[depth];
            const k = x - y;
            const idx = k + max;
            let prevK;
            if (k === -depth || (k !== depth && v[idx - 1] < v[idx + 1])) {
                prevK = k + 1;
            } else {
                prevK = k - 1;
            }
            const prevX = v[prevK + max];
            const prevY = prevX - prevK;

            while (x > prevX && y > prevY) {
                diff.push({ type: 'eq', line: oldLines[x - 1] });
                x -= 1;
                y -= 1;
            }
            if (depth === 0) break;
            if (x === prevX) {
                diff.push({ type: 'add', line: newLines[y - 1] });
                y -= 1;
            } else {
                diff.push({ type: 'del', line: oldLines[x - 1] });
                x -= 1;
            }
        }
        diff.reverse();
        return diff;
    },

    restoreSelectedVersion() {
        if (!this.state.selectedVersionId) {
            showToast('Select a version to restore');
            return;
        }

        const entry = this.state.versionHistory.find(v => v.id === this.state.selectedVersionId);
        if (!entry) return;

        if (!confirm('Restore this version? Current edits will be replaced.')) return;
        const content = entry.content || '';
        this.state.currentFile = entry.filename;
        this.elements.codeEditor.value = content;
        this.elements.codeEditor.disabled = false;
        this.elements.btnSave.disabled = false;
        this.elements.btnDelete.style.display = 'inline-block';
        this.state.fileContents[entry.filename] = content;
        this.state.dirtyFiles[entry.filename] = true;
        this.updateCurrentFileLabel();
        this.updateLineNumbers();
        this.updateEditorStats();
        this.buildOutlineFromContent(content);
        this.renderComments();
        this.renderFileList();
        showToast('Version loaded');
    },

    copySelectedVersion() {
        if (!this.state.selectedVersionId) {
            showToast('Select a version to copy');
            return;
        }

        const entry = this.state.versionHistory.find(v => v.id === this.state.selectedVersionId);
        if (!entry) return;

        const content = entry.content || '';
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(content).then(() => {
                showToast('Version copied');
            }).catch(() => {
                prompt('Copy version content:', content);
            });
        } else {
            prompt('Copy version content:', content);
        }
    },

    toggleCompileLog() {
        if (!this.elements.compileLogPanel) return;
        const isOpen = this.elements.compileLogPanel.style.display === 'flex';
        if (isOpen) {
            this.hideCompileLog();
        } else {
            this.elements.compileLogPanel.style.display = 'flex';
            this.renderCompileLog();
        }
    },

    hideCompileLog() {
        if (!this.elements.compileLogPanel) return;
        this.elements.compileLogPanel.style.display = 'none';
    },

    summarizeCompileLog(logText) {
        const lines = (logText || '').split('\n');
        let errors = 0;
        let warnings = 0;
        lines.forEach(line => {
            if (line.startsWith('!') || /error/i.test(line)) errors += 1;
            if (/warning/i.test(line)) warnings += 1;
        });
        return { errors, warnings, lines: lines.length };
    },

    renderCompileLog() {
        const log = this.state.lastCompileLog || '';
        const filter = this.elements.compileLogFilter?.value || 'all';
        const contentEl = this.elements.compileLogContent;
        const metaEl = this.elements.compileLogMeta;
        if (!contentEl || !metaEl) return;

        if (!log) {
            contentEl.textContent = '';
            metaEl.textContent = 'No log loaded';
            return;
        }

        const summary = this.summarizeCompileLog(log);
        this.state.lastCompileSummary = { errors: summary.errors, warnings: summary.warnings };
        metaEl.textContent = `Errors: ${summary.errors} | Warnings: ${summary.warnings} | Lines: ${summary.lines}`;

        const lines = log.split('\n');
        const rendered = lines.map(line => {
            const isError = line.startsWith('!') || /error/i.test(line);
            const isWarning = /warning/i.test(line);
            if (filter === 'errors' && !isError) return null;
            if (filter === 'warnings' && !isWarning) return null;
            const cls = isError ? 'error' : (isWarning ? 'warning' : '');
            const safe = escapeHtml(line);
            return cls ? `<span class="compile-log-line ${cls}">${safe}</span>` : safe;
        }).filter(Boolean);

        contentEl.innerHTML = rendered.join('\n');
    },

    copyCompileLog() {
        const log = this.state.lastCompileLog || '';
        if (!log) {
            showToast('No log to copy');
            return;
        }

        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(log).then(() => {
                showToast('Log copied');
            }).catch(() => {
                prompt('Copy compile log:', log);
            });
        } else {
            prompt('Copy compile log:', log);
        }
    },

    scheduleAutoSave() {
        if (!this.state.autoSaveEnabled || !this.state.currentFile) return;
        if (!this.isDirty(this.state.currentFile)) return;

        if (this.state.autoSaveTimer) {
            clearTimeout(this.state.autoSaveTimer);
        }

        this.state.autoSaveTimer = setTimeout(() => {
            const now = Date.now();
            if (now - this.state.lastAutoSaveAt < 2000) return;
            this.state.lastAutoSaveAt = now;
            this.saveCurrent({ silent: true, recordHistory: false, recordVersion: true, source: 'auto' });
        }, 1200);
    },

    async saveCurrent(options = {}) {
        if (!this.state.currentFile) return;
        if (this.state.editorMode === 'visual') {
            this.syncVisualToCode();
        }

        const {
            silent = false,
            recordHistory = true,
            recordVersion = true,
            source = 'manual',
            skipAutoCompile = false
        } = options;

        const content = this.elements.codeEditor.value;
        if (!silent) {
            this.elements.btnSave.innerHTML = '<span class="spinner-small"></span> SAVING...';
        }

        try {
            if (this.isLocalProject()) {
                const ok = ProjectManager.saveLocalFile(this.state.currentFile, content);
                if (!ok) throw new Error('Save failed');
            } else {
                const res = await fetch(this.withProjectParam(`${API_BASE}/api/project/save`), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        filename: this.state.currentFile,
                        content: content
                    })
                });
                if (!res.ok) throw new Error('Save failed');
            }

            if (this.isLocalProject()) {
                this.state.files = ProjectManager.getLocalFilesList();
            }

            if (!silent) {
                this.elements.btnSave.innerHTML = '<i class="bi bi-check"></i> SAVED';
            }
            if (recordHistory) {
                const label = source === 'auto' ? 'Auto saved file' : 'Saved file';
                this.saveToHistory(label, this.state.currentFile);
            }
            if (recordVersion) {
                this.saveVersionSnapshot(this.state.currentFile, content, source);
            }
            this.state.fileContents[this.state.currentFile] = content;
            this.state.dirtyFiles[this.state.currentFile] = false;
            this.updateCurrentFileLabel();
            this.renderFileList();
            if (this.getActiveProject()?.id) {
                ProjectManager.touchProject(this.getActiveProject().id);
            }
            setTimeout(() => {
                if (!silent) {
                    this.elements.btnSave.innerHTML = '<i class="bi bi-floppy"></i> SAVE';
                }
            }, 2000);

            if (silent) {
                showToast('Auto-saved');
            }

            if (!skipAutoCompile && this.state.autoCompileEnabled && this.state.currentFile.endsWith('.tex')) {
                if (this.isLocalProject()) {
                    showToast('Auto compile needs a server project');
                } else {
                    const now = Date.now();
                    if (now - this.state.lastCompileAt > 2500) {
                        this.state.lastCompileAt = now;
                        this.compileCurrent();
                    }
                }
            }
            return true;
        } catch (e) {
            console.error(e);
            alert("Failed to save file");
            if (!silent) {
                this.elements.btnSave.innerHTML = '<i class="bi bi-floppy"></i> SAVE';
            }
            return false;
        }
    },

    async saveAndCompile() {
        const saved = await this.saveCurrent({ skipAutoCompile: true });
        if (saved && this.state.currentFile?.endsWith('.tex')) {
            if (this.isLocalProject()) {
                showToast('Compile requires a server project');
                return;
            }
            showToast('Compiling...');
            await this.compileCurrent();
        }
    },

    async handleUpload(e) {
        const file = e.target.files[0];
        if (!file) return;

        if (this.isLocalProject()) {
            this.elements.btnUpload.innerHTML = '<span class="spinner-small"></span>';
            const isTextFile = /\.(tex|bib|txt|md)$/i.test(file.name);
            if (!isTextFile) {
                ProjectManager.saveLocalFile(file.name, `% Binary file uploaded: ${file.name}\n`);
                this.saveToHistory('Uploaded file', file.name);
                await this.loadFiles();
                this.elements.fileUploadInput.value = '';
                this.elements.btnUpload.innerHTML = '<i class="bi bi-upload"></i>';
                showToast('Binary file stored as placeholder');
                return;
            }

            const reader = new FileReader();
            reader.onload = async () => {
                const content = typeof reader.result === 'string' ? reader.result : '';
                ProjectManager.saveLocalFile(file.name, content);
                this.saveToHistory('Uploaded file', file.name);
                await this.loadFiles();
                this.elements.fileUploadInput.value = '';
                this.elements.btnUpload.innerHTML = '<i class="bi bi-upload"></i>';
                showToast('File uploaded');
            };
            reader.onerror = () => {
                console.error('Upload error');
                showToast('Upload error');
                this.elements.btnUpload.innerHTML = '<i class="bi bi-upload"></i>';
            };
            reader.readAsText(file);
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            this.elements.btnUpload.innerHTML = '<span class="spinner-small"></span>';
            const res = await fetch(this.withProjectParam(`${API_BASE}/api/project/upload`), {
                method: 'POST',
                body: formData
            });

            if (res.ok) {
                await this.loadFiles();
                this.elements.fileUploadInput.value = ''; // Reset
            } else {
                alert("Upload failed");
            }
        } catch (e) {
            console.error(e);
            alert("Upload error");
        } finally {
            this.elements.btnUpload.innerHTML = '<i class="bi bi-upload"></i>';
        }
    },

    async deleteCurrent() {
        if (!this.state.currentFile) return;
        if (this.isDirty(this.state.currentFile)) {
            if (!confirm(`"${this.state.currentFile}" has unsaved changes. Delete anyway?`)) return;
        }
        if (!confirm(`Are you sure you want to delete ${this.state.currentFile}?`)) return;

        try {
            if (this.isLocalProject()) {
                const ok = ProjectManager.deleteLocalFile(this.state.currentFile);
                if (!ok) throw new Error('Delete failed');
                const deleted = this.state.currentFile;
                this.state.currentFile = null;
                this.elements.currentFileName.textContent = "Select a file...";
                this.elements.codeEditor.value = "";
                this.elements.codeEditor.disabled = true;
                this.elements.btnSave.disabled = true;
                this.elements.btnDelete.style.display = 'none';
                delete this.state.fileContents[deleted];
                delete this.state.dirtyFiles[deleted];
                this.state.commentSelection = null;
                this.updateCommentContext();
                this.updateEditorStats();
                this.state.comments = this.state.comments.filter(comment => comment.filename !== deleted);
                this.saveComments();
                this.state.versionHistory = this.state.versionHistory.filter(entry => entry.filename !== deleted);
                try {
                    localStorage.setItem(this.getProjectStorageKey('paperreader_versions'), JSON.stringify(this.state.versionHistory));
                } catch (e) {
                    console.warn('Failed to update version history after delete');
                }
                this.renderComments();
                this.saveToHistory('Deleted file', deleted);
                await this.loadFiles();
                return;
            }

            const res = await fetch(this.withProjectParam(`${API_BASE}/api/project/file/${this.state.currentFile}`), {
                method: 'DELETE'
            });

            if (res.ok) {
                const deleted = this.state.currentFile;
                this.state.currentFile = null;
                this.elements.currentFileName.textContent = "Select a file...";
                this.elements.codeEditor.value = "";
                this.elements.codeEditor.disabled = true;
                this.elements.btnSave.disabled = true;
                this.elements.btnDelete.style.display = 'none';
                delete this.state.fileContents[deleted];
                delete this.state.dirtyFiles[deleted];
                this.state.commentSelection = null;
                this.updateCommentContext();
                this.updateEditorStats();
                this.state.comments = this.state.comments.filter(comment => comment.filename !== deleted);
                this.saveComments();
                this.state.versionHistory = this.state.versionHistory.filter(entry => entry.filename !== deleted);
                try {
                    localStorage.setItem(this.getProjectStorageKey('paperreader_versions'), JSON.stringify(this.state.versionHistory));
                } catch (e) {
                    console.warn('Failed to update version history after delete');
                }
                this.renderComments();
                this.saveToHistory('Deleted file', deleted);
                await this.loadFiles();
            } else {
                alert("Delete failed (File likely protected)");
            }
        } catch (e) {
            console.error(e);
            alert("Delete error");
        }
    },

    async loadCitationMap() {
        if (this.isLocalProject()) {
            this.state.citationMap = {};
            this.state.citationKeys = [];
            return;
        }
        try {
            const res = await fetch(this.withProjectParam(`${API_BASE}/api/citations/map`));
            const data = await res.json();
            this.state.citationMap = data.mapping || {};
            this.state.citationKeys = Object.keys(this.state.citationMap).sort();
        } catch (e) {
            console.error("Failed to load citation map", e);
            this.state.citationKeys = [];
        }
    },

    async compileCurrent() {
        if (this.isLocalProject()) {
            showToast('Compile requires a server project');
            return;
        }
        if (this.state.editorMode === 'visual') {
            this.syncVisualToCode();
        }

        // Always compile the main tex file usually, or the current one?
        // Let's assume lam_main_latest.tex is the main one or try to identify it.
        // For this demo, let's just compile the currently selected file if it is .tex,
        // otherwise default to 'lam_main_latest.tex' if available.

        let target = this.state.currentFile;
        // fallback
        if (!target || !target.endsWith('.tex')) {
            const main = this.state.files.find(f => f.name.includes('main') && f.name.endsWith('.tex'));
            target = main ? main.name : null;
        }

        if (!target) {
            alert("Please select a .tex file to compile.");
            return;
        }

        // Add loading state to compile button
        this.elements.btnCompile.classList.add('loading');
        this.elements.btnCompile.innerHTML = '<i class="bi bi-arrow-repeat"></i> <span class="btn-text">COMPILING...</span>';

        this.state.lastCompileAt = Date.now();
        this.elements.compileStatus.style.display = 'flex';
        this.elements.compileStatus.innerHTML = '<span class="spinner-small"></span> Using pdflatex...';

        try {
            const res = await fetch(this.withProjectParam(`${API_BASE}/api/compile`), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: target })
            });
            const result = await res.json();

            this.state.lastCompileLog = result.log || '';
            if (this.elements.compileLogFilter) {
                this.elements.compileLogFilter.value = 'all';
            }
            this.renderCompileLog();

            if (result.success) {
                this.elements.compileStatus.innerHTML = '<i class="bi bi-check-circle-fill"></i> Compilation Success';
                this.elements.compileStatus.style.color = 'var(--success)';
                this.elements.compileStatus.style.borderColor = 'var(--success)';
                setTimeout(() => {
                    this.elements.compileStatus.style.display = 'none';
                }, 3000);

                // Show PDF
                const pdfFilename = result.pdf_path.split(/[\\/]/).pop();
                this.state.lastCompiledPdfUrl = this.getCompiledPdfUrl(pdfFilename);
                this.saveLastCompiledPdf(pdfFilename);
                this.elements.btnDownloadCompiled.disabled = false;
                this.loadPDF(pdfFilename); // filename on server, resolved via API
                this.saveToHistory('Compiled document', target);

                // Success animation on button
                this.elements.btnCompile.classList.remove('loading');
                this.elements.btnCompile.classList.add('success');
                this.elements.btnCompile.innerHTML = '<i class="bi bi-check-lg"></i> <span class="btn-text">SUCCESS</span>';
                setTimeout(() => {
                    this.elements.btnCompile.classList.remove('success');
                    this.elements.btnCompile.innerHTML = '<i class="bi bi-play-fill"></i> <span class="btn-text">RECOMPILE</span>';
                }, 2000);
            } else {
                this.elements.compileStatus.innerHTML = '<i class="bi bi-x-circle-fill"></i> Detailed Compilation Log in Console';
                this.elements.compileStatus.style.color = 'var(--error)';
                this.elements.compileStatus.style.borderColor = 'var(--error)';
                console.error("Compilation Log:", result.log);
                alert("Compilation failed. Check console for logs.");
                this.state.lastCompiledPdfUrl = null;
                this.elements.btnDownloadCompiled.disabled = true;
                if (this.elements.compileLogFilter) {
                    this.elements.compileLogFilter.value = 'errors';
                }
                if (this.elements.compileLogPanel) {
                    this.elements.compileLogPanel.style.display = 'flex';
                }
                this.renderCompileLog();

                // Reset button
                this.elements.btnCompile.classList.remove('loading');
                this.elements.btnCompile.innerHTML = '<i class="bi bi-play-fill"></i> <span class="btn-text">RECOMPILE</span>';
            }
        } catch (e) {
            this.elements.compileStatus.innerHTML = 'Error';
            console.error(e);
            this.state.lastCompiledPdfUrl = null;
            this.elements.btnDownloadCompiled.disabled = true;
            this.state.lastCompileLog = String(e);
            this.renderCompileLog();
            this.elements.compileStatus.style.color = 'var(--error)';
            this.elements.compileStatus.style.borderColor = 'var(--error)';

            // Reset button
            this.elements.btnCompile.classList.remove('loading');
            this.elements.btnCompile.innerHTML = '<i class="bi bi-play-fill"></i> <span class="btn-text">RECOMPILE</span>';
        }
    },

    downloadCompiled() {
        if (this.isLocalProject()) {
            showToast('Compile requires a server project');
            return;
        }
        if (!this.state.lastCompiledPdfUrl) {
            showToast('Compile first to download the PDF');
            return;
        }

        const link = document.createElement('a');
        link.href = this.state.lastCompiledPdfUrl;
        link.download = '';
        link.click();
        showToast('Downloading compiled PDF...');
    },

    async loadPDF(pdfPath) {
        if (this.isLocalProject()) {
            showToast('PDF preview requires a server project');
            return;
        }

        // The path returned is absolute server path. We need to fetch via API.
        // Endpoint: /api/project/file/{filename}
        const filename = pdfPath.split(/[\\/]/).pop();
        const url = this.withProjectParam(`${API_BASE}/api/project/file/${filename}`);

        this.elements.pdfContainer.innerHTML = ''; // Clear
        this.state.currentPreviewPage = 1;

        try {
            const loadingTask = pdfjsLib.getDocument(url);
            this.state.pdfDoc = await loadingTask.promise;

            // Render all pages
            for (let pageNum = 1; pageNum <= this.state.pdfDoc.numPages; pageNum++) {
                await this.renderPage(pageNum);
            }
            this.updatePageCount();
            this.setupPdfScrollHandler();
        } catch (e) {
            console.error("PDF Render Error", e);
            this.elements.pdfContainer.innerHTML = '<div style="color:red; padding:20px;">Failed to load PDF</div>';
        }
    },

    updatePageCount() {
        const total = this.state.pdfDoc?.numPages || 0;
        const current = this.state.currentPreviewPage || 1;
        if (this.elements.pageCountDisplay) {
            this.elements.pageCountDisplay.textContent = total ? `Page ${current} of ${total}` : 'Page 0 of 0';
        }
    },

    setupPdfScrollHandler() {
        const container = this.elements.pdfContainer;
        if (!container) return;

        if (this.state.pdfScrollHandler) {
            container.removeEventListener('scroll', this.state.pdfScrollHandler);
        }

        const handler = () => {
            const pages = Array.from(container.querySelectorAll('.pdf-page'));
            if (!pages.length) return;

            const containerTop = container.getBoundingClientRect().top;
            let closest = pages[0];
            let closestOffset = Math.abs(closest.getBoundingClientRect().top - containerTop);

            pages.forEach(page => {
                const offset = Math.abs(page.getBoundingClientRect().top - containerTop);
                if (offset < closestOffset) {
                    closest = page;
                    closestOffset = offset;
                }
            });

            const pageNum = parseInt(closest.dataset.pageNumber, 10) || 1;
            if (pageNum !== this.state.currentPreviewPage) {
                this.state.currentPreviewPage = pageNum;
                this.updatePageCount();
            }
        };

        this.state.pdfScrollHandler = handler;
        container.addEventListener('scroll', handler, { passive: true });
        handler();
    },

    async renderPage(pageNum) {
        const page = await this.state.pdfDoc.getPage(pageNum);

        const outputScale = window.devicePixelRatio || 1;
        const scale = this.state.pdfScale;
        const viewport = page.getViewport({ scale });

        // Wrapper for page
        const pageDiv = document.createElement('div');
        pageDiv.className = 'pdf-page';
        pageDiv.dataset.pageNumber = pageNum;
        pageDiv.style.position = 'relative';
        pageDiv.style.marginBottom = '20px';
        pageDiv.style.width = `${viewport.width}px`;
        pageDiv.style.height = `${viewport.height}px`;
        this.elements.pdfContainer.appendChild(pageDiv);

        // Canvas
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = Math.floor(viewport.width * outputScale);
        canvas.height = Math.floor(viewport.height * outputScale);
        canvas.style.width = `${viewport.width}px`;
        canvas.style.height = `${viewport.height}px`;
        canvas.className = 'pdf-canvas';
        pageDiv.appendChild(canvas);

        const renderContext = {
            canvasContext: context,
            viewport: viewport,
            transform: [outputScale, 0, 0, outputScale, 0, 0]
        };

        await page.render(renderContext).promise;

        // Text Layer for selection
        const textLayerDiv = document.createElement('div');
        textLayerDiv.className = 'textLayer';
        textLayerDiv.style.width = `${viewport.width}px`;
        textLayerDiv.style.height = `${viewport.height}px`;
        pageDiv.appendChild(textLayerDiv);

        const textContent = await page.getTextContent();
        await pdfjsLib.renderTextLayer({
            textContent: textContent,
            container: textLayerDiv,
            viewport: viewport,
            textDivs: []
        }).promise;

        // SyncTex / Click handler
        // Double click on text layer to go to source
        pageDiv.addEventListener('dblclick', (e) => {
            const rect = pageDiv.getBoundingClientRect();
            const textSpan = e.target.closest('.textLayer span');
            let x = e.clientX - rect.left;
            let y = e.clientY - rect.top;

            if (textSpan) {
                const spanRect = textSpan.getBoundingClientRect();
                x = spanRect.left + spanRect.width / 2 - rect.left;
                y = spanRect.top + spanRect.height / 2 - rect.top;
            }

            // Convert viewport coordinates back to PDF points (unscaled)
            // viewport.convertToPdfPoint(x, y) returns [pdfX, pdfY]
            // where pdfY is from bottom
            const [pdfX, pdfY] = viewport.convertToPdfPoint(x, y);

            // SyncTex expects top-down coordinates usually? 
            // Or does it? Let's check. 
            // Actually, we can just send the coordinates to the backend and let it handle.
            // But we need to know what 'synctex edit' expects.
            // It expects: page:x:y:file, where y is from top.

            const pdfY_fromTop = viewport.viewBox[3] - pdfY;

            this.handleSyncTex(pageNum, pdfX, pdfY_fromTop);
        });

        // Annotation Layer
        const annotationLayerDiv = document.createElement('div');
        annotationLayerDiv.className = 'annotationLayer';
        annotationLayerDiv.style.width = `${viewport.width}px`;
        annotationLayerDiv.style.height = `${viewport.height}px`;
        pageDiv.appendChild(annotationLayerDiv);

        const annotations = await page.getAnnotations();
        annotations.forEach(annotation => {
            if (annotation.subtype === 'Link') {
                const rect = viewport.convertToViewportRectangle(annotation.rect);
                const left = Math.min(rect[0], rect[2]);
                const top = Math.min(rect[1], rect[3]);
                const width = Math.abs(rect[0] - rect[2]);
                const height = Math.abs(rect[1] - rect[3]);

                const linkDiv = document.createElement('div');
                linkDiv.className = 'pdf-link-annotation';
                linkDiv.style.left = `${left}px`;
                linkDiv.style.top = `${top}px`;
                linkDiv.style.width = `${width}px`;
                linkDiv.style.height = `${height}px`;

                if (annotation.url) {
                    linkDiv.onclick = () => window.open(annotation.url, '_blank');
                } else if (annotation.dest) {
                    linkDiv.onclick = (e) => this.handleLinkClick(e, annotation.dest);
                }

                annotationLayerDiv.appendChild(linkDiv);
            }
        });
    },

    async handleSyncTex(page, x, y) {
        // Need to find which PDF we are viewing. 
        // lastCompiledPdfUrl is something like http://.../api/project/file/lam_main_latest.pdf
        if (!this.state.lastCompiledPdfUrl) return;
        if (this.isLocalProject()) {
            showToast('SyncTeX requires a server project');
            return;
        }

        const pdfName = this.state.lastCompiledPdfUrl.split('/').pop();

        try {
            const res = await fetch(this.withProjectParam(`${API_BASE}/api/synctex`), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename: pdfName,
                    page: page,
                    x: x,
                    y: y
                })
            });
            const result = await res.json();

            if (result.success) {
                // Focus the file and line
                const file = this.state.files.find(f => f.name === result.file);
                if (file) {
                    await this.selectFile(file);
                    this.gotoLine(result.line);
                    showToast(`Going to ${result.file}:${result.line}`);
                }
            } else {
                console.warn("SyncTex failed", result);
            }
        } catch (e) {
            console.error("SyncTex error", e);
        }
    },

    gotoLine(lineNum) {
        const editor = this.elements.codeEditor;
        if (!editor) return;

        const lines = editor.value.split('\n');
        let pos = 0;
        for (let i = 0; i < lineNum - 1 && i < lines.length; i++) {
            pos += lines[i].length + 1; // +1 for newline
        }

        editor.focus();
        // Highlight the line
        const lineEnd = pos + (lines[lineNum - 1]?.length || 0);
        editor.setSelectionRange(pos, lineEnd);

        // Scroll to line
        const lineHeight = parseFloat(getComputedStyle(editor).lineHeight) || 20;
        const targetTop = (lineNum - 1) * lineHeight;
        const offset = Math.max(0, targetTop - editor.clientHeight / 2);
        editor.scrollTop = offset;

        this.updateLineNumbers();
        this.updateCurrentLine();
    },

    handleLinkClick(e, dest) {
        e.stopPropagation();
        // Dest is usually a named destination string like "cite.yih2016webqsp"
        let key = null;

        if (typeof dest === 'string') {
            key = dest;
        } else if (Array.isArray(dest) && dest.length > 0) {
            // pdf.js sometimes returns [{num, gen, name}] or similar structures
            const maybeName = dest[0]?.name || dest[0];
            if (typeof maybeName === 'string') {
                key = maybeName;
            }
        }

        if (!key) {
            console.log('Unsupported citation destination format:', dest);
            return;
        }

        if (key.startsWith('cite.')) key = key.substring(5);

        if (this.state.citationMap[key]) {
            const paperId = this.state.citationMap[key];
            openPaper(paperId); // Global function
        } else {
            console.log('Citation not found in map:', key);
            showToast('Citation not linked to a PDF');
        }
    },

    // ==========================================
    // FIND IN TEX FUNCTIONALITY
    // ==========================================

    toggleFindBar() {
        const findBar = this.elements.findBar;
        if (!findBar) return;

        if (findBar.style.display === 'none') {
            findBar.style.display = 'block';
            this.elements.findInput?.focus();
        } else {
            this.closeFindBar();
        }
    },

    closeFindBar() {
        if (this.elements.findBar) {
            this.elements.findBar.style.display = 'none';
        }
        this.clearFindHighlights();
    },

    performFind() {
        const query = this.elements.findInput?.value?.toLowerCase() || '';
        const editor = this.elements.codeEditor;
        if (!editor || !query) {
            this.elements.findCount.textContent = '0 results';
            this.state.findMatches = [];
            return;
        }

        const content = editor.value.toLowerCase();
        const matches = [];
        let pos = 0;

        while ((pos = content.indexOf(query, pos)) !== -1) {
            matches.push(pos);
            pos += query.length;
        }

        this.state.findMatches = matches;
        this.state.currentFindIndex = matches.length > 0 ? 0 : -1;
        this.elements.findCount.textContent = `${matches.length} result${matches.length !== 1 ? 's' : ''}`;

        if (matches.length > 0 && document.activeElement !== this.elements.findInput) {
            this.highlightCurrentMatch(true);
        }
    },

    findNext() {
        if (this.state.findMatches.length === 0) return;
        this.state.currentFindIndex = (this.state.currentFindIndex + 1) % this.state.findMatches.length;
        this.highlightCurrentMatch(true);
    },

    findPrev() {
        if (this.state.findMatches.length === 0) return;
        this.state.currentFindIndex = this.state.currentFindIndex <= 0
            ? this.state.findMatches.length - 1
            : this.state.currentFindIndex - 1;
        this.highlightCurrentMatch(true);
    },

    highlightCurrentMatch(forceFocus = false) {
        const editor = this.elements.codeEditor;
        const query = this.elements.findInput?.value || '';
        if (!editor || this.state.currentFindIndex < 0) return;

        const pos = this.state.findMatches[this.state.currentFindIndex];
        const findInput = this.elements.findInput;
        const keepFindFocus = document.activeElement === findInput;
        if (forceFocus && !keepFindFocus) {
            editor.focus();
        }
        editor.setSelectionRange(pos, pos + query.length);

        // Scroll to selection
        const lineHeight = parseFloat(getComputedStyle(editor).lineHeight) || 20;
        const linesBeforeMatch = editor.value.substring(0, pos).split('\n').length;
        editor.scrollTop = (linesBeforeMatch - 5) * lineHeight;

        this.elements.findCount.textContent = `${this.state.currentFindIndex + 1}/${this.state.findMatches.length}`;

        if (keepFindFocus && findInput) {
            requestAnimationFrame(() => {
                findInput.focus({ preventScroll: true });
            });
        }
    },

    clearFindHighlights() {
        this.state.findMatches = [];
        this.state.currentFindIndex = -1;
    },

    // ==========================================
    // FIND IN PDF FUNCTIONALITY
    // ==========================================

    togglePdfFindBar() {
        const findBar = this.elements.pdfFindBar;
        if (!findBar) return;

        if (findBar.style.display === 'none') {
            findBar.style.display = 'block';
            this.elements.pdfFindInput?.focus();
        } else {
            this.closePdfFindBar();
        }
    },

    closePdfFindBar() {
        if (this.elements.pdfFindBar) {
            this.elements.pdfFindBar.style.display = 'none';
        }
    },

    async performPdfFind() {
        const query = this.elements.pdfFindInput?.value || '';
        if (!this.state.pdfDoc || !query) {
            this.elements.pdfFindCount.textContent = '0 results';
            return;
        }

        // PDF.js text search is complex - simplified version
        let totalMatches = 0;
        for (let i = 1; i <= this.state.pdfDoc.numPages; i++) {
            const page = await this.state.pdfDoc.getPage(i);
            const textContent = await page.getTextContent();
            const text = textContent.items.map(item => item.str).join(' ').toLowerCase();
            const safeQuery = query.toLowerCase().replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const matches = (text.match(new RegExp(safeQuery, 'g')) || []).length;
            totalMatches += matches;
        }

        this.elements.pdfFindCount.textContent = `${totalMatches} result${totalMatches !== 1 ? 's' : ''}`;
        if (totalMatches > 0) {
            showToast(`Found ${totalMatches} matches in PDF`);
        }
    },

    pdfFindNext() {
        showToast('Navigate to next match');
    },

    pdfFindPrev() {
        showToast('Navigate to previous match');
    },

    // ==========================================
    // EDIT HISTORY FUNCTIONALITY
    // ==========================================

    loadHistory() {
        try {
            const saved = localStorage.getItem(this.getProjectStorageKey('paperreader_history'));
            this.state.editHistory = saved ? JSON.parse(saved) : [];
        } catch (e) {
            this.state.editHistory = [];
        }
    },

    saveToHistory(action, filename, content = null) {
        const entry = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            action,
            filename,
            content: content ? content.substring(0, 500) : null // Store preview only
        };

        this.state.editHistory.unshift(entry);

        // Keep only last 50 entries
        if (this.state.editHistory.length > 50) {
            this.state.editHistory = this.state.editHistory.slice(0, 50);
        }

        try {
            localStorage.setItem(this.getProjectStorageKey('paperreader_history'), JSON.stringify(this.state.editHistory));
        } catch (e) {
            console.warn('Failed to save history to localStorage');
        }
    },

    showHistory() {
        if (!this.elements.historyModal) return;

        this.elements.historyModal.style.display = 'flex';
        this.renderHistory();
        this.updateHistoryFileOptions();
        this.switchHistoryTab('activity');
        this.renderVersionHistory();
    },

    closeHistory() {
        if (this.elements.historyModal) {
            this.elements.historyModal.style.display = 'none';
        }
    },

    renderHistory() {
        const container = this.elements.historyList;
        if (!container) return;

        if (this.state.editHistory.length === 0) {
            container.innerHTML = `
                <div class="history-empty">
                    <i class="bi bi-clock-history"></i>
                    <p>No edit history yet</p>
                    <p style="font-size: 0.8rem; opacity: 0.7;">Your edits will appear here</p>
                </div>
            `;
            return;
        }

        container.innerHTML = this.state.editHistory.map(entry => {
            const date = new Date(entry.timestamp);
            const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            const dateStr = date.toLocaleDateString([], { month: 'short', day: 'numeric' });

            return `
                <div class="history-item" data-id="${entry.id}">
                    <div class="history-time">${timeStr}<br><small>${dateStr}</small></div>
                    <div class="history-details">
                        <div class="history-action">${this.escapeHtml(entry.action)}</div>
                        <div class="history-file">${this.escapeHtml(entry.filename)}</div>
                    </div>
                </div>
            `;
        }).join('');
    },

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    // ==========================================
    // ZOOM FUNCTIONALITY
    // ==========================================

    zoomIn() {
        this.state.pdfScale = Math.min(this.state.pdfScale + 0.2, 3.0);
        if (this.state.lastCompiledPdfUrl) {
            this.reRenderPdf();
        }
        showToast(`Zoom: ${Math.round(this.state.pdfScale * 100)}%`);
    },

    zoomOut() {
        this.state.pdfScale = Math.max(this.state.pdfScale - 0.2, 0.5);
        if (this.state.lastCompiledPdfUrl) {
            this.reRenderPdf();
        }
        showToast(`Zoom: ${Math.round(this.state.pdfScale * 100)}%`);
    },

    async reRenderPdf() {
        if (!this.state.pdfDoc) return;

        this.elements.pdfContainer.innerHTML = '';
        for (let pageNum = 1; pageNum <= this.state.pdfDoc.numPages; pageNum++) {
            await this.renderPage(pageNum);
        }
        this.updatePageCount();
        this.setupPdfScrollHandler();
    },

    // ==========================================
    // EDITOR VIEW MODES
    // ==========================================

    loadEditorPreferences() {
        const wrapPref = localStorage.getItem('paperreader_linewrap');
        if (wrapPref !== null) {
            this.state.wrapEnabled = wrapPref === 'true';
        }
        const modePref = localStorage.getItem('paperreader_editor_mode');
        if (modePref === 'visual' || modePref === 'code') {
            this.state.editorMode = modePref;
        }
    },

    applyLineWrap() {
        const editor = this.elements.codeEditor;
        if (!editor) return;
        editor.classList.toggle('wrap', this.state.wrapEnabled);
        this.elements.codeHighlight?.classList.toggle('wrap', this.state.wrapEnabled);
        this.elements.btnLineWrap?.classList.toggle('active', this.state.wrapEnabled);
    },

    toggleLineWrap() {
        this.state.wrapEnabled = !this.state.wrapEnabled;
        localStorage.setItem('paperreader_linewrap', this.state.wrapEnabled ? 'true' : 'false');
        this.applyLineWrap();
        showToast(this.state.wrapEnabled ? 'Line wrap on' : 'Line wrap off');
    },

    applyEditorMode() {
        this.setEditorMode(this.state.editorMode, { silent: true });
    },

    setEditorMode(mode, options = {}) {
        const { silent = false } = options;
        if (mode !== 'code' && mode !== 'visual') return;

        this.state.editorMode = mode;
        localStorage.setItem('paperreader_editor_mode', mode);

        this.elements.sourceView?.classList.toggle('visual-mode', mode === 'visual');
        this.elements.btnModeCode?.classList.toggle('active', mode === 'code');
        this.elements.btnModeVisual?.classList.toggle('active', mode === 'visual');

        if (mode === 'visual') {
            this.hideLatexAutocomplete();
            this.syncCodeToVisual();
            this.setVisualEditorEnabled(!!this.state.currentFile);
            if (this.elements.codeEditor) {
                this.elements.codeEditor.disabled = true;
            }
            if (document.queryCommandSupported && document.queryCommandSupported('defaultParagraphSeparator')) {
                document.execCommand('defaultParagraphSeparator', false, 'p');
            }
        } else {
            this.syncVisualToCode();
            if (this.elements.codeEditor) {
                this.elements.codeEditor.disabled = !this.state.currentFile;
            }
        }

        if (!silent) {
            showToast(mode === 'visual' ? 'Visual editor on' : 'Code editor on');
        }
    },

    setVisualEditorEnabled(enabled) {
        if (!this.elements.visualEditor) return;
        this.elements.visualEditor.setAttribute('contenteditable', enabled ? 'true' : 'false');
        this.elements.visualEditor.classList.toggle('disabled', !enabled);
    },

    handleVisualToolbar(event) {
        const button = event.target.closest('button[data-action]');
        if (!button) return;
        const action = button.dataset.action;
        if (!action) return;

        if (action === 'bold') {
            document.execCommand('bold');
        } else if (action === 'italic') {
            document.execCommand('italic');
        } else if (action === 'underline') {
            document.execCommand('underline');
        } else if (action === 'section') {
            document.execCommand('formatBlock', false, 'h2');
        } else if (action === 'subsection') {
            document.execCommand('formatBlock', false, 'h3');
        } else if (action === 'itemize') {
            document.execCommand('insertUnorderedList');
        } else if (action === 'enumerate') {
            document.execCommand('insertOrderedList');
        }
        this.handleVisualInput();
    },

    handleVisualKeydown(event) {
        if (event.key === 'Tab') {
            event.preventDefault();
            document.execCommand('insertText', false, '    ');
        }
    },

    handleVisualInput() {
        if (this.state.visualSyncLock) return;
        if (this.state.visualSyncTimer) {
            clearTimeout(this.state.visualSyncTimer);
        }
        this.state.visualSyncTimer = setTimeout(() => {
            this.syncVisualToCode();
        }, 200);
    },

    syncCodeToVisual() {
        if (!this.elements.visualEditor || !this.elements.codeEditor) return;
        this.state.visualSyncLock = true;
        const latex = this.elements.codeEditor.value || '';
        this.elements.visualEditor.innerHTML = this.latexToVisualHtml(latex);
        this.state.visualSyncLock = false;
    },

    syncVisualToCode() {
        if (!this.elements.visualEditor || !this.elements.codeEditor) return;
        this.state.visualSyncLock = true;
        const html = this.elements.visualEditor.innerHTML || '';
        const latex = this.visualHtmlToLatex(html);
        this.elements.codeEditor.value = latex;
        this.state.visualSyncLock = false;
        this.handleEditorInput();
    },

    latexToVisualHtml(text) {
        if (!text) return '';
        const lines = text.split('\n');
        const output = [];
        let paragraph = [];
        let listMode = null;

        const flushParagraph = () => {
            if (!paragraph.length) return;
            const paragraphText = paragraph.join(' ');
            output.push(`<p>${this.formatInlineLatex(paragraphText)}</p>`);
            paragraph = [];
        };

        const openList = (type) => {
            if (listMode === type) return;
            closeList();
            listMode = type;
            output.push(type === 'ol' ? '<ol>' : '<ul>');
        };

        const closeList = () => {
            if (!listMode) return;
            output.push(listMode === 'ol' ? '</ol>' : '</ul>');
            listMode = null;
        };

        lines.forEach((line) => {
            const trimmed = line.trim();
            if (!trimmed) {
                flushParagraph();
                closeList();
                return;
            }

            const sectionMatch = trimmed.match(/^\\section\*?\{([^}]*)\}/);
            const subsectionMatch = trimmed.match(/^\\subsection\*?\{([^}]*)\}/);
            const subsubMatch = trimmed.match(/^\\subsubsection\*?\{([^}]*)\}/);
            const paragraphMatch = trimmed.match(/^\\paragraph\*?\{([^}]*)\}/);
            const beginItemize = /^\\begin\{itemize\}/.test(trimmed);
            const endItemize = /^\\end\{itemize\}/.test(trimmed);
            const beginEnum = /^\\begin\{enumerate\}/.test(trimmed);
            const endEnum = /^\\end\{enumerate\}/.test(trimmed);
            const itemMatch = trimmed.match(/^\\item\s*(.*)/);

            if (sectionMatch) {
                flushParagraph();
                closeList();
                output.push(`<h2>${this.escapeHtml(sectionMatch[1])}</h2>`);
                return;
            }
            if (subsectionMatch) {
                flushParagraph();
                closeList();
                output.push(`<h3>${this.escapeHtml(subsectionMatch[1])}</h3>`);
                return;
            }
            if (subsubMatch) {
                flushParagraph();
                closeList();
                output.push(`<h4>${this.escapeHtml(subsubMatch[1])}</h4>`);
                return;
            }
            if (paragraphMatch) {
                flushParagraph();
                closeList();
                output.push(`<h5>${this.escapeHtml(paragraphMatch[1])}</h5>`);
                return;
            }

            if (beginItemize) {
                flushParagraph();
                openList('ul');
                return;
            }
            if (endItemize) {
                flushParagraph();
                closeList();
                return;
            }
            if (beginEnum) {
                flushParagraph();
                openList('ol');
                return;
            }
            if (endEnum) {
                flushParagraph();
                closeList();
                return;
            }

            if (itemMatch) {
                flushParagraph();
                openList(listMode || 'ul');
                output.push(`<li>${this.formatInlineLatex(itemMatch[1])}</li>`);
                return;
            }

            paragraph.push(trimmed);
        });

        flushParagraph();
        closeList();
        return output.join('\n');
    },

    formatInlineLatex(text) {
        let output = this.escapeHtml(text);
        output = output.replace(/\\textbf\{([^}]*)\}/g, '<strong>$1</strong>');
        output = output.replace(/\\textit\{([^}]*)\}/g, '<em>$1</em>');
        output = output.replace(/\\emph\{([^}]*)\}/g, '<em>$1</em>');
        output = output.replace(/\\underline\{([^}]*)\}/g, '<u>$1</u>');
        output = output.replace(/\\texttt\{([^}]*)\}/g, '<code>$1</code>');
        return output;
    },

    visualHtmlToLatex(html) {
        const container = document.createElement('div');
        container.innerHTML = html;

        const serialize = (node) => {
            if (node.nodeType === Node.TEXT_NODE) {
                return node.textContent || '';
            }
            if (node.nodeType !== Node.ELEMENT_NODE) {
                return '';
            }

            const tag = node.tagName.toLowerCase();
            const children = Array.from(node.childNodes).map(serialize).join('');

            if (tag === 'strong') return `\\textbf{${children}}`;
            if (tag === 'em') return `\\emph{${children}}`;
            if (tag === 'u') return `\\underline{${children}}`;
            if (tag === 'code') return `\\texttt{${children}}`;
            if (tag === 'br') return '\n';
            if (tag === 'h2') return `\\section{${children}}\n`;
            if (tag === 'h3') return `\\subsection{${children}}\n`;
            if (tag === 'h4') return `\\subsubsection{${children}}\n`;
            if (tag === 'h5') return `\\paragraph{${children}}\n`;

            if (tag === 'ul' || tag === 'ol') {
                const env = tag === 'ul' ? 'itemize' : 'enumerate';
                const items = Array.from(node.children)
                    .filter((child) => child.tagName && child.tagName.toLowerCase() === 'li')
                    .map((child) => `\\item ${Array.from(child.childNodes).map(serialize).join('').trim()}`)
                    .join('\n');
                return `\\begin{${env}}\n${items}\n\\end{${env}}\n`;
            }

            if (tag === 'p' || tag === 'div') {
                const content = children.trim();
                return content ? `${content}\n\n` : '\n';
            }

            return children;
        };

        const bodyContent = Array.from(container.childNodes).map(serialize).join('').trim();
        return bodyContent ? `${bodyContent}\n` : '';
    },

    // ==========================================
    // CODE HIGHLIGHTING
    // ==========================================

    scheduleHighlightUpdate() {
        if (this.state.highlightTimer) {
            clearTimeout(this.state.highlightTimer);
        }
        this.state.highlightTimer = setTimeout(() => {
            this.updateCodeHighlight();
        }, 60);
    },

    updateCodeHighlight() {
        const editor = this.elements.codeEditor;
        const highlight = this.elements.codeHighlightContent;
        if (!editor || !highlight) return;
        const text = editor.value || '';
        const html = this.highlightLatex(text);
        highlight.innerHTML = html || '&nbsp;';
        this.syncHighlightScroll();
    },

    syncHighlightScroll() {
        const editor = this.elements.codeEditor;
        const highlight = this.elements.codeHighlight;
        if (!editor || !highlight) return;
        highlight.scrollTop = editor.scrollTop;
        highlight.scrollLeft = editor.scrollLeft;
    },

    highlightLatex(text) {
        const lines = (text || '').split('\n');
        return lines.map((line) => {
            const { code, comment } = this.splitLatexComment(line);
            const highlighted = this.highlightLatexCode(code);
            if (comment === null) {
                return highlighted;
            }
            return `${highlighted}<span class="token-comment">${this.escapeHtml(comment)}</span>`;
        }).join('\n');
    },

    splitLatexComment(line) {
        for (let i = 0; i < line.length; i += 1) {
            if (line[i] === '%' && (i === 0 || line[i - 1] !== '\\')) {
                return { code: line.slice(0, i), comment: line.slice(i) };
            }
        }
        return { code: line, comment: null };
    },

    highlightLatexCode(code) {
        let output = '';
        let i = 0;
        while (i < code.length) {
            const ch = code[i];
            if (ch === '\\') {
                let j = i + 1;
                while (j < code.length && /[A-Za-z@*]/.test(code[j])) j += 1;
                const cmd = code.slice(i, j);
                output += `<span class="token-command">${this.escapeHtml(cmd)}</span>`;
                i = j;

                if ((cmd === '\\begin' || cmd === '\\end') && code[i] === '{') {
                    let k = i + 1;
                    while (k < code.length && code[k] !== '}') k += 1;
                    const env = code.slice(i + 1, k);
                    output += `<span class="token-brace">{</span><span class="token-env">${this.escapeHtml(env)}</span>`;
                    if (k < code.length && code[k] === '}') {
                        output += `<span class="token-brace">}</span>`;
                        i = k + 1;
                    }
                }
                continue;
            }
            if (ch === '{' || ch === '}') {
                output += `<span class="token-brace">${this.escapeHtml(ch)}</span>`;
                i += 1;
                continue;
            }
            if (ch === '$') {
                let j = i + 1;
                while (j < code.length) {
                    if (code[j] === '$' && code[j - 1] !== '\\') break;
                    j += 1;
                }
                const math = code.slice(i, Math.min(j + 1, code.length));
                output += `<span class="token-math">${this.escapeHtml(math)}</span>`;
                i = j + 1;
                continue;
            }
            if (ch === '&') {
                output += `<span class="token-operator">&amp;</span>`;
                i += 1;
                continue;
            }
            if (/\d/.test(ch)) {
                let j = i + 1;
                while (j < code.length && /[\d\.]/.test(code[j])) j += 1;
                const num = code.slice(i, j);
                output += `<span class="token-number">${this.escapeHtml(num)}</span>`;
                i = j;
                continue;
            }
            output += this.escapeHtml(ch);
            i += 1;
        }
        return output;
    },

    // ==========================================
    // LINE NUMBERS FUNCTIONALITY
    // ==========================================

    toggleLineNumbers() {
        const lineNumbers = this.elements.lineNumbers;
        const btn = this.elements.btnLineNumbers;
        if (!lineNumbers) return;

        lineNumbers.classList.toggle('hidden');
        btn?.classList.toggle('active');

        // Save preference
        const isVisible = !lineNumbers.classList.contains('hidden');
        localStorage.setItem('paperreader_linenumbers', isVisible ? 'true' : 'false');

        showToast(isVisible ? 'Line numbers shown' : 'Line numbers hidden');
    },

    loadLineNumbersPreference() {
        const pref = localStorage.getItem('paperreader_linenumbers');
        if (pref === 'false') {
            this.elements.lineNumbers?.classList.add('hidden');
            this.elements.btnLineNumbers?.classList.remove('active');
        }
    },

    updateLineNumbers() {
        const editor = this.elements.codeEditor;
        const lineNumbers = this.elements.lineNumbers;
        if (!editor || !lineNumbers) return;

        const lines = editor.value.split('\n');
        const lineCount = lines.length;

        let html = '';
        for (let i = 1; i <= lineCount; i++) {
            html += `<span data-line="${i}">${i}</span>`;
        }
        lineNumbers.innerHTML = html;
    },

    syncLineNumberScroll() {
        const editor = this.elements.codeEditor;
        const lineNumbers = this.elements.lineNumbers;
        if (!editor || !lineNumbers) return;

        lineNumbers.scrollTop = editor.scrollTop;
    },

    updateCurrentLine() {
        const editor = this.elements.codeEditor;
        const lineNumbers = this.elements.lineNumbers;
        if (!editor || !lineNumbers) return;

        const cursorPos = editor.selectionStart;
        const textBefore = editor.value.substring(0, cursorPos);
        const currentLine = textBefore.split('\n').length;

        // Remove previous active
        lineNumbers.querySelectorAll('.active').forEach(el => el.classList.remove('active'));

        // Add active to current line
        const lineEl = lineNumbers.querySelector(`[data-line="${currentLine}"]`);
        if (lineEl) lineEl.classList.add('active');
    },

    // ==========================================
    // PANEL RESIZER FUNCTIONALITY
    // ==========================================

    setupPanelResizer() {
        const resizer = this.elements.panelResizer;
        const sourceView = this.elements.sourceView;
        const previewView = this.elements.previewView;

        if (!resizer || !sourceView || !previewView) return;

        let isResizing = false;
        let startX = 0;
        let startSourceWidth = 0;
        let startPreviewWidth = 0;

        const startResize = (e) => {
            isResizing = true;
            startX = e.clientX || e.touches?.[0]?.clientX || 0;
            startSourceWidth = sourceView.offsetWidth;
            startPreviewWidth = previewView.offsetWidth;

            resizer.classList.add('dragging');
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';

            // Disable transitions during drag for smooth resizing
            sourceView.style.transition = 'none';
            previewView.style.transition = 'none';
        };

        const doResize = (e) => {
            if (!isResizing) return;

            const currentX = e.clientX || e.touches?.[0]?.clientX || 0;
            const delta = currentX - startX;

            const containerWidth = sourceView.parentElement.offsetWidth - resizer.offsetWidth;
            const minWidth = 200;

            let newSourceWidth = startSourceWidth + delta;
            let newPreviewWidth = startPreviewWidth - delta;

            // Enforce minimum widths
            if (newSourceWidth < minWidth) {
                newSourceWidth = minWidth;
                newPreviewWidth = containerWidth - minWidth;
            }
            if (newPreviewWidth < minWidth) {
                newPreviewWidth = minWidth;
                newSourceWidth = containerWidth - minWidth;
            }

            // Apply as flex-basis for smooth sizing
            sourceView.style.flex = `0 0 ${newSourceWidth}px`;
            previewView.style.flex = `0 0 ${newPreviewWidth}px`;
        };

        const stopResize = () => {
            if (!isResizing) return;
            isResizing = false;

            resizer.classList.remove('dragging');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';

            // Re-enable transitions
            sourceView.style.transition = '';
            previewView.style.transition = '';

            // Save panel sizes
            const sourcePercent = (sourceView.offsetWidth / sourceView.parentElement.offsetWidth) * 100;
            localStorage.setItem('paperreader_panel_ratio', sourcePercent.toString());
        };

        // Mouse events
        resizer.addEventListener('mousedown', startResize);
        document.addEventListener('mousemove', doResize);
        document.addEventListener('mouseup', stopResize);

        // Touch events for mobile
        resizer.addEventListener('touchstart', startResize, { passive: true });
        document.addEventListener('touchmove', doResize, { passive: true });
        document.addEventListener('touchend', stopResize);

        // Load saved panel ratio
        this.loadPanelRatio();
    },

    loadPanelRatio() {
        const saved = localStorage.getItem('paperreader_panel_ratio');
        if (saved) {
            const sourcePercent = parseFloat(saved);
            if (sourcePercent > 10 && sourcePercent < 90) {
                const sourceView = this.elements.sourceView;
                const previewView = this.elements.previewView;
                if (sourceView && previewView) {
                    sourceView.style.flex = `0 0 ${sourcePercent}%`;
                    previewView.style.flex = `0 0 ${100 - sourcePercent - 1}%`; // Account for resizer
                }
            }
        }
    },

    async syncSourceToPdf() {
        if (!this.state.currentFile || !this.state.currentFile.endsWith('.tex')) {
            showToast("Open a .tex file to sync");
            return;
        }
        if (this.isLocalProject()) {
            showToast('SyncTeX requires a server project');
            return;
        }

        const { line, column } = this.getCursorPosition();
        const pdfName = this.state.lastCompiledPdfUrl?.split('/').pop();

        if (!pdfName) {
            showToast("Compile first to sync");
            return;
        }

        try {
            const res = await fetch(this.withProjectParam(`${API_BASE}/api/synctex/forward`), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    tex_file: this.state.currentFile,
                    line: line,
                    column: column,
                    pdf_file: pdfName
                })
            });
            const result = await res.json();

            if (result.success) {
                this.scrollToPdfLocation(result.page, result.x, result.y);
                showToast(`Syncing to page ${result.page}`);
            } else {
                console.warn("Forward Sync failed", result);
                showToast("No PDF location found for this line");
            }
        } catch (e) {
            console.error("Forward Sync error", e);
        }
    },

    getCursorPosition() {
        const editor = this.elements.codeEditor;
        if (!editor) return { line: 1, column: 1 };
        const pos = editor.selectionStart || 0;
        const textBeforeCursor = editor.value.substring(0, pos);
        const line = textBeforeCursor.split('\n').length;
        const lastNewline = textBeforeCursor.lastIndexOf('\n');
        const column = pos - lastNewline;
        return { line, column };
    },

    getCurrentLine() {
        return this.getCursorPosition().line;
    },

    scrollToPdfLocation(page, x, y) {
        const pageEl = document.querySelector(`.pdf-page[data-page-number="${page}"]`);
        if (!pageEl) return;

        // Scroll the container to this page
        pageEl.scrollIntoView({ behavior: 'smooth', block: 'center' });

        // Add a temporary highlight at the location
        const marker = document.createElement('div');
        const scale = this.state.pdfScale || 1;
        const pageHeight = pageEl.clientHeight || 0;
        let markerX = x * scale;
        let markerY = y * scale;
        if (markerY > pageHeight || markerY < 0) {
            markerY = pageHeight - (y * scale);
        }

        marker.style.position = 'absolute';
        marker.style.left = `${markerX}px`;
        marker.style.top = `${markerY}px`;
        marker.style.width = '100px';
        marker.style.height = '20px';
        marker.style.background = 'rgba(0, 255, 255, 0.4)';
        marker.style.boxShadow = '0 0 10px rgba(0, 255, 255, 0.6)';
        marker.style.borderRadius = '4px';
        marker.style.pointerEvents = 'none';
        marker.style.zIndex = '100';
        marker.style.transform = 'translate(-20px, -50%)';

        pageEl.appendChild(marker);
        setTimeout(() => marker.remove(), 2500);
    }
};

/**
 * Team Chat Controller - multi-user chat panel
 */
const TeamChatController = {
    isOpen: false,
    fabPosition: null,
    messages: [],
    channel: null,
    channelName: null,
    participants: new Map(),
    presenceUsers: [],
    projectId: 'default',
    projectName: 'PROJECT',
    elements: {
        fab: null,
        panel: null,
        backdrop: null,
        messages: null,
        input: null,
        send: null,
        close: null,
        usersBtn: null,
        usersPanel: null,
        usersList: null,
        userCount: null,
        projectLabel: null
    },

    init() {
        this.cacheElements();
        if (!this.elements.fab || !this.elements.panel) return;
        this.bindEvents();
        this.setProject(ProjectManager.activeProject);
        this.loadFabPosition();
        this.applyFabPosition();
        this.makeDraggable();
        window.addEventListener('resize', () => {
            this.clampFabPosition();
            if (this.isOpen) this.positionPanel();
        });
    },

    cacheElements() {
        this.elements.fab = document.getElementById('teamchat-fab');
        this.elements.panel = document.getElementById('teamchat-panel');
        this.elements.backdrop = document.getElementById('teamchat-backdrop');
        this.elements.messages = document.getElementById('teamchat-messages');
        this.elements.input = document.getElementById('teamchat-input');
        this.elements.send = document.getElementById('teamchat-send');
        this.elements.close = document.getElementById('teamchat-close');
        this.elements.usersBtn = document.getElementById('teamchat-users');
        this.elements.usersPanel = document.getElementById('teamchat-users-panel');
        this.elements.usersList = document.getElementById('teamchat-user-list');
        this.elements.userCount = document.getElementById('teamchat-user-count');
        this.elements.projectLabel = document.getElementById('teamchat-project');
    },

    bindEvents() {
        this.elements.close?.addEventListener('click', () => this.close());
        this.elements.backdrop?.addEventListener('click', () => this.close());
        this.elements.send?.addEventListener('click', () => this.sendMessage());
        this.elements.input?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        this.elements.usersBtn?.addEventListener('click', () => this.toggleUsersPanel());
    },

    makeDraggable() {
        const fab = this.elements.fab;
        if (!fab) return;

        let isMouseDown = false;
        let isDragging = false;
        let startX, startY, startLeft, startBottom;
        let totalDelta = 0;

        const promoteToDragThreshold = 10;
        const tapThreshold = 14;

        const onStart = (e) => {
            isMouseDown = true;
            isDragging = false;
            const clientX = e.type.includes('touch') ? e.touches[0].clientX : e.clientX;
            const clientY = e.type.includes('touch') ? e.touches[0].clientY : e.clientY;
            startX = clientX;
            startY = clientY;

            const rect = fab.getBoundingClientRect();
            startLeft = rect.left;
            startBottom = window.innerHeight - rect.bottom;

            fab.style.transition = 'none';
            fab.style.cursor = 'pointer';
        };

        const onMove = (e) => {
            if (!isMouseDown) return;
            const clientX = e.type.includes('touch') ? e.touches[0].clientX : e.clientX;
            const clientY = e.type.includes('touch') ? e.touches[0].clientY : e.clientY;
            const deltaX = clientX - startX;
            const deltaY = clientY - startY;

            totalDelta = Math.max(Math.abs(deltaX), Math.abs(deltaY));

            if (!isDragging && (Math.abs(deltaX) > promoteToDragThreshold || Math.abs(deltaY) > promoteToDragThreshold)) {
                isDragging = true;
                fab.style.cursor = 'grabbing';
            }

            if (!isDragging) return;

            const rect = fab.getBoundingClientRect();
            const newLeft = startLeft + deltaX;
            const newBottom = startBottom - deltaY;
            const maxLeft = window.innerWidth - rect.width - 10;
            const maxBottom = window.innerHeight - rect.height - 10;

            const clampedLeft = Math.max(10, Math.min(newLeft, maxLeft));
            const clampedBottom = Math.max(10, Math.min(newBottom, maxBottom));

            fab.style.left = `${clampedLeft}px`;
            fab.style.right = 'auto';
            fab.style.bottom = `${clampedBottom}px`;

            this.fabPosition = { left: clampedLeft, bottom: clampedBottom };
        };

        const onEnd = () => {
            if (!isMouseDown) return;
            fab.style.transition = '';
            fab.style.cursor = 'pointer';

            if (!isDragging && totalDelta < tapThreshold) {
                this.toggle();
            } else if (this.fabPosition) {
                this.saveFabPosition();
            }

            isMouseDown = false;
            isDragging = false;
            totalDelta = 0;
        };

        fab.addEventListener('mousedown', onStart);
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onEnd);

        fab.addEventListener('touchstart', onStart, { passive: true });
        document.addEventListener('touchmove', onMove, { passive: true });
        document.addEventListener('touchend', onEnd);
    },

    loadFabPosition() {
        try {
            const saved = localStorage.getItem('paperreader_teamchat_fab_pos');
            this.fabPosition = saved ? JSON.parse(saved) : null;
        } catch (e) {
            this.fabPosition = null;
        }
    },

    saveFabPosition() {
        if (!this.fabPosition) return;
        try {
            localStorage.setItem('paperreader_teamchat_fab_pos', JSON.stringify(this.fabPosition));
        } catch (e) {
            console.warn('Failed to save chat button position');
        }
    },

    applyFabPosition() {
        if (!this.elements.fab || !this.fabPosition) return;
        this.elements.fab.style.left = `${this.fabPosition.left}px`;
        this.elements.fab.style.right = 'auto';
        this.elements.fab.style.bottom = `${this.fabPosition.bottom}px`;
    },

    clampFabPosition() {
        if (!this.elements.fab || !this.fabPosition) return;
        const rect = this.elements.fab.getBoundingClientRect();
        const maxLeft = window.innerWidth - rect.width - 10;
        const maxBottom = window.innerHeight - rect.height - 10;
        const clampedLeft = Math.max(10, Math.min(this.fabPosition.left, maxLeft));
        const clampedBottom = Math.max(10, Math.min(this.fabPosition.bottom, maxBottom));
        this.fabPosition = { left: clampedLeft, bottom: clampedBottom };
        this.applyFabPosition();
    },

    toggle() {
        if (this.isOpen) {
            this.close();
        } else {
            this.open();
        }
    },

    open() {
        this.isOpen = true;
        this.elements.panel?.classList.add('open');
        this.elements.fab?.classList.add('active');
        this.elements.backdrop?.classList.add('visible');
        this.positionPanel();
        if (!this.isMobileView()) {
            this.elements.input?.focus();
        }
    },

    close() {
        this.isOpen = false;
        this.elements.panel?.classList.remove('open');
        this.elements.fab?.classList.remove('active');
        this.elements.backdrop?.classList.remove('visible');
        if (this.elements.usersPanel) {
            this.elements.usersPanel.style.display = 'none';
        }
    },

    isMobileView() {
        return window.matchMedia('(max-width: 480px)').matches;
    },

    positionPanel() {
        if (!this.elements.panel || !this.elements.fab) return;
        const panel = this.elements.panel;
        const fab = this.elements.fab;

        if (this.isMobileView()) {
            panel.style.left = '0';
            panel.style.right = '0';
            panel.style.bottom = '0';
            panel.style.top = 'auto';
            return;
        }

        const rect = fab.getBoundingClientRect();
        const panelWidth = panel.offsetWidth || 380;
        let left = rect.left - 10;
        let top = rect.bottom + 12;

        if (left + panelWidth > window.innerWidth - 20) {
            left = window.innerWidth - panelWidth - 20;
        }
        if (left < 20) left = 20;

        if (top + panel.offsetHeight > window.innerHeight - 20) {
            top = window.innerHeight - panel.offsetHeight - 20;
            if (top < rect.top) {
                top = rect.top - panel.offsetHeight - 12;
            }
        }

        panel.style.left = `${left}px`;
        panel.style.top = `${top}px`;
        panel.style.right = 'auto';
        panel.style.bottom = 'auto';
    },

    setProject(project) {
        this.projectId = project?.id || 'default';
        this.projectName = project?.name || 'PROJECT';
        this.participants = new Map();
        this.updateProjectLabel();
        this.loadMessages();
        this.renderMessages();
        this.initChannel();
        this.refreshUserList();
    },

    updateProjectLabel() {
        if (this.elements.projectLabel) {
            this.elements.projectLabel.textContent = this.truncateLabel(this.projectName.toUpperCase(), 18);
        }
    },

    truncateLabel(text, maxLen) {
        if (!text) return '';
        if (text.length <= maxLen) return text;
        return `${text.slice(0, maxLen - 3)}...`;
    },

    getStorageKey() {
        return `paperreader_chat_${this.projectId || 'default'}`;
    },

    loadMessages() {
        try {
            const saved = localStorage.getItem(this.getStorageKey());
            this.messages = saved ? JSON.parse(saved) : [];
            this.messages = this.messages.map(message => ({
                ...message,
                projectId: message.projectId || this.projectId
            }));
            this.messages.sort((a, b) => new Date(a.timestamp || 0) - new Date(b.timestamp || 0));
        } catch (e) {
            this.messages = [];
        }
    },

    saveMessages() {
        try {
            localStorage.setItem(this.getStorageKey(), JSON.stringify(this.messages));
        } catch (e) {
            console.warn('Failed to save chat history');
        }
    },

    addMessage(message, fromChannel = false) {
        if (!message || message.projectId !== this.projectId) return;
        if (this.messages.find(msg => msg.id === message.id)) return;
        this.messages.push(message);
        if (this.messages.length > 200) {
            this.messages = this.messages.slice(-200);
        }
        if (!fromChannel) {
            this.broadcast({ type: 'message', projectId: this.projectId, message });
        }
        this.saveMessages();
        this.trackParticipant(message);
        this.renderMessages();
    },

    renderMessages() {
        if (!this.elements.messages) return;
        const currentUser = this.getCurrentUser();
        const currentId = currentUser?.id || currentUser?.name;
        if (!this.messages.length) {
            this.elements.messages.innerHTML = '<div class="teamchat-empty">No messages yet. Say hello!</div>';
            return;
        }
        this.elements.messages.innerHTML = this.messages.map(message => {
            const time = new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            const isSelf = message.userId && currentId && message.userId === currentId;
            const name = message.userName || message.user || 'Guest';
            const safeBody = escapeHtml(message.text).replace(/\n/g, '<br>');
            return `
                <div class="teamchat-message ${isSelf ? 'self' : ''}">
                    <div class="meta">
                        <span>${escapeHtml(name)}</span>
                        <span>${time}</span>
                    </div>
                    <div class="body">${safeBody}</div>
                </div>
            `;
        }).join('');
        this.elements.messages.scrollTop = this.elements.messages.scrollHeight;
    },

    sendMessage() {
        const input = this.elements.input;
        const text = (input?.value || '').trim();
        if (!text) return;
        const user = this.getCurrentUser();
        const message = {
            id: `${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
            projectId: this.projectId,
            userId: user?.id || user?.name || 'guest',
            userName: user?.name || 'Guest',
            avatar: user?.avatar || '',
            text,
            timestamp: new Date().toISOString()
        };
        this.addMessage(message);
        if (input) input.value = '';
    },

    initChannel() {
        this.closeChannel();
        if (!('BroadcastChannel' in window)) return;
        this.channelName = `paperreader-chat-${this.projectId}`;
        this.channel = new BroadcastChannel(this.channelName);
        this.channel.onmessage = (event) => this.handleChannelMessage(event.data);
        this.broadcastPresence();
    },

    closeChannel() {
        if (this.channel) {
            this.channel.close();
            this.channel = null;
        }
    },

    broadcast(payload) {
        if (!this.channel) return;
        this.channel.postMessage(payload);
    },

    broadcastPresence() {
        const user = this.getCurrentUser();
        if (!user) return;
        this.broadcast({ type: 'presence', projectId: this.projectId, user });
    },

    handleChannelMessage(data) {
        if (!data || data.projectId !== this.projectId) return;
        if (data.type === 'message') {
            this.addMessage(data.message, true);
        }
        if (data.type === 'presence') {
            this.trackParticipant(data.user);
            this.refreshUserList();
        }
    },

    trackParticipant(messageOrUser) {
        const user = messageOrUser?.userName || messageOrUser?.user ? {
            id: messageOrUser.userId || messageOrUser.user || messageOrUser.userName,
            name: messageOrUser.userName || messageOrUser.user,
            avatar: messageOrUser.avatar
        } : messageOrUser;
        if (!user || !user.name) return;
        this.participants.set(user.id || user.name, user);
        this.refreshUserList();
    },

    toggleUsersPanel() {
        if (!this.elements.usersPanel) return;
        const isVisible = this.elements.usersPanel.style.display === 'block';
        this.elements.usersPanel.style.display = isVisible ? 'none' : 'block';
    },

    updateUsersFromPresence(users) {
        this.presenceUsers = users || [];
        this.refreshUserList();
    },

    refreshUserList() {
        if (!this.elements.usersList || !this.elements.userCount) return;
        const list = this.presenceUsers.length
            ? this.presenceUsers.map(user => ({ name: user.name, id: user.name }))
            : Array.from(this.participants.values());

        this.elements.userCount.textContent = list.length.toString();
        if (!list.length) {
            this.elements.usersList.innerHTML = '<div class="outline-empty">No users yet</div>';
            return;
        }
        this.elements.usersList.innerHTML = list.map(user => `
            <div class="teamchat-user">
                <i class="bi bi-person-circle"></i>
                <span>${escapeHtml(user.name)}</span>
            </div>
        `).join('');
    },

    refreshUser() {
        this.broadcastPresence();
        this.refreshUserList();
    },

    getCurrentUser() {
        const authProfile = AuthController.getPresenceProfile();
        if (authProfile) return authProfile;
        if (PresenceController?.user) return PresenceController.user;
        return { name: 'Guest', id: 'guest' };
    }
};

/**
 * Copilot Controller - AI Assistant with MLLM Integration
 */
const CopilotController = {
    isOpen: false,
    isListening: false,
    recognition: null,
    selectedContext: null,
    conversationHistory: [],
    pendingImage: null,
    lastDragDistance: 0,
    settingsOpen: false,
    availableModels: [],
    llmBase: null,
    llmModel: null,

    elements: {
        fab: null,
        panel: null,
        backdrop: null,
        messagesContainer: null,
        input: null,
        sendBtn: null,
        voiceBtn: null,
        imageBtn: null,
        imageInput: null,
        closeBtn: null,
        clearBtn: null,
        settingsBtn: null,
        settingsPanel: null,
        apiBaseInput: null,
        modelSelect: null,
        modelRefresh: null,
        contextBlock: null,
        contextCode: null,
        contextLabel: null,
        attachments: null
    },

    init() {
        // Get DOM elements
        this.elements.fab = document.getElementById('copilot-fab');
        this.elements.panel = document.getElementById('copilot-panel');
        this.elements.messagesContainer = document.getElementById('copilot-messages');
        this.elements.input = document.getElementById('copilot-input');
        this.elements.sendBtn = document.getElementById('copilot-send');
        this.elements.voiceBtn = document.getElementById('copilot-voice');
        this.elements.imageBtn = document.getElementById('copilot-attach-image');
        this.elements.imageInput = document.getElementById('copilot-image-input');
        this.elements.closeBtn = document.getElementById('copilot-minimize');
        this.elements.clearBtn = document.getElementById('copilot-clear');
        this.elements.settingsBtn = document.getElementById('copilot-settings');
        this.elements.settingsPanel = document.getElementById('copilot-settings-panel');
        this.elements.apiBaseInput = document.getElementById('copilot-api-base');
        this.elements.modelSelect = document.getElementById('copilot-model-select');
        this.elements.modelRefresh = document.getElementById('copilot-model-refresh');
        this.elements.contextBlock = document.getElementById('copilot-context');
        this.elements.contextCode = document.getElementById('context-code');
        this.elements.contextLabel = document.querySelector('#copilot-context .context-label span');
        this.elements.attachments = document.getElementById('copilot-attachments');
        this.elements.resizer = document.getElementById('copilot-resizer');

        if (!this.elements.fab) return;

        this.setupBackdrop();

        // Setup resizer
        this.setupPanelResizer();

        // Setup event listeners
        this.setupEventListeners();
        this.makeDraggable();
        this.setupPanelGestures();
        this.setupSpeechRecognition();
        this.setupTextSelection();
        this.loadLlmSettings();
        this.refreshModels();

        window.addEventListener('resize', () => {
            if (this.isOpen) {
                this.positionPanel();
            }
        });

        // Add welcome message
        this.addMessage('assistant', 'Hello! I\'m your LaTeX Copilot. Select text from the editor for context, then ask me anything about your paper. I can help with writing, formatting, citations, and more!');
    },

    setupEventListeners() {
        // FAB click to toggle panel - handled in makeDraggable to avoid drag conflicts

        // Close button
        this.elements.closeBtn?.addEventListener('click', () => this.close());

        // Clear button
        this.elements.clearBtn?.addEventListener('click', () => this.clearChat());

        // Settings button
        this.elements.settingsBtn?.addEventListener('click', () => this.toggleSettings());

        // Send button
        this.elements.sendBtn?.addEventListener('click', () => this.sendMessage());

        // Input enter key
        this.elements.input?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Voice button
        this.elements.voiceBtn?.addEventListener('click', () => this.toggleVoice());

        this.elements.apiBaseInput?.addEventListener('change', () => this.handleApiBaseChange());
        this.elements.modelSelect?.addEventListener('change', () => this.handleModelChange());
        this.elements.modelRefresh?.addEventListener('click', () => this.refreshModels(true));

        // Image button and input
        this.elements.imageBtn?.addEventListener('click', () => {
            this.elements.imageInput?.click();
        });

        this.elements.imageInput?.addEventListener('change', (e) => {
            this.handleImageUpload(e);
        });

        // Paste image support
        this.elements.input?.addEventListener('paste', (e) => this.handlePaste(e));

        // Clear selected context
        document.getElementById('context-clear')?.addEventListener('click', () => this.clearContext());
    },

    normalizeLlmBase(base) {
        if (!base) return base;
        const trimmed = base.trim().replace(/\/+$/, '');
        if (trimmed.includes('/api/llm')) {
            return trimmed;
        }
        if (trimmed.endsWith('/llm')) {
            return `${trimmed}/v1`;
        }
        if (trimmed.endsWith('/llm/v1') || trimmed.endsWith('/v1')) {
            return trimmed;
        }
        return trimmed;
    },

    loadLlmSettings() {
        const defaultBase = isLocalDev
            ? 'http://127.0.0.1:22222/llm/v1'
            : 'https://game.agaii.org/llm/v1';
        const storedBase = localStorage.getItem('paperreader_llm_base') || defaultBase;
        const base = this.normalizeLlmBase(storedBase);
        const model = localStorage.getItem('paperreader_llm_model') || '';
        this.llmBase = base;
        this.llmModel = model;
        if (this.elements.apiBaseInput) {
            this.elements.apiBaseInput.value = base;
        }
        if (storedBase !== base) {
            this.saveLlmSettings();
        }
    },

    saveLlmSettings() {
        if (this.llmBase) localStorage.setItem('paperreader_llm_base', this.llmBase);
        if (this.llmModel) localStorage.setItem('paperreader_llm_model', this.llmModel);
    },

    toggleSettings() {
        this.settingsOpen = !this.settingsOpen;
        if (this.elements.settingsPanel) {
            this.elements.settingsPanel.classList.toggle('visible', this.settingsOpen);
            this.elements.settingsPanel.setAttribute('aria-hidden', this.settingsOpen ? 'false' : 'true');
        }
    },

    handleApiBaseChange() {
        const value = this.elements.apiBaseInput?.value?.trim();
        if (!value) return;
        const normalized = this.normalizeLlmBase(value);
        this.llmBase = normalized;
        if (this.elements.apiBaseInput) {
            this.elements.apiBaseInput.value = normalized;
        }
        this.saveLlmSettings();
        this.refreshModels(true);
    },

    handleModelChange() {
        const value = this.elements.modelSelect?.value;
        if (!value) return;
        this.llmModel = value;
        this.saveLlmSettings();
    },

    resolveLlmEndpoint(base, type) {
        const normalized = this.normalizeLlmBase(base);
        if (this.shouldUseProxy(normalized)) {
            const proxyBase = this.getProxyBase();
            return `${proxyBase}/${type}`;
        }
        if (normalized.includes('/api/llm')) {
            return `${normalized}/${type}`;
        }
        if (type === 'chat') return `${normalized}/chat/completions`;
        return `${normalized}/models`;
    },

    getProxyBase() {
        return `${window.location.protocol}//${window.location.hostname}/api/llm`;
    },

    shouldUseProxy(base) {
        if (!base) return false;
        if (base.includes('/api/llm')) return false;
        try {
            const targetOrigin = new URL(base).origin;
            if (targetOrigin === window.location.origin) return false;
        } catch (e) {
            return false;
        }
        return base.includes('game.agaii.org/llm');
    },

    async refreshModels(force = false) {
        if (!this.llmBase) return;
        if (!force && this.availableModels.length) return;

        const url = this.resolveLlmEndpoint(this.llmBase, 'models');
        try {
            const response = await fetch(url, { method: 'GET' });
            if (!response.ok) throw new Error(`Model fetch failed: ${response.status}`);
            const data = await response.json();
            const rawModels = Array.isArray(data)
                ? data
                : (data.data || data.models || []);
            const models = rawModels
                .map((item) => item.id || item.model || item.name || item)
                .filter(Boolean);
            this.availableModels = Array.from(new Set(models));
            this.renderModelOptions();
        } catch (e) {
            console.warn('Failed to load models', e);
            if (!this.availableModels.length) {
                this.availableModels = ['gpt-4o-mini'];
                this.renderModelOptions();
            }
        }
    },

    renderModelOptions() {
        const select = this.elements.modelSelect;
        if (!select) return;

        select.innerHTML = '';
        if (!this.availableModels.length) {
            select.innerHTML = '<option value="">No models</option>';
            return;
        }

        this.availableModels.forEach((model) => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            select.appendChild(option);
        });

        if (this.llmModel && this.availableModels.includes(this.llmModel)) {
            select.value = this.llmModel;
        } else {
            this.llmModel = this.availableModels[0];
            select.value = this.llmModel;
            this.saveLlmSettings();
        }
    },

    setupBackdrop() {
        if (this.elements.backdrop) return;
        const backdrop = document.createElement('div');
        backdrop.className = 'copilot-backdrop';
        backdrop.addEventListener('click', () => this.close());
        document.body.appendChild(backdrop);
        this.elements.backdrop = backdrop;
    },

    makeDraggable() {
        const fab = this.elements.fab;
        if (!fab) return;

        let isMouseDown = false;
        let isDragging = false;
        let startX, startY, startLeft, startBottom;
        let totalDelta = 0;

        const promoteToDragThreshold = 10; // px, reduce accidental drags
        const tapThreshold = 14; // px, tolerate finger jitter on mobile

        const onStart = (e) => {
            isMouseDown = true;
            isDragging = false;

            const clientX = e.type.includes('touch') ? e.touches[0].clientX : e.clientX;
            const clientY = e.type.includes('touch') ? e.touches[0].clientY : e.clientY;

            startX = clientX;
            startY = clientY;

            const rect = fab.getBoundingClientRect();
            startLeft = rect.left;
            startBottom = window.innerHeight - rect.bottom;

            fab.style.transition = 'none';
            fab.style.cursor = 'pointer';
        };

        const onMove = (e) => {
            if (!isMouseDown) return;

            const clientX = e.type.includes('touch') ? e.touches[0].clientX : e.clientX;
            const clientY = e.type.includes('touch') ? e.touches[0].clientY : e.clientY;

            const deltaX = clientX - startX;
            const deltaY = clientY - startY;

            totalDelta = Math.max(Math.abs(deltaX), Math.abs(deltaY));

            // Only start dragging after threshold to avoid accidental drags on click
            if (!isDragging && (Math.abs(deltaX) > promoteToDragThreshold || Math.abs(deltaY) > promoteToDragThreshold)) {
                isDragging = true;
                fab.style.cursor = 'grabbing';
            }

            if (!isDragging) return;

            const newLeft = startLeft + deltaX;
            const newBottom = startBottom - deltaY;

            // Constrain to viewport
            const fabSize = 60;
            const maxLeft = window.innerWidth - fabSize - 10;
            const maxBottom = window.innerHeight - fabSize - 10;

            fab.style.left = `${Math.max(10, Math.min(newLeft, maxLeft))}px`;
            fab.style.right = 'auto';
            fab.style.bottom = `${Math.max(10, Math.min(newBottom, maxBottom))}px`;

            if (this.isOpen) {
                this.positionPanel();
            }
        };

        const onEnd = () => {
            if (!isMouseDown) return;

            fab.style.transition = '';
            fab.style.cursor = 'pointer';

            // Only treat as click when movement was tiny
            if (!isDragging && totalDelta < tapThreshold) {
                this.toggle();
            }

            isMouseDown = false;
            isDragging = false;
            totalDelta = 0;
        };

        fab.addEventListener('mousedown', onStart);
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onEnd);

        fab.addEventListener('touchstart', onStart, { passive: true });
        document.addEventListener('touchmove', onMove, { passive: true });
        document.addEventListener('touchend', onEnd);
    },

    setupPanelGestures() {
        const panel = this.elements.panel;
        const header = panel?.querySelector('.copilot-header');
        if (!panel || !header) return;

        let isDragging = false;
        let startY = 0;
        let deltaY = 0;

        const start = (e) => {
            if (!this.isOpen) return;
            if (e.target.closest('button, textarea, input, select')) return;
            isDragging = true;
            startY = e.touches ? e.touches[0].clientY : e.clientY;
            deltaY = 0;
            panel.classList.add('dragging');
        };

        const move = (e) => {
            if (!isDragging) return;
            const currentY = e.touches ? e.touches[0].clientY : e.clientY;
            deltaY = Math.max(0, currentY - startY);
            panel.style.transform = `translateY(${deltaY}px)`;
            panel.style.opacity = `${Math.max(0.4, 1 - deltaY / 240)}`;
        };

        const end = () => {
            if (!isDragging) return;
            panel.classList.remove('dragging');
            panel.style.opacity = '';
            if (deltaY > 120) {
                this.close();
            } else {
                panel.style.transform = '';
            }
            isDragging = false;
        };

        header.addEventListener('mousedown', start);
        document.addEventListener('mousemove', move);
        document.addEventListener('mouseup', end);

        header.addEventListener('touchstart', start, { passive: true });
        header.addEventListener('touchmove', move, { passive: true });
        header.addEventListener('touchend', end);
    },

    setupSpeechRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            console.warn('Speech recognition not supported');
            if (this.elements.voiceBtn) {
                this.elements.voiceBtn.style.display = 'none';
            }
            return;
        }

        this.recognition = new SpeechRecognition();
        this.recognition.continuous = false;
        this.recognition.interimResults = true;
        this.recognition.lang = 'en-US';

        this.recognition.onstart = () => {
            this.isListening = true;
            this.elements.voiceBtn?.classList.add('listening');
        };

        this.recognition.onend = () => {
            this.isListening = false;
            this.elements.voiceBtn?.classList.remove('listening');
        };

        this.recognition.onresult = (event) => {
            let finalTranscript = '';
            let interimTranscript = '';

            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript;
                } else {
                    interimTranscript += transcript;
                }
            }

            if (finalTranscript) {
                this.elements.input.value += finalTranscript + ' ';
            }
        };

        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this.isListening = false;
            this.elements.voiceBtn?.classList.remove('listening');
        };
    },

    setupTextSelection() {
        // Listen for text selection in the editor
        document.addEventListener('mouseup', () => {
            setTimeout(() => this.captureSelection(), 10);
        });

        document.addEventListener('keyup', (e) => {
            if (e.shiftKey) {
                setTimeout(() => this.captureSelection(), 10);
            }
        });
    },

    captureSelection() {
        const selection = window.getSelection();
        const selectedText = selection?.toString().trim();
        const codeEditor = document.getElementById('code-editor');

        // Prefer textarea selection when available (gives us line numbers)
        if (codeEditor && codeEditor.selectionStart !== undefined && codeEditor.selectionEnd !== undefined) {
            const start = codeEditor.selectionStart;
            const end = codeEditor.selectionEnd;
            if (start !== end) {
                const fullText = codeEditor.value || '';
                const text = fullText.slice(start, end).trim();
                if (text) {
                    const startLine = fullText.slice(0, start).split('\n').length;
                    const endLine = fullText.slice(0, end).split('\n').length;
                    this.selectedContext = { text, startLine, endLine };
                    this.showContextToast(text, startLine, endLine);
                    return;
                }
            }
        }

        // Fallback to DOM selection (outside textarea)
        if (selectedText && selectedText.length > 0) {
            this.selectedContext = { text: selectedText, startLine: null, endLine: null };
            this.showContextToast(selectedText, null, null);
        }
    },

    showContextToast(text, startLine, endLine) {
        const block = this.elements.contextBlock;
        if (!block) return;

        const previewText = text.length > 200 ? `${text.slice(0, 200)}...` : text;
        const linesLabel = startLine && endLine ? `Context: Lines ${startLine}–${endLine}` : 'Selected Context';

        if (this.elements.contextLabel) {
            this.elements.contextLabel.textContent = linesLabel;
        }
        if (this.elements.contextCode) {
            this.elements.contextCode.textContent = previewText;
        }

        block.style.display = 'block';
    },

    toggle() {
        if (this.isOpen) {
            this.close();
        } else {
            this.open();
        }
    },

    open() {
        this.isOpen = true;
        this.elements.panel?.classList.add('open');
        this.elements.fab?.classList.add('active');
        this.elements.backdrop?.classList.add('visible');
        this.positionPanel();
        if (!this.isMobileView()) {
            this.elements.input?.focus();
        }
    },

    close() {
        this.isOpen = false;
        this.elements.panel?.classList.remove('open');
        this.elements.fab?.classList.remove('active');
        this.elements.backdrop?.classList.remove('visible');
        this.resetPanelTransform();
    },

    clearContext() {
        this.selectedContext = null;
        if (this.elements.contextBlock) {
            this.elements.contextBlock.style.display = 'none';
        }
    },

    positionPanel() {
        const fab = this.elements.fab;
        const panel = this.elements.panel;
        if (!fab || !panel) return;

        if (this.isMobileView()) {
            panel.style.left = '0';
            panel.style.right = '0';
            panel.style.bottom = '0';
            panel.style.top = 'auto';
            return;
        }

        const rect = fab.getBoundingClientRect();
        const panelWidth = panel.offsetWidth || 400;

        // Position strictly below the FAB
        let left = rect.left + (rect.width / 2) - (panelWidth / 2);
        let top = rect.bottom + 15;

        // Keep within horizontal viewport
        if (left + panelWidth > window.innerWidth - 20) {
            left = window.innerWidth - panelWidth - 20;
        }
        if (left < 20) left = 20;

        // Vertically, if it goes off bottom, we move it up a bit but keep it starting below or at FAB bottom
        if (top + panel.offsetHeight > window.innerHeight - 20) {
            top = window.innerHeight - panel.offsetHeight - 20;
            // Ensure it doesn't cover the FAB if possible
            if (top < rect.bottom) top = rect.bottom + 5;
        }

        panel.style.left = `${left}px`;
        panel.style.top = `${top}px`;
        panel.style.right = 'auto';
        panel.style.bottom = 'auto';
    },

    resetPanelTransform() {
        if (!this.elements.panel) return;
        this.elements.panel.style.transform = '';
        this.elements.panel.style.opacity = '';
    },

    isMobileView() {
        return window.matchMedia('(max-width: 480px)').matches;
    },

    toggleVoice() {
        if (!this.recognition) return;

        if (this.isListening) {
            this.recognition.stop();
        } else {
            this.recognition.start();
        }
    },

    handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            const base64 = e.target.result;
            this.pendingImage = {
                data: base64,
                name: file.name,
                type: file.type
            };

            // Show image preview in input area
            this.showImagePreview(base64);
        };
        reader.readAsDataURL(file);

        // Clear the input so same file can be selected again
        event.target.value = '';
    },

    async handlePaste(event) {
        const items = event.clipboardData?.items;
        if (!items) return;

        for (const item of items) {
            if (item.type && item.type.startsWith('image/')) {
                const file = item.getAsFile();
                if (file) {
                    const base64 = await this.fileToBase64(file);
                    this.pendingImage = {
                        data: base64,
                        name: file.name || 'pasted-image',
                        type: file.type
                    };
                    this.showImagePreview(base64);
                    event.preventDefault();
                    break;
                }
            }
        }
    },

    fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    },

    showImagePreview(base64) {
        const container = this.elements.attachments;
        if (!container) return;

        container.innerHTML = '';

        const preview = document.createElement('div');
        preview.className = 'copilot-attachment';
        preview.innerHTML = `
            <img src="${base64}" alt="Upload preview">
            <button class="remove" title="Remove image">×</button>
        `;

        preview.querySelector('.remove')?.addEventListener('click', () => {
            this.pendingImage = null;
            preview.remove();
        });

        container.appendChild(preview);
    },

    addMessage(role, content, image = null) {
        const container = this.elements.messagesContainer;
        if (!container) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `copilot-message ${role}`;

        let messageContent = '';

        if (image) {
            messageContent += `<img src="${image}" alt="Attached image" class="message-image">`;
        }

        // Convert markdown-like formatting
        const formattedContent = this.formatMessage(content);
        messageContent += `<div class="message-content">${formattedContent}</div>`;

        messageDiv.innerHTML = messageContent;
        container.appendChild(messageDiv);

        // Scroll to bottom
        container.scrollTop = container.scrollHeight;
    },

    formatMessage(content) {
        // Basic markdown formatting
        return content
            .replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            .replace(/\*([^*]+)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');
    },

    async sendMessage() {
        const input = this.elements.input;
        const text = input?.value.trim();

        if (!text && !this.pendingImage) return;

        // Build message with context
        let userDisplayMessage = text;
        let fullApiMessage = text;

        if (this.selectedContext) {
            const contextInfo = this.selectedContext.startLine
                ? `Lines ${this.selectedContext.startLine}-${this.selectedContext.endLine}`
                : "Selected text";
            // Truncate context if it's too large for the message
            const truncatedText = this.selectedContext.text.length > 10000
                ? this.selectedContext.text.slice(0, 10000) + "... [truncated]"
                : this.selectedContext.text;
            fullApiMessage = `[Context (${contextInfo}): "${truncatedText}"]\n\n${text}`;
        }

        // Add user message to chat
        this.addMessage('user', text, this.pendingImage?.data);

        // Clear input
        input.value = '';
        this.elements.attachments.innerHTML = '';
        const imageToUpload = this.pendingImage;
        this.pendingImage = null;

        // Prepare message container for streaming
        const messageDiv = document.createElement('div');
        messageDiv.className = 'copilot-message assistant';
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        messageDiv.appendChild(contentDiv);
        this.elements.messagesContainer.appendChild(messageDiv);

        let assistantResponse = '';

        try {
            const messages = this.buildMessages(fullApiMessage);

            await this.callMLLMStream(messages, imageToUpload, (chunk) => {
                assistantResponse += chunk;
                contentDiv.innerHTML = this.formatMessage(assistantResponse);
                this.elements.messagesContainer.scrollTop = this.elements.messagesContainer.scrollHeight;
            });

            // Update history
            this.conversationHistory.push({ role: 'assistant', content: assistantResponse });

            // Clear context after successful send
            this.clearContext();

        } catch (error) {
            console.error('MLLM API error:', error);
            contentDiv.innerHTML = `<span style="color:var(--error)">Error: ${error.message}</span>`;
        }
    },

    buildMessages(userMessage) {
        // System message is always included
        const sysMsg = {
            role: 'system',
            content: 'You are a helpful LaTeX and academic writing assistant.'
        };

        // Estimate character count (as a proxy for tokens)
        // Max context ~8000 tokens ≈ 32,000 chars. 
        // We limit input to ~15,000 chars to be safe.
        const CHAR_LIMIT = 15000;

        let messages = [];
        let totalChars = sysMsg.content.length + userMessage.length;

        // Add history in reverse until we hit the char limit
        const historyToInclude = [];
        for (let i = this.conversationHistory.length - 1; i >= 0; i--) {
            const msg = this.conversationHistory[i];
            const msgLen = (msg.content || "").length;
            if (totalChars + msgLen > CHAR_LIMIT) {
                // Remove older ones from global history too to "dump" them
                this.conversationHistory = this.conversationHistory.slice(i + 1);
                break;
            }
            historyToInclude.unshift(msg);
            totalChars += msgLen;
        }

        messages.push(sysMsg);
        messages.push(...historyToInclude);
        messages.push({ role: 'user', content: userMessage });

        // Update global history
        this.conversationHistory.push({ role: 'user', content: userMessage });

        return messages;
    },

    async callMLLMStream(messages, image = null, onChunk) {
        if (!this.llmBase) {
            this.loadLlmSettings();
        }

        const apiUrl = this.resolveLlmEndpoint(this.llmBase, 'chat');
        const model = this.llmModel || 'gpt-4o-mini';

        const requestBody = {
            model: model,
            messages: messages,
            max_tokens: 2048,
            temperature: 0.7,
            stream: true
        };

        if (image) {
            const lastMessage = requestBody.messages[requestBody.messages.length - 1];
            lastMessage.content = [
                { type: 'image_url', image_url: { url: image.data } },
                { type: 'text', text: lastMessage.content || "" }
            ];
        }

        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) throw new Error(`API request failed: ${response.status}`);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                const cleanLine = line.trim();
                if (!cleanLine || cleanLine === 'data: [DONE]') continue;
                if (cleanLine.startsWith('data: ')) {
                    try {
                        const json = JSON.parse(cleanLine.slice(6));
                        const content = json.choices?.[0]?.delta?.content;
                        if (content) onChunk(content);
                    } catch (e) {
                        console.warn("Error parsing stream chunk", e);
                    }
                }
            }
        }
    },

    setupPanelResizer() {
        const resizer = this.elements.resizer;
        const panel = this.elements.panel;
        if (!resizer || !panel) return;

        let isResizing = false;
        let startX, startY, startWidth, startHeight, startLeft, startTop;

        resizer.addEventListener('mousedown', (e) => {
            isResizing = true;
            startX = e.clientX;
            startY = e.clientY;
            startWidth = panel.offsetWidth;
            startHeight = panel.offsetHeight;
            const rect = panel.getBoundingClientRect();
            startLeft = rect.left;
            startTop = rect.top;

            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', () => {
                isResizing = false;
                document.removeEventListener('mousemove', handleMouseMove);
            });
            e.preventDefault();
        });

        const handleMouseMove = (e) => {
            if (!isResizing) return;
            const deltaX = startX - e.clientX;
            const deltaY = startY - e.clientY;

            const newWidth = startWidth + deltaX;
            const newHeight = startHeight + deltaY;

            if (newWidth > 300) {
                panel.style.width = newWidth + 'px';
                panel.style.left = (startLeft - deltaX) + 'px';
            }
            if (newHeight > 350) {
                panel.style.height = newHeight + 'px';
                panel.style.top = (startTop - deltaY) + 'px';
            }
        };
    },

    addLoadingMessage() {
        const container = this.elements.messagesContainer;
        if (!container) return null;

        const id = 'loading-' + Date.now();
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'copilot-message assistant loading';
        loadingDiv.id = id;
        loadingDiv.innerHTML = `
            <div class="loading-dots">
                <span></span><span></span><span></span>
            </div>
        `;
        container.appendChild(loadingDiv);
        container.scrollTop = container.scrollHeight;

        return id;
    },

    removeLoadingMessage(id) {
        if (!id) return;
        document.getElementById(id)?.remove();
    },

    clearChat() {
        if (this.elements.messagesContainer) {
            this.elements.messagesContainer.innerHTML = '';
        }
        this.conversationHistory = [];
        this.addMessage('assistant', 'Chat cleared. How can I help you?');
    }
};

function setupEditor() {
    EditorController.init();
    CopilotController.init();
    TeamChatController.init();
}
