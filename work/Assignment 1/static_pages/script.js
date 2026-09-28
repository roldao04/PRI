// API Base URL
const API_BASE = '/api/v1';

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadStatistics();

    // Allow Enter key to trigger search
    document.getElementById('searchInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            performSearch();
        }
    });
});

// Toggle advanced options panel
function toggleAdvancedOptions() {
    const panel = document.getElementById('advancedPanel');
    panel.classList.toggle('hidden');
}

// Load index statistics
async function loadStatistics() {
    try {
        const response = await fetch(`${API_BASE}/stats`);
        if (!response.ok) {
            throw new Error('Failed to load statistics');
        }

        const stats = await response.json();
        displayStatistics(stats);
    } catch (error) {
        console.error('Error loading statistics:', error);
        document.getElementById('stats').innerHTML = '<p>Erro ao carregar estatísticas</p>';
    }
}

// Display statistics
function displayStatistics(stats) {
    const statsDiv = document.getElementById('stats');
    statsDiv.innerHTML = `
        <div class="stat-item">
            <div class="stat-value">${stats.num_documents.toLocaleString('pt-PT')}</div>
            <div class="stat-label">Documentos</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${stats.num_terms.toLocaleString('pt-PT')}</div>
            <div class="stat-label">Termos Únicos</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${stats.avg_doc_length.toFixed(1)}</div>
            <div class="stat-label">Tamanho Médio</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">BM25</div>
            <div class="stat-label">Algoritmo de Ranking</div>
        </div>
    `;
}

// Perform search
async function performSearch() {
    const query = document.getElementById('searchInput').value.trim();

    if (!query) {
        alert('Por favor, insira uma consulta de pesquisa.');
        return;
    }

    // Get parameters
    const numResults = document.getElementById('numResults').value;
    const k1 = document.getElementById('k1Param').value;
    const b = document.getElementById('bParam').value;
    const minScore = document.getElementById('minScore').value;

    // Show loading
    showLoading();
    hideSearchInfo();
    clearResults();

    try {
        const params = new URLSearchParams({
            query: query,
            num_results: numResults,
            k1: k1,
            b: b,
            min_score: minScore
        });

        const startTime = performance.now();
        const response = await fetch(`${API_BASE}/search?${params}`);

        if (!response.ok) {
            throw new Error(`Search failed: ${response.statusText}`);
        }

        const data = await response.json();
        const endTime = performance.now();

        hideLoading();
        displayResults(data, endTime - startTime);
    } catch (error) {
        console.error('Search error:', error);
        hideLoading();
        showError('Erro ao realizar a pesquisa. Por favor, tente novamente.');
    }
}

// Search for similar documents
async function searchSimilar(docId, docTitle) {
    const numResults = document.getElementById('numResults').value;
    const minScore = document.getElementById('minScore').value;

    showLoading();
    hideSearchInfo();
    clearResults();

    try {
        const params = new URLSearchParams({
            doc_id: docId,
            num_results: numResults,
            min_score: minScore
        });

        const response = await fetch(`${API_BASE}/search_similar?${params}`);

        if (!response.ok) {
            throw new Error(`Similar search failed: ${response.statusText}`);
        }

        const data = await response.json();

        hideLoading();
        displayResults(data, null, `Documentos similares a: "${docTitle}"`);
    } catch (error) {
        console.error('Similar search error:', error);
        hideLoading();
        showError('Erro ao buscar documentos similares.');
    }
}

// Display search results
function displayResults(data, clientTime = null, customMessage = null) {
    const resultsDiv = document.getElementById('results');
    const searchInfoDiv = document.getElementById('searchInfo');

    // Show search info
    let infoText = customMessage || `Encontrados ${data.num_results} resultados para "${data.query}"`;
    if (data.execution_time_ms !== undefined) {
        infoText += ` (${data.execution_time_ms}ms no servidor`;
        if (clientTime) {
            infoText += `, ${clientTime.toFixed(0)}ms total`;
        }
        infoText += ')';
    }
    searchInfoDiv.textContent = infoText;
    searchInfoDiv.classList.remove('hidden');

    // Display results
    if (data.results.length === 0) {
        resultsDiv.innerHTML = `
            <div class="no-results">
                <div class="no-results-icon">🔍</div>
                <h2>Nenhum resultado encontrado</h2>
                <p>Tente usar diferentes palavras-chave ou ajustar os parâmetros de pesquisa.</p>
            </div>
        `;
        return;
    }

    resultsDiv.innerHTML = data.results.map(result => `
        <div class="result-item" onclick="showDocumentModal(${result.id})">
            <div class="result-header">
                <div>
                    <h2 class="result-title">
                        ${escapeHtml(result.title)}
                    </h2>
                    <p class="result-content">${escapeHtml(result.content)}</p>
                    ${result.url ? `<a href="${result.url}" class="result-url" target="_blank" onclick="event.stopPropagation()">${result.url}</a>` : ''}
                </div>
                <div class="result-score">Score: ${result.score.toFixed(2)}</div>
            </div>
            <div class="result-actions">
                <button class="similar-button" onclick="event.stopPropagation(); searchSimilar(${result.id}, '${escapeHtml(result.title).replace(/'/g, "\\'")}')">
                    🔗 Documentos Similares
                </button>
                <button class="similar-button" onclick="event.stopPropagation(); showDocumentModal(${result.id})">
                    📄 Ver Documento Completo
                </button>
            </div>
        </div>
    `).join('');
}

// Show loading spinner
function showLoading() {
    document.getElementById('loadingSpinner').classList.remove('hidden');
}

// Hide loading spinner
function hideLoading() {
    document.getElementById('loadingSpinner').classList.add('hidden');
}

// Show search info
function hideSearchInfo() {
    document.getElementById('searchInfo').classList.add('hidden');
}

// Clear results
function clearResults() {
    document.getElementById('results').innerHTML = '';
}

// Show error message
function showError(message) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = `
        <div class="no-results">
            <div class="no-results-icon">⚠️</div>
            <h2>Erro</h2>
            <p>${message}</p>
        </div>
    `;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Show full document in a modal
async function showDocumentModal(docId) {
    try {
        const response = await fetch(`${API_BASE}/document/${docId}`);

        if (!response.ok) {
            throw new Error(`Failed to fetch document: ${response.statusText}`);
        }

        const doc = await response.json();

        // Create modal HTML
        const modalHtml = `
            <div class="modal-overlay" onclick="closeDocumentModal()">
                <div class="modal-content" onclick="event.stopPropagation()">
                    <div class="modal-header">
                        <h2>${escapeHtml(doc.title)}</h2>
                        <button class="modal-close" onclick="closeDocumentModal()">✕</button>
                    </div>
                    <div class="modal-body">
                        ${doc.url ? `<p class="modal-url"><a href="${doc.url}" target="_blank">🔗 Ver no Wikipedia</a></p>` : ''}
                        <div class="modal-text">${escapeHtml(doc.content).replace(/\n/g, '<br>')}</div>
                    </div>
                    <div class="modal-footer">
                        <button onclick="searchSimilar(${docId}, '${escapeHtml(doc.title).replace(/'/g, "\\'")}'); closeDocumentModal()">
                            🔗 Encontrar Similares
                        </button>
                        <button onclick="closeDocumentModal()">Fechar</button>
                    </div>
                </div>
            </div>
        `;

        // Add modal to page
        const modalDiv = document.createElement('div');
        modalDiv.id = 'documentModal';
        modalDiv.innerHTML = modalHtml;
        document.body.appendChild(modalDiv);

        // Prevent body scroll
        document.body.style.overflow = 'hidden';

    } catch (error) {
        console.error('Error loading document:', error);
        alert('Erro ao carregar documento. Por favor, tente novamente.');
    }
}

// Close document modal
function closeDocumentModal() {
    const modal = document.getElementById('documentModal');
    if (modal) {
        modal.remove();
        document.body.style.overflow = 'auto';
    }
}
