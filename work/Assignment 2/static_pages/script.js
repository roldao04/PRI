// API Base URL
const API_BASE = '/api/v1';

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadStatistics();
    checkEmbeddingStatus();  // NEW: Check if embeddings are loaded

    // Allow Enter key to trigger search
    document.getElementById('searchInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            performSearch();
        }
    });

    // Toggle reranking model selector when checkbox is changed
    document.getElementById('useReranking').addEventListener('change', (e) => {
        const rerankingOptions = document.getElementById('rerankingOptions');
        rerankingOptions.style.display = e.target.checked ? 'block' : 'none';

        // Preload model when reranking is enabled
        if (e.target.checked) {
            const modelName = document.getElementById('rerankModel').value;
            preloadModel(modelName);
        }
    });

    // Preload model when dropdown selection changes
    document.getElementById('rerankModel').addEventListener('change', (e) => {
        const modelName = e.target.value;
        preloadModel(modelName);
    });

});

// Toggle advanced options panel
function toggleAdvancedOptions() {
    const panel = document.getElementById('advancedPanel');
    panel.classList.toggle('hidden');
}

// Toggle BM25 parameters subsection
function toggleBM25Params() {
    const panel = document.getElementById('bm25Panel');
    const arrow = document.getElementById('bm25Arrow');

    panel.classList.toggle('hidden');

    // Rotate arrow
    if (panel.classList.contains('hidden')) {
        arrow.textContent = '▶';
    } else {
        arrow.textContent = '▼';
    }
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
            <div class="stat-value">BM25F + Neural</div>
            <div class="stat-label">Algoritmo de Ranking</div>
        </div>
        <div class="stat-item" id="embeddingStatusStat">
            <div class="stat-value">
                <span class="loading-indicator">⏳</span>
            </div>
            <div class="stat-label">Embeddings</div>
        </div>
    `;
}

// Check embedding status and update UI
async function checkEmbeddingStatus() {
    try {
        const response = await fetch(`${API_BASE}/embeddings/status`);
        if (!response.ok) {
            console.warn('Failed to check embedding status');
            updateEmbeddingStatusUI({ status: 'unknown', is_loaded: false });
            return;
        }

        const data = await response.json();
        console.log('Embedding status:', data);
        updateEmbeddingStatusUI(data);
    } catch (error) {
        console.error('Error checking embedding status:', error);
        updateEmbeddingStatusUI({ status: 'error', is_loaded: false });
    }
}

// Update embedding status in UI
function updateEmbeddingStatusUI(status) {
    const statDiv = document.getElementById('embeddingStatusStat');
    if (!statDiv) return;

    if (status.is_loaded) {
        const memoryMB = status.memory_mb || 0;
        const memoryGB = (memoryMB / 1024).toFixed(1);
        statDiv.innerHTML = `
            <div class="stat-value" style="color: #4CAF50;">✓ ${memoryGB}GB</div>
            <div class="stat-label">Embeddings ${status.has_mapping ? '(Mapped)' : ''}</div>
        `;
    } else {
        statDiv.innerHTML = `
            <div class="stat-value" style="color: #ff9800;">⚠</div>
            <div class="stat-label">Embeddings N/D</div>
        `;

        // Disable embedding similarity method if not loaded
        const similaritySelect = document.getElementById('similarityMethod');
        if (similaritySelect) {
            const embeddingOption = Array.from(similaritySelect.options).find(opt => opt.value === 'embedding');
            if (embeddingOption) {
                embeddingOption.disabled = true;
                embeddingOption.text = 'Semântico (Embeddings) - Não disponível';
            }
        }
    }
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
    const useReranking = document.getElementById('useReranking').checked;
    const rerankModel = document.getElementById('rerankModel').value;
    const generateAnswers = document.getElementById('generateAnswers').checked;
    const expandQuery = document.getElementById('expandQuery').checked;
    const extractSnippets = document.getElementById('extractSnippets').checked;

    // Show loading
    showLoading();
    hideSearchInfo();
    clearResults();
    clearAnswerSnippets();

    try {
        const params = new URLSearchParams({
            query: query,
            num_results: numResults,
            k1: k1,
            b: b,
            min_score: minScore
        });

        // Only add reranking params if enabled
        if (useReranking) {
            params.append('rerank', 'true');
            params.append('reranker_model', rerankModel);
        } else {
            params.append('rerank', 'false');
        }

        // Only add answer generation params if enabled
        if (generateAnswers) {
            params.append('generate_answers', 'true');
        } else {
            params.append('generate_answers', 'false');
        }

        // Only add query expansion params if enabled
        if (expandQuery) {
            params.append('expand_query', 'true');
        } else {
            params.append('expand_query', 'false');
        }

        // Only add snippet extraction params if enabled
        if (extractSnippets) {
            params.append('extract_snippets', 'true');
            // Note: snippet extraction requires a reranker model
            // If reranking is not enabled, still use the model for snippet scoring
            if (!useReranking) {
                params.append('reranker_model', rerankModel);
            }
        } else {
            params.append('extract_snippets', 'false');
        }

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
    const similarityMethod = document.getElementById('similarityMethod').value;

    showLoading();
    hideSearchInfo();
    clearResults();
    clearAnswerSnippets();

    try {
        const params = new URLSearchParams({
            doc_id: docId,
            num_results: numResults,
            min_score: minScore,
            similarity_method: similarityMethod
        });

        const response = await fetch(`${API_BASE}/search_similar?${params}`);

        if (!response.ok) {
            throw new Error(`Similar search failed: ${response.statusText}`);
        }

        const data = await response.json();

        hideLoading();

        // Create custom message with similarity method info
        const methodLabels = {
            'embedding': 'Semântico (Embeddings)',
            'hybrid': 'Híbrido (TF-IDF + Campos)',
            'cosine': 'Coseno TF-IDF',
            'field-weighted': 'TF-IDF Ponderado',
            'jaccard': 'Jaccard'
        };
        const methodLabel = methodLabels[similarityMethod] || similarityMethod;

        displayResults(data, null, `Documentos similares a: "${docTitle}" (Método: ${methodLabel})`);
    } catch (error) {
        console.error('Similar search error:', error);
        hideLoading();

        // Check if error is related to missing embeddings
        if (similarityMethod === 'embedding' && error.message.includes('500')) {
            showError(
                'Erro: Embeddings não disponíveis. ' +
                'Para usar similaridade semântica, execute: python scripts/generate_embeddings.py'
            );
        } else {
            showError('Erro ao buscar documentos similares. Por favor, tente novamente.');
        }
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
    // Clear previous content and set new info text (using innerHTML to allow child elements)
    searchInfoDiv.innerHTML = `<p style="margin: 0;">${infoText}</p>`;
    searchInfoDiv.classList.remove('hidden');

    // Show query expansion info if used
    if (data.query_expansion_used && data.expanded_terms && data.expanded_terms.length > 0) {
        const expansionDiv = document.createElement('div');
        expansionDiv.className = 'query-expansion-info';
        expansionDiv.innerHTML = `
            <strong>📝 Consulta original:</strong> "${data.original_query}"<br>
            <strong>➕ Termos adicionados:</strong> ${data.expanded_terms.join(', ')}
        `;
        searchInfoDiv.appendChild(expansionDiv);
    }

    // Display AI answer snippets if available
    if (data.answer_snippets && data.answer_snippets.length > 0) {
        displayAnswerSnippets(data.answer_snippets);
    }

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

    resultsDiv.innerHTML = data.results.map(result => {
        // Show both scores if reranking was used
        let scoreDisplay = '';
        if (typeof result.neural_score === 'number' && typeof result.bm25_score === 'number') {
            scoreDisplay = `
                <div class="result-score">
                    <div>Neural: ${result.neural_score.toFixed(2)}</div>
                    <div style="font-size: 0.85em; opacity: 0.7;">BM25: ${result.bm25_score.toFixed(2)}</div>
                </div>
            `;
        } else {
            scoreDisplay = `<div class="result-score">Score: ${result.score.toFixed(2)}</div>`;
        }

        return `
            <div class="result-item" onclick="showDocumentModal(${result.id})">
                <div class="result-header">
                    <div>
                        <h2 class="result-title">
                            ${escapeHtml(result.title)}
                        </h2>
                        <p class="result-content">${escapeHtml(result.content)}</p>
                        ${result.url ? `<a href="${result.url}" class="result-url" target="_blank" onclick="event.stopPropagation()">${result.url}</a>` : ''}
                    </div>
                    ${scoreDisplay}
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
        `;
    }).join('');
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

// Clear answer snippets
function clearAnswerSnippets() {
    const snippetsDiv = document.getElementById('answerSnippets');
    snippetsDiv.innerHTML = '';
    snippetsDiv.classList.add('hidden');
}

// Display AI answer snippets
function displayAnswerSnippets(snippets) {
    const snippetsDiv = document.getElementById('answerSnippets');

    snippetsDiv.innerHTML = snippets.map((snippet, index) => {
        if (!snippet.success) {
            return `
                <div class="answer-snippet error">
                    <div class="answer-header">
                        <span class="answer-icon">⚠️</span>
                        <span class="answer-title">Erro ao gerar resposta</span>
                    </div>
                    <p class="answer-text">${escapeHtml(snippet.error || 'Erro desconhecido')}</p>
                </div>
            `;
        }

        // Build sources display
        let sourcesHtml = '';
        if (snippet.sources_used && snippet.sources_used.length > 0) {
            const sourceButtons = snippet.sources_used.map(docId =>
                `<button class="source-doc-button" onclick="showDocumentModal(${docId})" title="Ver documento ${docId}">
                    Doc ${docId}
                </button>`
            ).join('');

            sourcesHtml = `
                <div class="answer-sources">
                    <span class="sources-label">📚 Fontes utilizadas:</span>
                    <div class="sources-buttons">${sourceButtons}</div>
                </div>
            `;
        } else {
            // Fallback to old behavior if no sources provided
            sourcesHtml = `
                <div class="answer-sources">
                    <span class="sources-label">📚 Baseado em:</span>
                    <button class="source-doc-button" onclick="showDocumentModal(${snippet.doc_id})">
                        Doc ${snippet.doc_id}
                    </button>
                </div>
            `;
        }

        return `
            <div class="answer-snippet">
                <div class="answer-header">
                    <span class="answer-icon">✨</span>
                    <span class="answer-title">Resposta Gerada por IA</span>
                    ${snippets.length > 1 ? `<span class="answer-index">#${index + 1}</span>` : ''}
                </div>
                <p class="answer-text">${escapeHtml(snippet.answer)}</p>
                ${sourcesHtml}
            </div>
        `;
    }).join('');

    snippetsDiv.classList.remove('hidden');
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

        // Check if user has a query (to show paragraph scores)
        const currentQuery = document.getElementById('searchInput').value.trim();
        const showScoresButton = currentQuery ? `
            <button onclick="showParagraphScores(${docId}, '${escapeHtml(currentQuery).replace(/'/g, "\\'")}')">
                📊 Ver Scores por Parágrafo
            </button>
        ` : '';

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
                        <div class="modal-text" id="modalTextContent">${escapeHtml(doc.content).replace(/\n/g, '<br>')}</div>
                    </div>
                    <div class="modal-footer">
                        ${showScoresButton}
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

// Show paragraph scores for a document
async function showParagraphScores(docId, query) {
    const modalTextContent = document.getElementById('modalTextContent');
    if (!modalTextContent) return;

    try {
        // Show loading in modal
        modalTextContent.innerHTML = '<p style="text-align: center;">📊 A carregar scores dos parágrafos...</p>';

        const rerankModel = document.getElementById('rerankModel').value;
        const params = new URLSearchParams({
            query: query,
            reranker_model: rerankModel
        });

        const response = await fetch(`${API_BASE}/document/${docId}/paragraph_scores?${params}`);

        if (!response.ok) {
            throw new Error(`Failed to fetch paragraph scores: ${response.statusText}`);
        }

        const data = await response.json();

        // Build HTML with paragraph scores
        let html = `
            <div style="margin-bottom: 20px; padding: 15px; background: #f5f5f5; border-radius: 8px;">
                <h3 style="margin: 0 0 10px 0;">📊 Scores de Relevância por Parágrafo</h3>
                <p style="margin: 5px 0;"><strong>Query:</strong> "${escapeHtml(data.query)}"</p>
                <p style="margin: 5px 0;"><strong>Melhor Score:</strong> ${data.best_score.toFixed(3)}</p>
                <p style="margin: 5px 0;"><strong>Total de Parágrafos:</strong> ${data.num_paragraphs}</p>
                <p style="margin: 10px 0 5px 0; font-size: 0.9em; opacity: 0.8;">
                    ℹ️ Scores mais altos indicam maior relevância para a query
                </p>
            </div>
        `;

        // Add each paragraph with its score
        data.paragraph_scores.forEach((para, index) => {
            const score = para.score;
            const isBest = index === 0; // Assuming sorted by score

            // Color coding based on score
            let scoreColor = '#666';
            let bgColor = '#fff';
            if (score > 5) {
                scoreColor = '#4CAF50';
                bgColor = '#e8f5e9';
            } else if (score > 2) {
                scoreColor = '#ff9800';
                bgColor = '#fff3e0';
            }

            html += `
                <div style="
                    margin: 15px 0;
                    padding: 12px;
                    border-left: 4px solid ${scoreColor};
                    background: ${bgColor};
                    border-radius: 4px;
                    ${isBest ? 'box-shadow: 0 2px 8px rgba(76, 175, 80, 0.3);' : ''}
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: bold; color: ${scoreColor};">
                            ${isBest ? '⭐ ' : ''}Parágrafo #${index + 1}
                        </span>
                        <span style="
                            background: ${scoreColor};
                            color: white;
                            padding: 4px 12px;
                            border-radius: 12px;
                            font-weight: bold;
                            font-size: 0.9em;
                        ">
                            Score: ${score.toFixed(3)}
                        </span>
                    </div>
                    <div style="color: #333; line-height: 1.6;">
                        ${escapeHtml(para.text)}
                    </div>
                </div>
            `;
        });

        modalTextContent.innerHTML = html;

    } catch (error) {
        console.error('Error loading paragraph scores:', error);
        modalTextContent.innerHTML = `
            <p style="color: #f44336;">
                ⚠️ Erro ao carregar scores dos parágrafos: ${error.message}
            </p>
        `;
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

// Preload a neural reranking model
async function preloadModel(modelName) {
    try {
        console.log(`🔄 Preloading model: ${modelName}`);

        const response = await fetch(`${API_BASE}/preload_model/${modelName}`);

        if (!response.ok) {
            console.warn(`Failed to preload model ${modelName}: ${response.statusText}`);
            return;
        }

        const data = await response.json();
        console.log(`Model ${modelName} preloaded:`, data);
    } catch (error) {
        console.error(`Error preloading model ${modelName}:`, error);
        // Don't show error to user - preloading is a performance optimization
        // and should fail silently to not disrupt the user experience
    }
}
