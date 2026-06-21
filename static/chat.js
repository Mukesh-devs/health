/**
 * Chat Application Frontend
 * Supports UUID-based public session URLs
 */

// ───────────────────────────────────────────────
// Configuration
// ───────────────────────────────────────────────
const API_BASE = '/api';
let currentSessionId = null;      // Internal integer ID (used for API calls)
let currentPublicId = null;       // Public UUID (used for URLs)
let authToken = localStorage.getItem('access_token');

// ───────────────────────────────────────────────
// URL Parsing
// ───────────────────────────────────────────────
function parsePublicIdFromUrl() {
    const pathParts = window.location.pathname.split('/');
    // URL format: /chat/<public_id>
    if (pathParts.length > 2 && pathParts[1] === 'chat') {
        const potentialUuid = pathParts[2];
        // Validate UUID format
        const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
        if (uuidRegex.test(potentialUuid)) {
            return potentialUuid;
        }
    }
    return null;
}

// ───────────────────────────────────────────────
// Session Management
// ───────────────────────────────────────────────
async function loadSessionList() {
    try {
        const response = await fetch(`${API_BASE}/sessions`, {
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        if (!response.ok) throw new Error('Failed to load sessions');

        const sessions = await response.json();
        renderSessionList(sessions);
        return sessions;
    } catch (error) {
        console.error('Error loading sessions:', error);
        return [];
    }
}

function renderSessionList(sessions) {
    const sessionListEl = document.getElementById('session-list');
    if (!sessionListEl) return;

    sessionListEl.innerHTML = '';

    sessions.forEach(session => {
        const sessionEl = document.createElement('div');
        sessionEl.className = 'session-item';
        sessionEl.dataset.publicId = session.public_id;
        sessionEl.dataset.sessionId = session.id;

        if (session.public_id === currentPublicId) {
            sessionEl.classList.add('active');
        }

        sessionEl.innerHTML = `
            <div class="session-title">${escapeHtml(session.title)}</div>
            <div class="session-meta">${session.message_count} messages</div>
        `;

        sessionEl.addEventListener('click', () => {
            navigateToSession(session.public_id);
        });

        sessionListEl.appendChild(sessionEl);
    });
}

function navigateToSession(publicId) {
    window.history.pushState({}, '', `/chat/${publicId}`);
    currentPublicId = publicId;
    loadSessionByPublicId(publicId);
}

async function loadSessionByPublicId(publicId) {
    try {
        // Fetch session details by public_id
        const response = await fetch(`${API_BASE}/sessions/${publicId}`, {
            headers: { 'Authorization': `Bearer ${authToken}` }
        });

        if (response.status === 404) {
            // Session not found or not owned by user
            showError('Session not found or access denied.');
            window.history.pushState({}, '', '/chat');
            currentPublicId = null;
            currentSessionId = null;
            clearChat();
            return;
        }

        if (!response.ok) throw new Error('Failed to load session');

        const session = await response.json();
        currentSessionId = session.id;
        currentPublicId = session.public_id;

        // Load chat history for this session
        await loadChatHistory(publicId);

        // Highlight active session in sidebar
        highlightActiveSession(publicId);

    } catch (error) {
        console.error('Error loading session:', error);
        showError('Failed to load session.');
    }
}

async function loadChatHistory(publicId) {
    try {
        const response = await fetch(`${API_BASE}/sessions/${publicId}/history`, {
            headers: { 'Authorization': `Bearer ${authToken}` }
        });

        if (!response.ok) throw new Error('Failed to load history');

        const history = await response.json();
        renderChatHistory(history);

    } catch (error) {
        console.error('Error loading chat history:', error);
    }
}

function renderChatHistory(history) {
    const chatContainer = document.getElementById('chat-messages');
    if (!chatContainer) return;

    chatContainer.innerHTML = '';

    history.forEach(item => {
        appendUserMessage(item.query);

        const aiResponse = item.response;
        if (aiResponse.predefinedResponse) {
            appendAIMessage(aiResponse.predefinedResponse);
        } else if (aiResponse.textualResponse) {
            appendAIMessage(aiResponse.textualResponse, aiResponse);
        }
    });

    scrollToBottom();
}

// ───────────────────────────────────────────────
// New Session Flow
// ───────────────────────────────────────────────
async function createNewSession() {
    try {
        const response = await fetch(`${API_BASE}/sessions`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${authToken}`,
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) throw new Error('Failed to create session');

        const session = await response.json();
        currentSessionId = session.id;
        currentPublicId = session.public_id;

        // Update URL without page reload
        window.history.pushState({}, '', `/chat/${session.public_id}`);

        // Refresh session list
        await loadSessionList();
        highlightActiveSession(session.public_id);

        return session;

    } catch (error) {
        console.error('Error creating session:', error);
        showError('Failed to create new session.');
        return null;
    }
}

// ───────────────────────────────────────────────
// Message Sending
// ───────────────────────────────────────────────
async function sendMessage(query, answerLength = 'Moderate') {
    if (!query.trim()) return;

    // If no session exists, create one first
    if (!currentSessionId) {
        const session = await createNewSession();
        if (!session) return;
    }

    appendUserMessage(query);
    showLoadingIndicator();

    try {
        const response = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${authToken}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                session_id: currentSessionId,
                answerLength: answerLength
            })
        });

        if (!response.ok) throw new Error('Query failed');

        const data = await response.json();
        hideLoadingIndicator();

        if (data.predefinedResponse) {
            appendAIMessage(data.predefinedResponse);
        } else {
            appendAIMessage(data.textualResponse, data);
            if (data.graphData) renderGraph(data.graphData);
            if (data.validatedTriples) renderEvidence(data.validatedTriples);
            if (data.recommendations) renderRecommendations(data.recommendations);
        }

        // Refresh session list to update message count and title
        await loadSessionList();

    } catch (error) {
        console.error('Error sending message:', error);
        hideLoadingIndicator();
        showError('Failed to send message. Please try again.');
    }
}

// ───────────────────────────────────────────────
// UI Helpers
// ───────────────────────────────────────────────
function appendUserMessage(text) {
    const container = document.getElementById('chat-messages');
    if (!container) return;

    const msgEl = document.createElement('div');
    msgEl.className = 'message user-message';
    msgEl.textContent = text;
    container.appendChild(msgEl);
    scrollToBottom();
}

function appendAIMessage(text, data = null) {
    const container = document.getElementById('chat-messages');
    if (!container) return;

    const msgEl = document.createElement('div');
    msgEl.className = 'message ai-message';
    msgEl.innerHTML = formatResponse(text);
    container.appendChild(msgEl);
    scrollToBottom();
}

function formatResponse(text) {
    // Convert markdown-like formatting
    return escapeHtml(text)
        .replace(/\*\|([^|]+)\|\*/g, '<span class="entity-highlight">$1</span>');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function scrollToBottom() {
    const container = document.getElementById('chat-messages');
    if (container) container.scrollTop = container.scrollHeight;
}

function showLoadingIndicator() {
    const container = document.getElementById('chat-messages');
    if (!container) return;

    const loader = document.createElement('div');
    loader.id = 'loading-indicator';
    loader.className = 'loading-indicator';
    loader.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
    container.appendChild(loader);
    scrollToBottom();
}

function hideLoadingIndicator() {
    const loader = document.getElementById('loading-indicator');
    if (loader) loader.remove();
}

function showError(message) {
    // Simple error display - can be enhanced
    console.error(message);
    const container = document.getElementById('chat-messages');
    if (container) {
        const errorEl = document.createElement('div');
        errorEl.className = 'message error-message';
        errorEl.textContent = message;
        container.appendChild(errorEl);
        scrollToBottom();
    }
}

function clearChat() {
    const container = document.getElementById('chat-messages');
    if (container) container.innerHTML = '';
}

function highlightActiveSession(publicId) {
    document.querySelectorAll('.session-item').forEach(el => {
        el.classList.toggle('active', el.dataset.publicId === publicId);
    });
}

// ───────────────────────────────────────────────
// Graph & Evidence Rendering (placeholders)
// ───────────────────────────────────────────────
function renderGraph(graphData) {
    // Implementation depends on your graph visualization library
    console.log('Graph data:', graphData);
}

function renderEvidence(triples) {
    // Implementation for evidence panel
    console.log('Evidence triples:', triples);
}

function renderRecommendations(recommendations) {
    // Implementation for recommendation chips
    console.log('Recommendations:', recommendations);
}

// ───────────────────────────────────────────────
// Event Listeners & Initialization
// ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    // Check authentication
    if (!authToken) {
        window.location.href = '/login';
        return;
    }

    // Load session list
    await loadSessionList();

    // Check URL for public_id
    const publicId = parsePublicIdFromUrl();
    if (publicId) {
        currentPublicId = publicId;
        await loadSessionByPublicId(publicId);
    }

    // Setup send button
    const sendBtn = document.getElementById('send-btn');
    const inputEl = document.getElementById('message-input');
    const answerLengthSelect = document.getElementById('answer-length');

    if (sendBtn && inputEl) {
        sendBtn.addEventListener('click', () => {
            const query = inputEl.value.trim();
            const answerLength = answerLengthSelect ? answerLengthSelect.value : 'Moderate';
            if (query) {
                sendMessage(query, answerLength);
                inputEl.value = '';
            }
        });

        inputEl.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendBtn.click();
            }
        });
    }

    // Setup new chat button
    const newChatBtn = document.getElementById('new-chat-btn');
    if (newChatBtn) {
        newChatBtn.addEventListener('click', async () => {
            currentSessionId = null;
            currentPublicId = null;
            clearChat();
            window.history.pushState({}, '', '/chat');
            highlightActiveSession(null);
        });
    }

    // Handle browser back/forward buttons
    window.addEventListener('popstate', async () => {
        const publicId = parsePublicIdFromUrl();
        if (publicId) {
            currentPublicId = publicId;
            await loadSessionByPublicId(publicId);
        } else {
            currentSessionId = null;
            currentPublicId = null;
            clearChat();
            highlightActiveSession(null);
        }
    });
});