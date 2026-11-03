/**
 * Semantic Cache Studio - Modern LLM Chat Frontend
 * Clean architectural separation: State, API, Markdown Parser, Chat Engine, Panels, UI Helpers.
 */

// ============================================================================
// 1. STATE & CONSTANTS
// ============================================================================
const $ = selector => document.querySelector(selector);
const $$ = selector => document.querySelectorAll(selector);

const LBL = {
  L1_exact: "L1 Exact Hit",
  L2_semantic: "L2 Semantic Hit",
  L3_llm: "L3 LLM Call"
};

const STATE = {
  currentChatId: null,
  chats: [], // [{ id, title, messages: [{ role: 'user'|'assistant', text, meta, error }] }]
  activePanel: "chat-panel",
  isStreaming: false,
  abortController: null,
  latestInspection: null,
  userScrolledUp: false
};

// ============================================================================
// 2. UTILS & MARKDOWN PARSER (Safe from XSS)
// ============================================================================
function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, c => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }[c]));
}

/**
 * Basic safe markdown rendering:
 * Escapes HTML first, then transforms code blocks, inline code, bold, lists, paragraphs.
 */
function renderMarkdown(rawText) {
  if (!rawText) return "";

  // 1. Escape HTML for strict XSS prevention
  let text = escapeHtml(rawText);

  // 2. Fenced code blocks ```lang ... ```
  text = text.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
    const language = lang.trim() || "code";
    return `
      <div class="code-block-wrapper">
        <div class="code-block-header">
          <span>${language}</span>
          <button class="code-copy-btn" onclick="copySnippet(this)">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
            <span>Copy</span>
          </button>
        </div>
        <pre><code>${code.trim()}</code></pre>
      </div>
    `;
  });

  // 3. Headings: ###, ##, #
  text = text.replace(/^### (.*$)/gim, '<h4 class="md-heading">$1</h4>');
  text = text.replace(/^## (.*$)/gim, '<h3 class="md-heading">$1</h3>');
  text = text.replace(/^# (.*$)/gim, '<h2 class="md-heading">$1</h2>');

  // 4. Inline code `...`
  text = text.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');

  // 5. Bold text **...**
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

  // 6. Italic text *...*
  text = text.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, "<em>$1</em>");

  // 7. Ordered lists: 1. 2. 3.
  text = text.replace(/(?:^|\n)(\d+\.\s+.+(?:\n\d+\.\s+.+)*)/g, match => {
    const items = match.trim().split("\n").map(li => `<li>${li.replace(/^\d+\.\s+/, "")}</li>`).join("");
    return `\n<ol class="md-list">${items}</ol>\n`;
  });

  // 8. Unordered lists: lines starting with "- " or "* "
  text = text.replace(/(?:^|\n)([-*]\s+.+(?:\n[-*]\s+.+)*)/g, match => {
    const items = match.trim().split("\n").map(li => `<li>${li.replace(/^[-*]\s+/, "")}</li>`).join("");
    return `\n<ul class="md-list">${items}</ul>\n`;
  });

  // 9. Paragraphs (split by double newlines)
  const paragraphs = text.split(/\n{2,}/).map(chunk => {
    chunk = chunk.trim();
    if (!chunk) return "";
    if (
      chunk.startsWith("<div class=\"code-block-wrapper\"") ||
      chunk.startsWith("<ul") ||
      chunk.startsWith("<ol") ||
      chunk.startsWith("<h")
    ) {
      return chunk;
    }
    return `<p>${chunk.replace(/\n/g, "<br>")}</p>`;
  });

  return paragraphs.join("");
}

function toast(message, isError = false) {
  const t = $("#toast");
  t.textContent = message;
  t.style.background = isError ? "var(--danger)" : "var(--text-primary)";
  t.style.color = isError ? "#fff" : "var(--bg-primary)";
  t.classList.add("on");
  setTimeout(() => t.classList.remove("on"), 3000);
}

function copySnippet(btn) {
  const code = btn.closest(".code-block-wrapper").querySelector("code").innerText;
  navigator.clipboard.writeText(code).then(() => {
    const span = btn.querySelector("span");
    const original = span.textContent;
    span.textContent = "Copied!";
    setTimeout(() => { span.textContent = original; }, 2000);
  });
}

function copyText(text) {
  navigator.clipboard.writeText(text).then(() => {
    toast("Answer copied to clipboard");
  });
}

// ============================================================================
// 3. API CLIENT
// ============================================================================
async function api(path, method = "GET", body, signal) {
  const headers = { "Content-Type": "application/json" };
  const keyInput = $("#apikey");
  const key = keyInput ? keyInput.value.trim() : "";
  if (key) headers["x-api-key"] = key;

  const res = await fetch("/api" + path, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
    signal
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const msg = err.detail || `Request failed with status ${res.status}`;
    toast(msg, true);
    throw new Error(msg);
  }
  return res.json();
}

// ============================================================================
// 4. STORAGE & CHAT CONVERSATION MANAGEMENT
// ============================================================================
const STORAGE_KEY = "semantic_cache_chats_v2";

function loadChatsFromStorage() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    STATE.chats = saved ? JSON.parse(saved) : [];
  } catch (e) {
    console.error("Failed to load chats from localStorage:", e);
    STATE.chats = [];
  }
}

function saveChatsToStorage() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(STATE.chats));
  } catch (e) {
    console.error("Failed to save chats to localStorage:", e);
  }
}

function createNewChat() {
  // If streaming, abort
  if (STATE.isStreaming && STATE.abortController) {
    STATE.abortController.abort();
  }

  const newChat = {
    id: "chat_" + Date.now(),
    title: "New conversation",
    messages: []
  };
  STATE.chats.unshift(newChat);
  STATE.currentChatId = newChat.id;
  saveChatsToStorage();
  renderRecentChatsList();
  renderActiveChatMessages();

  // Focus textarea
  $("#q").focus();
}

function deleteChat(chatId, event) {
  if (event) {
    event.stopPropagation();
    event.preventDefault();
  }
  STATE.chats = STATE.chats.filter(c => c.id !== chatId);
  saveChatsToStorage();

  if (STATE.currentChatId === chatId) {
    if (STATE.chats.length > 0) {
      STATE.currentChatId = STATE.chats[0].id;
    } else {
      STATE.currentChatId = null;
    }
  }
  renderRecentChatsList();
  renderActiveChatMessages();
}

function clearAllChats() {
  if (confirm("Delete all conversation history?")) {
    STATE.chats = [];
    saveChatsToStorage();
    createNewChat();
  }
}

function selectChat(chatId) {
  if (STATE.isStreaming && STATE.abortController) {
    STATE.abortController.abort();
  }
  STATE.currentChatId = chatId;
  renderRecentChatsList();
  renderActiveChatMessages();

  // If on mobile, close sidebar drawer
  closeSidebar();
}

function renderRecentChatsList() {
  const container = $("#recent-chats-list");
  if (!STATE.chats.length) {
    container.innerHTML = '<div style="padding:10px 8px;font-size:12px;color:var(--text-muted);">No saved chats yet.</div>';
    return;
  }

  container.innerHTML = STATE.chats.map(c => `
    <div class="chat-item ${c.id === STATE.currentChatId ? 'active' : ''}">
      <button type="button" class="chat-item-btn" onclick="selectChat('${c.id}')" aria-label="${escapeHtml(c.title)}">
        <span class="chat-item-title">${escapeHtml(c.title)}</span>
      </button>
      <button type="button" class="chat-del-btn" onclick="event.stopPropagation(); deleteChat('${c.id}', event)" title="Delete conversation" aria-label="Delete conversation">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="pointer-events:none;"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
      </button>
    </div>
  `).join("");
}

// ============================================================================
// 5. CHAT RENDERING & STREAMING ENGINE
// ============================================================================
function getCurrentChat() {
  return STATE.chats.find(c => c.id === STATE.currentChatId);
}

function renderActiveChatMessages() {
  const chat = getCurrentChat();
  const streamContainer = $("#chat-stream");

  if (!chat || !chat.messages.length) {
    streamContainer.innerHTML = `
      <div id="empty-state" class="chat-empty-state">
        <div class="greeting-badge">⚡ Instant Semantic Acceleration</div>
        <h2 class="greeting-heading">Where should we start?</h2>
        <p class="greeting-desc">Ask a question to see multi-tier caching in action. Repeated questions hit L1 in &lt;15ms; semantic rephrasings hit L2 with Qdrant and Cross-Encoder verification.</p>

        <div class="prompt-chips-grid">
          <button class="prompt-chip" data-query="How do I reset my password?">
            <span class="chip-title">How do I reset my password?</span>
            <span class="chip-sub">Tests authentication FAQ caching</span>
          </button>
          <button class="prompt-chip" data-query="What is retrieval augmented generation (RAG)?">
            <span class="chip-title">What is RAG?</span>
            <span class="chip-sub">Architecture and vector databases</span>
          </button>
          <button class="prompt-chip" data-query="How to install Python on Windows 11?">
            <span class="chip-title">Install Python on Windows</span>
            <span class="chip-sub">Step-by-step setup procedure</span>
          </button>
          <button class="prompt-chip" data-query="Explain the difference between SQL and NoSQL">
            <span class="chip-title">SQL vs NoSQL comparison</span>
            <span class="chip-sub">Technical differences & tradeoffs</span>
          </button>
        </div>
      </div>
    `;
    attachChipEvents();
    return;
  }

  streamContainer.innerHTML = chat.messages.map((m, idx) => {
    if (m.role === "user") {
      return `
        <div class="msg-row user-row">
          <div class="user-bubble">${escapeHtml(m.text)}</div>
        </div>
      `;
    } else {
      return buildAssistantHtml(m.text, m.meta, m.error, idx);
    }
  }).join("");

  scrollToBottom();
}

function buildAssistantHtml(rawText, meta, isError = false, msgIndex = null) {
  if (isError) {
    return `
      <div class="msg-row assistant-row error-row">
        <div class="assistant-container">
          <div class="assistant-avatar" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
          </div>
          <div class="assistant-content">
            <div class="assistant-markdown">${escapeHtml(rawText)}</div>
            <button class="retry-btn" onclick="retryLastQuery()">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:13px;height:13px;"><path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.3"></path></svg>
              <span>Retry request</span>
            </button>
          </div>
        </div>
      </div>
    `;
  }

  let badgeHtml = "";
  let feedbackHtml = "";

  if (meta && meta.level) {
    badgeHtml = `
      <button class="cache-badge ${meta.level}" onclick="openInspectorWithData(${JSON.stringify(meta).replace(/"/g, '&quot;')})" title="Click to inspect tier decision">
        <span>${LBL[meta.level] || meta.level}</span>
        ${meta.ms ? `· ${Math.round(meta.ms)}ms` : ""}
      </button>
      ${meta.level === "L2_semantic" && meta.score ? `<span class="mute text-xs">Sim: <strong>${meta.score}</strong></span>` : ""}
      ${meta.coalesced ? `<span class="mute text-xs">· (shared with in-flight request)</span>` : ""}
    `;

    // L2 Feedback Buttons
    if (meta.level === "L2_semantic" && meta.log_id) {
      feedbackHtml = `
        <div class="feedback-actions">
          <button class="action-icon-btn" onclick="sendFeedback(${meta.log_id}, true, this)" title="Accurate semantic match">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path></svg>
          </button>
          <button class="action-icon-btn" onclick="sendFeedback(${meta.log_id}, false, this)" title="False semantic match">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"></path></svg>
          </button>
        </div>
      `;
    }
  }

  const renderedContent = renderMarkdown(rawText);

  return `
    <div class="msg-row assistant-row" data-msg-idx="${msgIndex !== null ? msgIndex : ''}">
      <div class="assistant-container">
        <div class="assistant-avatar" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
        </div>
        <div class="assistant-content">
          <div class="assistant-markdown">${renderedContent}</div>
          <div class="msg-footer">
            ${badgeHtml}
            <button class="action-icon-btn" onclick="copyText(this.closest('.assistant-content').querySelector('.assistant-markdown').innerText)" title="Copy answer">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
            </button>
            ${feedbackHtml}
          </div>
        </div>
      </div>
    </div>
  `;
}

function scrollToBottom(force = false) {
  if (STATE.userScrolledUp && !force) return;
  const viewport = $(".chat-viewport");
  if (viewport) {
    viewport.scrollTop = viewport.scrollHeight;
  }
}

// Track if user manually scrolled up
function setupScrollDetection() {
  const viewport = $(".chat-viewport");
  if (!viewport) return;
  viewport.addEventListener("scroll", () => {
    const isAtBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 40;
    STATE.userScrolledUp = !isAtBottom;
  });
}

async function sendFeedback(logId, good, btn) {
  try {
    await api("/feedback", "POST", { log_id: logId, good });
    btn.classList.add("active");
    toast(good ? "Feedback saved: Threshold relaxed slightly" : "Feedback saved: Threshold tightened");
  } catch (err) {
    console.error("Feedback error:", err);
  }
}

// ============================================================================
// 6. SUBMISSION & STREAM HANDLING
// ============================================================================
async function handleSend() {
  // If streaming is active, button acts as stop
  if (STATE.isStreaming) {
    if (STATE.abortController) {
      STATE.abortController.abort();
      STATE.abortController = null;
    }
    setStreamingState(false);
    return;
  }

  const textarea = $("#q");
  const query = textarea.value.trim();
  if (!query) return;

  // Ensure active chat
  let chat = getCurrentChat();
  if (!chat) {
    createNewChat();
    chat = getCurrentChat();
  }

  // Update title if first message
  if (chat.messages.length === 0) {
    chat.title = query.length > 32 ? query.slice(0, 32) + "…" : query;
    saveChatsToStorage();
    renderRecentChatsList();
  }

  // Append user message
  chat.messages.push({ role: "user", text: query });
  saveChatsToStorage();

  // Reset textarea & height
  textarea.value = "";
  textarea.style.height = "auto";

  // Re-render chat
  renderActiveChatMessages();
  STATE.userScrolledUp = false;
  scrollToBottom(true);

  // Request payload
  const body = {
    query,
    ns: $("#ns").value.trim() || "default",
    doc_ver: $("#ver").value.trim() || "v1",
    tag: $("#tag").value.trim(),
    use_cache: $("#usec").checked
  };

  const useStream = $("#stream").checked;
  setStreamingState(true);
  STATE.abortController = new AbortController();

  // Create temporary assistant placeholder in DOM
  const streamContainer = $("#chat-stream");
  const placeholderRow = document.createElement("div");
  placeholderRow.className = "msg-row assistant-row";
  placeholderRow.innerHTML = `
    <div class="assistant-container">
      <div class="assistant-avatar" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
      </div>
      <div class="assistant-content">
        <div class="assistant-markdown"><span class="streaming-cursor"></span></div>
        <div class="msg-footer">
          <span class="mute text-xs">Resolving query...</span>
        </div>
      </div>
    </div>
  `;
  streamContainer.appendChild(placeholderRow);
  scrollToBottom(true);

  const markdownDiv = placeholderRow.querySelector(".assistant-markdown");
  const footerDiv = placeholderRow.querySelector(".msg-footer");

  let accumulatedText = "";
  let resolvedMeta = null;

  try {
    if (useStream) {
      const headers = { "Content-Type": "application/json" };
      const keyInput = $("#apikey");
      const key = keyInput ? keyInput.value.trim() : "";
      if (key) headers["x-api-key"] = key;

      const response = await fetch("/api/ask/stream", {
        method: "POST",
        headers,
        body: JSON.stringify(body),
        signal: STATE.abortController.signal
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || "Stream connection failed");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      let lastRenderTime = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const payload = JSON.parse(line.slice(6));

          if (payload.meta) {
            resolvedMeta = payload.meta;
            updateInspector(resolvedMeta, query);
          }
          if (payload.t) {
            accumulatedText += payload.t;
            // Throttle innerHTML re-parsing to animation frames for butter-smooth 60fps rendering
            const now = performance.now();
            if (now - lastRenderTime > 20) {
              markdownDiv.innerHTML = renderMarkdown(accumulatedText) + '<span class="streaming-cursor"></span>';
              lastRenderTime = now;
              scrollToBottom();
            }
          }
        }
      }
      // Ensure final frame is rendered
      markdownDiv.innerHTML = renderMarkdown(accumulatedText) + '<span class="streaming-cursor"></span>';
      scrollToBottom();
    } else {
      // Non-streaming fallback
      const result = await api("/ask", "POST", body, STATE.abortController.signal);
      accumulatedText = result.answer;
      resolvedMeta = result;
      updateInspector(resolvedMeta, query);
    }

    // Finished streaming cleanly
    chat.messages.push({
      role: "assistant",
      text: accumulatedText,
      meta: resolvedMeta
    });
    saveChatsToStorage();

    // Finalize placeholder with full markdown and interactive buttons
    placeholderRow.outerHTML = buildAssistantHtml(accumulatedText, resolvedMeta, false, chat.messages.length - 1);
    scrollToBottom();
  } catch (err) {
    if (err.name === "AbortError") {
      // User aborted stream
      if (accumulatedText) {
        chat.messages.push({
          role: "assistant",
          text: accumulatedText + " *(Stream stopped)*",
          meta: resolvedMeta
        });
        saveChatsToStorage();
        placeholderRow.outerHTML = buildAssistantHtml(accumulatedText + " *(Stream stopped)*", resolvedMeta, false, chat.messages.length - 1);
      } else {
        placeholderRow.remove();
      }
    } else {
      // Real API or network error
      console.error("Query failed:", err);
      chat.messages.push({
        role: "assistant",
        text: `Error: ${err.message}`,
        error: true
      });
      saveChatsToStorage();
      placeholderRow.outerHTML = buildAssistantHtml(err.message, null, true, chat.messages.length - 1);
    }
  } finally {
    setStreamingState(false);
    STATE.abortController = null;
  }
}

function setStreamingState(isStreaming) {
  STATE.isStreaming = isStreaming;
  const btn = $("#send-btn");
  if (isStreaming) {
    btn.classList.add("streaming");
    btn.setAttribute("aria-label", "Stop generating");
  } else {
    btn.classList.remove("streaming");
    btn.setAttribute("aria-label", "Send message");
  }
}

function retryLastQuery() {
  const chat = getCurrentChat();
  if (!chat || !chat.messages.length) return;

  // Pop any failed message
  const last = chat.messages[chat.messages.length - 1];
  if (last.role === "assistant" && last.error) {
    chat.messages.pop();
  }

  // Find user prompt
  const lastUser = [...chat.messages].reverse().find(m => m.role === "user");
  if (lastUser) {
    $("#q").value = lastUser.text;
    handleSend();
  }
}

function attachChipEvents() {
  $$(".prompt-chip").forEach(chip => {
    chip.onclick = () => {
      $("#q").value = chip.dataset.query;
      handleSend();
    };
  });
}

// Auto-grow textarea
function setupComposerTextarea() {
  const textarea = $("#q");
  textarea.addEventListener("input", () => {
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 160) + "px";
  });

  textarea.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  $("#send-btn").onclick = handleSend;
  $("#new-chat-btn").onclick = createNewChat;
  $("#clear-chats-btn").onclick = clearAllChats;

  $("#toggle-settings-btn").onclick = () => {
    const box = $("#composer-settings");
    box.classList.toggle("collapsed");
  };
}

// ============================================================================
// 7. REQUEST INSPECTOR DRAWER
// ============================================================================
function updateInspector(meta, query) {
  if (!meta) return;
  STATE.latestInspection = { meta, query };

  let reason = "";
  if (meta.level === "L1_exact") {
    reason = "Exact normalized SHA-256 hash match found in L1 key-value cache (Redis/SQLite). Returned in sub-millisecond time with 0 LLM tokens consumed.";
  } else if (meta.level === "L2_semantic") {
    reason = `Cosine similarity score (${meta.score}) met category threshold (${meta.threshold}, ${meta.category}). Lexical antonym guard and Cross-Encoder reranker verified semantic equality.`;
  } else {
    reason = "No exact hash or semantic similarity above threshold. Delegated to L3 LLM generation. Cached entry stored for future hits.";
    if (meta.coalesced) {
      reason += " Coalesced: Request shared LLM execution with a concurrent identical in-flight query.";
    }
  }

  const body = $("#inspector-body");
  body.innerHTML = `
    <div class="inspector-metric-card">
      <b>Cache Tier Decision</b>
      <span class="badge ${meta.level}" style="font-size:13px;padding:3px 10px;">${LBL[meta.level] || meta.level}</span>
    </div>

    <div class="inspector-metric-card">
      <b>Response Latency</b>
      <span>${meta.ms ? Math.round(meta.ms) + " ms" : "-"}</span>
    </div>

    <div class="inspector-metric-card">
      <b>Cosine Similarity Score</b>
      <span>${meta.score > 0 ? meta.score.toFixed(4) : "N/A (Miss)"}</span>
    </div>

    <div class="inspector-metric-card">
      <b>Category & Threshold</b>
      <span>${meta.category || "general"} · thr ${meta.threshold || 0.88}</span>
    </div>

    <div class="inspector-metric-card">
      <b>Coalescing Status</b>
      <span>${meta.coalesced ? "Yes (Shared in-flight execution)" : "No (Standalone)"}</span>
    </div>

    <div class="routing-explanation">
      <strong style="display:block;margin-bottom:4px;font-size:12px;text-transform:uppercase;color:var(--accent);">Decision Analysis</strong>
      ${reason}
    </div>
  `;
}

function openInspectorWithData(meta) {
  updateInspector(meta, "Inspected message");
  openInspector();
}

function openInspector() {
  $("#inspector-drawer").classList.add("on");
  $("#inspector-backdrop").classList.add("on");
}

function closeInspector() {
  $("#inspector-drawer").classList.remove("on");
  $("#inspector-backdrop").classList.remove("on");
}

function setupInspector() {
  $("#open-inspector-btn").onclick = openInspector;
  $("#close-inspector-btn").onclick = closeInspector;
  $("#inspector-backdrop").onclick = closeInspector;
}

// ============================================================================
// 8. PANEL SWITCHING & MOBILE SIDEBAR
// ============================================================================
function switchPanel(panelId) {
  STATE.activePanel = panelId;

  $$(".nav-item").forEach(btn => {
    const isTarget = btn.dataset.panel === panelId;
    btn.classList.toggle("on", isTarget);
    btn.setAttribute("aria-selected", isTarget ? "true" : "false");
  });

  $$(".app-panel").forEach(panel => {
    panel.classList.toggle("on", panel.id === panelId);
  });

  const heading = {
    "chat-panel": "Chat",
    "stats-panel": "Stats & Inspector",
    "entries-panel": "Cache Entries",
    "lab-panel": "Lab & Benchmark"
  }[panelId] || "Semantic Cache";

  $("#topbar-heading").textContent = heading;

  if (panelId === "stats-panel") loadStats();
  if (panelId === "entries-panel") loadEntries();

  closeSidebar();
}

function openSidebar() {
  $("#sidebar").classList.add("open");
  $("#sidebar-backdrop").classList.add("on");
}

function closeSidebar() {
  $("#sidebar").classList.remove("open");
  $("#sidebar-backdrop").classList.remove("on");
}

function setupSidebarAndPanels() {
  $$(".nav-item").forEach(btn => {
    btn.onclick = () => switchPanel(btn.dataset.panel);
  });

  $("#hamburger-btn").onclick = openSidebar;
  $("#sidebar-close-btn").onclick = closeSidebar;
  $("#sidebar-backdrop").onclick = closeSidebar;
}

// ============================================================================
// 9. THEME TOGGLE (Persistent)
// ============================================================================
function setupTheme() {
  const savedTheme = localStorage.getItem("semantic_cache_theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const initialTheme = savedTheme || (prefersDark ? "dark" : "light");

  applyTheme(initialTheme);

  $("#theme-toggle").onclick = () => {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    const next = current === "dark" ? "light" : "dark";
    applyTheme(next);
  };
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("semantic_cache_theme", theme);

  if (theme === "light") {
    $("#theme-icon-sun").style.display = "none";
    $("#theme-icon-moon").style.display = "block";
    $("#theme-text").textContent = "Light";
  } else {
    $("#theme-icon-sun").style.display = "block";
    $("#theme-icon-moon").style.display = "none";
    $("#theme-text").textContent = "Dark";
  }
}

// ============================================================================
// 10. STATS, ENTRIES & LAB RESTYLED
// ============================================================================
const renderTable = (selector, headers, rows) => {
  const el = $(selector);
  if (!rows || !rows.length) {
    el.innerHTML = `<tbody><tr><td colspan="${headers.length}" style="text-align:center;padding:24px;color:var(--text-muted);">No records available.</td></tr></tbody>`;
    return;
  }
  el.innerHTML = `
    <thead><tr>${headers.map(h => `<th>${h}</th>`).join("")}</tr></thead>
    <tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody>
  `;
};

async function loadStats() {
  try {
    const s = await api("/stats");
    const L = s.levels;
    const ms = k => (L[k] ? Math.round(L[k].avg_ms) + " ms" : "-");

    $("#kpis").innerHTML = [
      ["Hit Rate", (s.hit_rate * 100).toFixed(1) + "%"],
      ["Total Requests", s.total],
      ["Cached Entries", s.entries],
      ["Tokens Saved", s.tokens_saved.toLocaleString()],
      ["Cost Saved", "$" + s.cost_saved.toFixed(4)],
      ["Avg L1 / L2 / LLM", `${ms("L1_exact")} / ${ms("L2_semantic")} / ${ms("L3_llm")}`]
    ].map(([label, val]) => `
      <div class="kpi">
        <b>${val}</b>
        <span>${label}</span>
      </div>
    `).join("");

    $("#thr").innerHTML = Object.entries(s.thresholds).map(([k, v]) => `
      <label class="setting-item" style="font-size:13px;">
        <span>${k}:</span>
        <input type="number" step="0.01" min="0.5" max="0.99" id="t_${k}" value="${v}" style="width:85px">
      </label>
    `).join("") + `<button class="btn" id="savethr" style="padding:6px 14px;">Save Thresholds</button>`;

    $("#savethr").onclick = async () => {
      await api("/threshold", "POST", { code: +$("#t_code").value, general: +$("#t_general").value });
      toast("Thresholds updated successfully");
    };

    renderTable(
      "#recent",
      ["Query", "Level", "Cosine Sim", "Latency"],
      s.recent.map(r => [
        escapeHtml(r.query),
        `<span class="badge ${r.level}">${LBL[r.level]}</span>`,
        r.score > 0 ? r.score.toFixed(3) : "-",
        Math.round(r.ms) + " ms"
      ])
    );

    renderTable(
      "#top",
      ["Query", "Hits (Reuses)"],
      s.top.filter(t => t.hits).map(t => [escapeHtml(t.query), `<strong>${t.hits}</strong>`])
    );
  } catch (err) {
    console.error("Failed to load stats:", err);
  }
}

async function loadEntries() {
  try {
    const entries = await api("/entries");
    renderTable(
      "#etable",
      ["Query", "Answer Preview", "Namespace", "Tag", "Hits", "Action"],
      entries.map(x => [
        `<strong>${escapeHtml(x.query)}</strong>`,
        escapeHtml(x.answer.slice(0, 80)) + (x.answer.length > 80 ? "…" : ""),
        escapeHtml(x.ns),
        escapeHtml(x.tag || "-"),
        x.hits,
        `<button class="btn ghost text-xs" style="padding:4px 8px;" data-id="${x.id}">Delete</button>`
      ])
    );

    $$("#etable [data-id]").forEach(b => {
      b.onclick = async () => {
        await api("/entries/" + b.dataset.id, "DELETE");
        toast("Cache entry deleted");
        loadEntries();
      };
    });
  } catch (err) {
    console.error("Failed to load entries:", err);
  }
}

function setupEntriesAndLab() {
  $("#clr").onclick = async () => {
    if (confirm("Purge every cached vector and database record?")) {
      await api("/clear", "POST");
      loadEntries();
      toast("Cache completely cleared");
    }
  };

  $("#inv").onclick = async () => {
    const tag = $("#invtag").value.trim();
    if (!tag) {
      toast("Please enter a tag to invalidate", true);
      return;
    }
    const r = await api("/invalidate", "POST", { tag });
    toast(`${r.deleted} entries invalidated`);
    $("#invtag").value = "";
    loadEntries();
  };

  $("#dowarm").onclick = async () => {
    const q = $("#warm").value.split("\n").filter(x => x.trim());
    if (!q.length) {
      toast("Enter one or more queries to warm", true);
      return;
    }
    toast("Warming cache in background...");
    const r = await api("/warm", "POST", { queries: q, ns: $("#ns").value.trim() || "default" });
    toast(`Warmed ${r.warmed} queries into semantic cache`);
    $("#warm").value = "";
  };

  $("#doeval").onclick = async () => {
    const btn = $("#doeval");
    btn.disabled = true;
    toast("Running accuracy & threshold evaluation...");

    try {
      const d = await api("/evaluate");
      const c = d.curve;
      const W = 700, H = 260;
      const padLeft = 45, padRight = 30, padTop = 45, padBottom = 35;
      const plotW = W - padLeft - padRight;
      const plotH = H - padTop - padBottom;
      const x = t => padLeft + ((t - 0.5) / 0.48) * plotW;
      const y = v => padTop + plotH - v * plotH;

      const line = (k, col) => `<polyline fill="none" stroke="${col}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" points="${c.map(p => `${x(p.t).toFixed(1)},${y(p[k]).toFixed(1)}`).join(" ")}"/>`;

      $("#evalout").innerHTML = `
        <div style="background:var(--bg-surface);border:1px solid var(--border-subtle);border-radius:var(--radius-md);padding:16px;">
          <!-- Clean HTML Legend with spacing -->
          <div style="display:flex;flex-wrap:wrap;gap:18px;align-items:center;margin-bottom:12px;font-size:12px;font-weight:600;">
            <div style="display:flex;align-items:center;gap:6px;">
              <span style="width:14px;height:3px;background:var(--accent);display:inline-block;border-radius:2px;"></span>
              <span style="color:var(--text-primary);">Raw Precision</span>
            </div>
            <div style="display:flex;align-items:center;gap:6px;">
              <span style="width:14px;height:3px;background:var(--hit-l3);display:inline-block;border-radius:2px;"></span>
              <span style="color:var(--text-primary);">Precision + Reranker</span>
            </div>
            <div style="display:flex;align-items:center;gap:6px;">
              <span style="width:14px;height:3px;background:var(--hit-l2);display:inline-block;border-radius:2px;"></span>
              <span style="color:var(--text-primary);">Recall</span>
            </div>
          </div>

          <svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;overflow:visible;" role="img" aria-label="Precision & Recall by Threshold">
            <!-- Grid lines & Y-Axis labels -->
            ${[0, 0.25, 0.5, 0.75, 1].map(v => `
              <line x1="${padLeft}" x2="${W - padRight}" y1="${y(v)}" y2="${y(v)}" stroke="var(--border-subtle)" stroke-width="1" stroke-dasharray="${v === 0 ? 'none' : '3,3'}"/>
              <text x="${padLeft - 8}" y="${y(v) + 4}" fill="var(--text-muted)" font-size="11" text-anchor="end">${v}</text>
            `).join("")}

            <!-- X-Axis Labels -->
            ${c.filter((_, i) => i % 3 === 0).map(p => `
              <line x1="${x(p.t)}" x2="${x(p.t)}" y1="${y(0)}" y2="${y(0) + 5}" stroke="var(--border-subtle)" stroke-width="1"/>
              <text x="${x(p.t)}" y="${H - 12}" fill="var(--text-muted)" font-size="11" text-anchor="middle">${p.t.toFixed(2)}</text>
            `).join("")}

            <!-- Data series lines -->
            ${line("precision", "var(--accent)")}
            ${line("precision_rr", "var(--hit-l3)")}
            ${line("recall", "var(--hit-l2)")}
          </svg>
        </div>
        <div class="scroll" style="margin-top:14px;">
          <table>
            <thead><tr><th>Test Pair</th><th>Paraphrase</th><th>Cosine Sim</th><th>Reranker</th><th>Status</th></tr></thead>
            <tbody>
              ${d.pairs.map(p => `
                <tr>
                  <td>${escapeHtml(p.a)}<br><span class="mute">${escapeHtml(p.b)}</span></td>
                  <td>${p.same ? '<strong style="color:var(--hit-l2)">Yes</strong>' : '<span class="mute">No</span>'}</td>
                  <td>${p.sim}</td>
                  <td>${p.rerank}</td>
                  <td>${p.pass ? '<span class="badge L2_semantic">Accepted</span>' : '<span class="badge L3_llm">Rejected</span>'}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      `;
      toast("Evaluation sweep completed");
    } catch (err) {
      toast("Evaluation failed: " + err.message, true);
    } finally {
      btn.disabled = false;
    }
  };

  $("#dosim").onclick = async () => {
    const r = await api(`/simulate?queries=${$("#sq").value}&hit_rate=${$("#sh").value}&llm_ms=${$("#sl").value}`);
    $("#simout").innerHTML = [
      ["LLM Calls Saved", r.llm_calls_saved.toLocaleString()],
      ["Time Saved", r.time_saved_min + " min"],
      ["Cost Saved", "$" + r.cost_saved.toFixed(2)],
      ["Blended Latency", r.avg_ms + " ms"]
    ].map(([l, v]) => `
      <div class="kpi">
        <b>${v}</b>
        <span>${l}</span>
      </div>
    `).join("");
  };
}

// Fetch live backend configuration
async function checkBackendConfig() {
  const cfgText = $("#cfg-text");
  const cfg = $("#cfg");
  if (!cfgText && !cfg) return;
  try {
    const c = await api("/config");
    if (cfgText) cfgText.textContent = `L1: ${c.l1} · Qdrant: ${c.vectors} · Reranker: ${c.reranker ? 'ON' : 'OFF'}`;
  } catch (err) {
    if (cfg) cfg.classList.add("offline");
    if (cfgText) cfgText.textContent = "Backend Offline";
  }
}

// ============================================================================
// 11. INITIALIZATION
// ============================================================================
window.addEventListener("DOMContentLoaded", () => {
  setupTheme();
  loadChatsFromStorage();

  if (!STATE.chats.length) {
    createNewChat();
  } else {
    STATE.currentChatId = STATE.chats[0].id;
    renderRecentChatsList();
    renderActiveChatMessages();
  }

  setupComposerTextarea();
  setupInspector();
  setupSidebarAndPanels();
  setupEntriesAndLab();
  setupScrollDetection();
  checkBackendConfig();
});
