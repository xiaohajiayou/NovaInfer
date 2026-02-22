const chatLog = document.getElementById("chatLog");
const promptInput = document.getElementById("promptInput");
const sendBtn = document.getElementById("sendBtn");
const cancelBtn = document.getElementById("cancelBtn");
const serverBase = document.getElementById("serverBase");
const debugStats = document.getElementById("debugStats");
const debugLog = document.getElementById("debugLog");
const sessionList = document.getElementById("sessionList");
const newSessionBtn = document.getElementById("newSessionBtn");
const sessionTitle = document.getElementById("sessionTitle");
const useServerDefaults = document.getElementById("useServerDefaults");
const maxTokensInput = document.getElementById("maxTokensInput");
const topKInput = document.getElementById("topKInput");
const topPInput = document.getElementById("topPInput");
const temperatureInput = document.getElementById("temperatureInput");
const PROMPT_MAX_HEIGHT_PX = 140;

let sessions = [];
let currentSessionId = null;
let sessionSeq = 0;

function nowMs() {
  return Date.now();
}

function resizePromptInput() {
  promptInput.style.height = "auto";
  const next = Math.min(promptInput.scrollHeight, PROMPT_MAX_HEIGHT_PX);
  promptInput.style.height = `${next}px`;
}

function bindExclusivePanels() {
  const panels = Array.from(document.querySelectorAll(".panel-toggle"));
  for (const panel of panels) {
    panel.addEventListener("toggle", () => {
      if (!panel.open) return;
      for (const other of panels) {
        if (other !== panel) {
          other.open = false;
        }
      }
    });
  }

  document.addEventListener("mousedown", (e) => {
    const target = e.target;
    if (!(target instanceof Node)) return;
    for (const panel of panels) {
      if (!panel.contains(target)) {
        panel.open = false;
      }
    }
  });
}

function createSession() {
  sessionSeq += 1;
  return {
    id: `s${sessionSeq}`,
    title: `Chat ${sessionSeq}`,
    messages: [],
    activeRequestId: null,
    reqStartedAtMs: 0,
    firstChunkAtMs: 0,
    chunkCount: 0,
    status: "idle",
    debugLines: [],
  };
}

function currentSession() {
  return sessions.find((s) => s.id === currentSessionId) || null;
}

function logDebug(s, line) {
  const ts = new Date().toLocaleTimeString();
  s.debugLines.push(`[${ts}] ${line}`);
  if (s.debugLines.length > 200) {
    s.debugLines.shift();
  }
  if (s.id === currentSessionId) {
    renderDebug();
  }
}

function setSessionStatus(s, status) {
  s.status = status;
  if (s.id === currentSessionId) {
    renderDebugStats();
    renderControls();
  }
}

function renderSessions() {
  sessionList.innerHTML = "";
  for (const s of sessions) {
    const btn = document.createElement("button");
    btn.className = `session-item ${s.id === currentSessionId ? "active" : ""}`;
    const suffix = s.activeRequestId ? " (running)" : "";
    btn.textContent = `${s.title}${suffix}`;
    btn.addEventListener("click", () => {
      currentSessionId = s.id;
      renderAll();
    });
    sessionList.appendChild(btn);
  }
}

function renderChat() {
  const s = currentSession();
  if (!s) return;
  const prevTop = chatLog.scrollTop;
  const stickToBottom =
    chatLog.scrollHeight - chatLog.clientHeight - chatLog.scrollTop < 24;
  chatLog.innerHTML = "";
  for (const m of s.messages) {
    const el = document.createElement("div");
    el.className = `msg ${m.role}`;
    if (m.role === "assistant") {
      const details = document.createElement("details");
      details.className = "assistant-reasoning";
      details.open = m.reasoningOpen !== false;
      const summary = document.createElement("summary");
      summary.textContent = m.reasoningRunning ? "思考中..." : "思考过程";
      const body = document.createElement("pre");
      body.className = "assistant-reasoning-body";
      body.textContent =
        typeof m.reasoning === "string" && m.reasoning.length > 0
          ? m.reasoning
          : (m.reasoningRunning ? "思考中..." : "（无思考内容）");
      details.addEventListener("toggle", () => {
        m.reasoningOpen = details.open;
      });
      details.appendChild(summary);
      details.appendChild(body);
      el.appendChild(details);

      const answer = document.createElement("div");
      answer.className = "assistant-answer";
      answer.textContent = m.text || "";
      el.appendChild(answer);
    } else {
      el.textContent = m.text;
    }
    chatLog.appendChild(el);
  }
  if (stickToBottom) {
    chatLog.scrollTop = chatLog.scrollHeight;
  } else {
    chatLog.scrollTop = prevTop;
  }
}

function renderDebugStats() {
  const s = currentSession();
  if (!s) return;
  const ttft = s.firstChunkAtMs > 0 ? `${s.firstChunkAtMs - s.reqStartedAtMs}ms` : "-";
  const elapsed = s.reqStartedAtMs > 0 ? `${nowMs() - s.reqStartedAtMs}ms` : "-";
  debugStats.textContent = `status=${s.status} request_id=${s.activeRequestId || "-"} chunks=${s.chunkCount} ttft=${ttft} elapsed=${elapsed}`;
}

function renderDebug() {
  const s = currentSession();
  if (!s) return;
  debugLog.textContent = s.debugLines.join("\n");
  debugLog.scrollTop = debugLog.scrollHeight;
  renderDebugStats();
}

function renderControls() {
  const s = currentSession();
  if (!s) return;
  const running = Boolean(s.activeRequestId);
  sendBtn.disabled = false;
  sendBtn.classList.toggle("is-running", running);
  sendBtn.setAttribute("aria-label", running ? "Cancel" : "Send");
  sendBtn.title = running ? "Cancel running request" : "Send";
  cancelBtn.disabled = !running;
}

function renderAll() {
  const s = currentSession();
  sessionTitle.textContent = s ? `NovaInfer Chat - ${s.title}` : "NovaInfer Chat";
  renderSessions();
  renderChat();
  renderDebug();
  renderControls();
}

function appendMessage(s, role, text) {
  s.messages.push({ role, text });
  if (s.id === currentSessionId) {
    renderChat();
  }
  return s.messages.length - 1;
}

async function sendPrompt() {
  const s = currentSession();
  if (!s) return;
  if (s.activeRequestId) return;
  const prompt = promptInput.value.trim();
  if (!prompt) return;

  promptInput.value = "";
  resizePromptInput();
  appendMessage(s, "user", prompt);
  const assistantIndex = appendMessage(s, "assistant", "");
  s.messages[assistantIndex].reasoning = "";
  s.messages[assistantIndex].reasoningOpen = true;
  s.messages[assistantIndex].reasoningRunning = true;
  s.reqStartedAtMs = nowMs();
  s.firstChunkAtMs = 0;
  s.chunkCount = 0;
  s.activeRequestId = null;
  setSessionStatus(s, "request_sent");
  logDebug(s, `send prompt len=${prompt.length}`);
  renderSessions();

  const payload = {
    model: "qwen2",
    stream: true,
    include_reasoning: true,
    messages: [{ role: "user", content: prompt }],
  };
  if (!useServerDefaults.checked) {
    const maxTokens = Number.parseInt(maxTokensInput.value, 10);
    const topK = Number.parseInt(topKInput.value, 10);
    const topP = Number.parseFloat(topPInput.value);
    const temperature = Number.parseFloat(temperatureInput.value);
    if (Number.isFinite(maxTokens) && maxTokens > 0) payload.max_tokens = maxTokens;
    if (Number.isFinite(topK) && topK >= 0) payload.top_k = topK;
    if (Number.isFinite(topP) && topP >= 0) payload.top_p = topP;
    if (Number.isFinite(temperature) && temperature >= 0) payload.temperature = temperature;
  }

  const url = `${serverBase.value.replace(/\/$/, "")}/v1/chat/completions`;
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!resp.ok || !resp.body) {
    appendMessage(s, "meta", `request failed: http ${resp.status}`);
    logDebug(s, `http error status=${resp.status}`);
    setSessionStatus(s, "http_error");
    return;
  }

  logDebug(s, `http ok status=${resp.status}`);
  setSessionStatus(s, "stream_open");

  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep;
    while ((sep = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, sep).trim();
      buffer = buffer.slice(sep + 2);
      if (!frame.startsWith("data:")) continue;
      const payloadText = frame.slice(5).trim();
      if (payloadText === "[DONE]") {
        logDebug(s, "stream done");
        s.messages[assistantIndex].reasoningRunning = false;
        s.activeRequestId = null;
        setSessionStatus(s, "done");
        renderSessions();
        if (s.id === currentSessionId) {
          renderChat();
        }
        continue;
      }
      try {
        const obj = JSON.parse(payloadText);
        s.activeRequestId = obj.request_id || s.activeRequestId;
        if (s.firstChunkAtMs === 0) {
          s.firstChunkAtMs = nowMs();
          logDebug(s, `first chunk ttft=${s.firstChunkAtMs - s.reqStartedAtMs}ms`);
        }
        s.chunkCount += 1;
        const delta = obj?.choices?.[0]?.delta?.content || "";
        const reasoningDelta = obj?.choices?.[0]?.delta?.reasoning || "";
        const doneFlag = Boolean(obj?.is_finished);
        const tokenId = obj?.token_id;
        logDebug(
          s,
          `chunk#${s.chunkCount} req=${s.activeRequestId} token_id=${tokenId} done=${doneFlag} delta_len=${delta.length} reasoning_len=${reasoningDelta.length} delta=${JSON.stringify(delta)}`
        );
        setSessionStatus(s, doneFlag ? "finishing" : "streaming");
        const renderDelta = delta.length > 0 ? delta : "";
        s.messages[assistantIndex].text += renderDelta;
        if (reasoningDelta.length > 0) {
          if (typeof s.messages[assistantIndex].reasoning !== "string") {
            s.messages[assistantIndex].reasoning = "";
          }
          s.messages[assistantIndex].reasoning += reasoningDelta;
        }
        if (s.id === currentSessionId) {
          renderChat();
        }
        if (doneFlag) {
          s.messages[assistantIndex].reasoningRunning = false;
        }
      } catch (err) {
        appendMessage(s, "meta", `parse error: ${String(err)}`);
        logDebug(s, `parse error: ${String(err)}`);
        setSessionStatus(s, "parse_error");
      }
    }
  }

  if (s.activeRequestId === null) {
    setSessionStatus(s, "idle");
  } else {
    setSessionStatus(s, "stream_closed_without_done");
  }
}

async function cancelRequest() {
  const s = currentSession();
  if (!s || !s.activeRequestId) return;
  const reqId = s.activeRequestId;
  const url = `${serverBase.value.replace(/\/$/, "")}/v1/requests/${reqId}/cancel`;
  const resp = await fetch(url, { method: "POST" });
  if (resp.ok) {
    appendMessage(s, "meta", `request cancelled: ${reqId}`);
    logDebug(s, `cancel ok request_id=${reqId}`);
    setSessionStatus(s, "cancelled");
  } else {
    appendMessage(s, "meta", `cancel failed: http ${resp.status}`);
    logDebug(s, `cancel failed status=${resp.status} request_id=${reqId}`);
    setSessionStatus(s, "cancel_error");
  }
  s.activeRequestId = null;
  renderSessions();
  renderControls();
}

newSessionBtn.addEventListener("click", () => {
  const s = createSession();
  sessions.push(s);
  currentSessionId = s.id;
  console.log("[webui] new session created:", s.id);
  logDebug(s, `session created id=${s.id}`);
  renderAll();
});

function sendWithErrorHandling() {
  return sendPrompt().catch((err) => {
    const s = currentSession();
    if (!s) return;
    appendMessage(s, "meta", `request error: ${String(err)}`);
    logDebug(s, `request error: ${String(err)}`);
    setSessionStatus(s, "request_error");
    s.activeRequestId = null;
    renderSessions();
    renderControls();
  });
}

function cancelWithErrorHandling() {
  return cancelRequest().catch((err) => {
    const s = currentSession();
    if (!s) return;
    logDebug(s, `cancel error: ${String(err)}`);
    setSessionStatus(s, "cancel_error");
  });
}

sendBtn.addEventListener("click", () => {
  const s = currentSession();
  if (!s) return;
  if (s.activeRequestId) {
    cancelWithErrorHandling();
    return;
  }
  sendWithErrorHandling();
});

cancelBtn.addEventListener("click", () => {
  cancelWithErrorHandling();
});

promptInput.addEventListener("keydown", (e) => {
  if (e.key !== "Enter" || e.shiftKey || e.isComposing) {
    return;
  }
  e.preventDefault();
  const s = currentSession();
  if (!s || s.activeRequestId) return;
  sendWithErrorHandling();
});

promptInput.addEventListener("input", () => {
  resizePromptInput();
});

// Init with one session.
const initial = createSession();
sessions.push(initial);
currentSessionId = initial.id;
bindExclusivePanels();
resizePromptInput();
renderAll();
