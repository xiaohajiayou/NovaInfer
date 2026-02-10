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

let sessions = [];
let currentSessionId = null;
let sessionSeq = 0;

function nowMs() {
  return Date.now();
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
  chatLog.innerHTML = "";
  for (const m of s.messages) {
    const el = document.createElement("div");
    el.className = `msg ${m.role}`;
    el.textContent = m.text;
    chatLog.appendChild(el);
  }
  chatLog.scrollTop = chatLog.scrollHeight;
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
  sendBtn.disabled = running;
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
  appendMessage(s, "user", prompt);
  const assistantIndex = appendMessage(s, "assistant", "");
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
    max_tokens: 32,
    top_k: 1,
    top_p: 1.0,
    temperature: 1.0,
    messages: [{ role: "user", content: prompt }],
  };

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
        s.activeRequestId = null;
        setSessionStatus(s, "done");
        renderSessions();
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
        const doneFlag = Boolean(obj?.is_finished);
        const tokenId = obj?.token_id;
        logDebug(
          s,
          `chunk#${s.chunkCount} req=${s.activeRequestId} token_id=${tokenId} done=${doneFlag} delta_len=${delta.length} delta=${JSON.stringify(delta)}`
        );
        setSessionStatus(s, doneFlag ? "finishing" : "streaming");
        const renderDelta =
          delta.length > 0
            ? delta
            : tokenId !== null && tokenId !== undefined
              ? `<${tokenId}>`
              : "";
        s.messages[assistantIndex].text += renderDelta;
        if (s.id === currentSessionId) {
          renderChat();
        }
        renderSessions();
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

sendBtn.addEventListener("click", () => {
  sendPrompt().catch((err) => {
    const s = currentSession();
    if (!s) return;
    appendMessage(s, "meta", `request error: ${String(err)}`);
    logDebug(s, `request error: ${String(err)}`);
    setSessionStatus(s, "request_error");
    s.activeRequestId = null;
    renderSessions();
    renderControls();
  });
});

cancelBtn.addEventListener("click", () => {
  cancelRequest().catch((err) => {
    const s = currentSession();
    if (!s) return;
    logDebug(s, `cancel error: ${String(err)}`);
    setSessionStatus(s, "cancel_error");
  });
});

// Init with one session.
const initial = createSession();
sessions.push(initial);
currentSessionId = initial.id;
renderAll();
