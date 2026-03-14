const chatLog = document.getElementById("chatLog");
const promptInput = document.getElementById("promptInput");
const sendBtn = document.getElementById("sendBtn");
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
const STORAGE_KEY = "novainfer_webui_state_v3";
const PENDING_REQUEST_ID = "__pending__";

let sessions = [];
let currentSessionId = null;
let sessionSeq = 0;
let nodeSeq = 0;
let editingMessageId = null;
let lastKvStats = null;
const liveControllers = new Map();

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

function nextNodeId() {
  nodeSeq += 1;
  return `n${nodeSeq}`;
}

function makeNode(role, text, parentId, extra = {}) {
  return {
    id: nextNodeId(),
    role,
    text,
    parentId,
    childrenIds: [],
    activeChildId: null,
    reasoning: "",
    reasoningOpen: true,
    reasoningRunning: false,
    requestId: null,
    editDraft: "",
    ...extra,
  };
}

function createSession() {
  sessionSeq += 1;
  const root = {
    id: nextNodeId(),
    role: "root",
    text: "",
    parentId: null,
    childrenIds: [],
    activeChildId: null,
  };
  return {
    id: `s${sessionSeq}`,
    title: `Chat ${sessionSeq}`,
    rootId: root.id,
    activeLeafId: root.id,
    nodes: { [root.id]: root },
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

function isVisibleSession(session) {
  return Boolean(session) && session.id === currentSessionId;
}

function getNode(session, nodeId) {
  return session.nodes[nodeId] || null;
}

function isRequestRunning(session) {
  return Boolean(session && session.activeRequestId);
}

function sanitizeSessionsForStorage() {
  return sessions.map((session) => {
    const nodes = {};
    for (const [nodeId, node] of Object.entries(session.nodes)) {
      nodes[nodeId] = {
        ...node,
        reasoningRunning: false,
        requestId: null,
      };
    }
    return {
      ...session,
      activeRequestId: null,
      reqStartedAtMs: 0,
      firstChunkAtMs: 0,
      chunkCount: 0,
      status: "idle",
      nodes,
    };
  });
}

function saveState() {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        sessions: sanitizeSessionsForStorage(),
        currentSessionId,
        sessionSeq,
        nodeSeq,
      })
    );
  } catch (err) {
    console.warn("[webui] save state failed:", err);
  }
}

function restoreState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return false;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed.sessions) || parsed.sessions.length === 0) return false;
    sessions = parsed.sessions;
    currentSessionId = parsed.currentSessionId || parsed.sessions[0].id;
    sessionSeq = Number.isFinite(parsed.sessionSeq) ? parsed.sessionSeq : parsed.sessions.length;
    nodeSeq = Number.isFinite(parsed.nodeSeq) ? parsed.nodeSeq : 0;
    for (const session of sessions) {
      if (!session.nodes || !session.rootId || !session.nodes[session.rootId]) return false;
      session.activeRequestId = null;
      session.reqStartedAtMs = 0;
      session.firstChunkAtMs = 0;
      session.chunkCount = 0;
      session.status = "idle";
      session.debugLines = Array.isArray(session.debugLines) ? session.debugLines : [];
      for (const node of Object.values(session.nodes)) {
        node.reasoningRunning = false;
        node.requestId = null;
        node.editDraft = typeof node.editDraft === "string" ? node.editDraft : "";
      }
    }
    return true;
  } catch (err) {
    console.warn("[webui] restore state failed:", err);
    return false;
  }
}

function updateSessionTitle(session) {
  const firstUser = getActivePathNodes(session).find((node) => node.role === "user");
  if (!firstUser) return;
  const raw = (firstUser.text || "").trim();
  if (raw) {
    session.title = raw.slice(0, 28);
  }
}

function logDebug(session, line) {
  const ts = new Date().toLocaleTimeString();
  session.debugLines.push(`[${ts}] ${line}`);
  if (session.debugLines.length > 200) {
    session.debugLines.shift();
  }
  saveState();
  if (session.id === currentSessionId) {
    renderDebug();
  }
}

function setSessionStatus(session, status) {
  session.status = status;
  saveState();
  if (session.id === currentSessionId) {
    renderDebugStats();
    renderControls();
  }
}

function linkChild(session, parentId, childId, activate = true) {
  const parent = getNode(session, parentId);
  if (!parent) {
    throw new Error(`missing parent node: ${parentId}`);
  }
  if (!parent.childrenIds.includes(childId)) {
    parent.childrenIds.push(childId);
  }
  if (activate) {
    parent.activeChildId = childId;
  }
}

function createChildNode(session, parentId, role, text, extra = {}, activate = true) {
  const node = makeNode(role, text, parentId, extra);
  session.nodes[node.id] = node;
  linkChild(session, parentId, node.id, activate);
  if (activate) {
    session.activeLeafId = node.id;
  }
  saveState();
  return node;
}

function deepestActiveLeafId(session) {
  let current = getNode(session, session.rootId);
  while (current && current.activeChildId) {
    current = getNode(session, current.activeChildId);
  }
  return current ? current.id : session.rootId;
}

function setActiveChild(session, parentId, childId) {
  const parent = getNode(session, parentId);
  if (!parent || !parent.childrenIds.includes(childId)) return;
  parent.activeChildId = childId;
  session.activeLeafId = deepestActiveLeafId(session);
  updateSessionTitle(session);
  saveState();
}

function getPathNodeIdsTo(session, nodeId) {
  const ids = [];
  let current = getNode(session, nodeId);
  while (current && current.role !== "root") {
    ids.push(current.id);
    current = getNode(session, current.parentId);
  }
  ids.reverse();
  return ids;
}

function getActivePathNodes(session) {
  const nodes = [];
  let current = getNode(session, session.rootId);
  while (current && current.activeChildId) {
    current = getNode(session, current.activeChildId);
    if (!current) break;
    nodes.push(current);
  }
  return nodes;
}

function buildRequestMessages(session, assistantNodeId) {
  const assistantNode = getNode(session, assistantNodeId);
  if (!assistantNode) {
    throw new Error(`missing assistant node: ${assistantNodeId}`);
  }
  const ancestorIds = getPathNodeIdsTo(session, assistantNode.parentId);
  return ancestorIds
    .map((id) => getNode(session, id))
    .filter((node) => node && (node.role === "user" || node.role === "assistant"))
    .map((node) => ({ role: node.role, content: node.text || "" }));
}

function siblingInfo(session, node) {
  if (!node || !node.parentId) return null;
  const parent = getNode(session, node.parentId);
  if (!parent) return null;
  const siblings = parent.childrenIds.filter((childId) => {
    const child = getNode(session, childId);
    return child && child.role === node.role;
  });
  if (siblings.length <= 1) return null;
  const index = siblings.indexOf(node.id);
  return {
    parentId: parent.id,
    siblings,
    index,
    total: siblings.length,
    prevId: siblings[(index - 1 + siblings.length) % siblings.length],
    nextId: siblings[(index + 1) % siblings.length],
  };
}

function switchBranch(nodeId, direction) {
  const session = currentSession();
  if (!session || isRequestRunning(session)) return;
  const node = getNode(session, nodeId);
  const info = siblingInfo(session, node);
  if (!info) return;
  setActiveChild(session, info.parentId, direction < 0 ? info.prevId : info.nextId);
  if (editingMessageId && !getPathNodeIdsTo(session, session.activeLeafId).includes(editingMessageId)) {
    editingMessageId = null;
  }
  renderAll();
}

function startEditingUserNode(nodeId) {
  const session = currentSession();
  if (!session || isRequestRunning(session)) return;
  const node = getNode(session, nodeId);
  if (!node || node.role !== "user") return;
  editingMessageId = node.id;
  node.editDraft = node.text || "";
  renderAll();
}

function cancelInlineEdit() {
  const session = currentSession();
  if (session && editingMessageId) {
    const node = getNode(session, editingMessageId);
    if (node) {
      node.editDraft = node.text || "";
    }
  }
  editingMessageId = null;
  saveState();
  renderAll();
}

function renderSessions() {
  sessionList.innerHTML = "";
  for (const session of sessions) {
    const btn = document.createElement("button");
    btn.className = `session-item ${session.id === currentSessionId ? "active" : ""}`;
    const suffix = isRequestRunning(session) ? " (running)" : "";
    btn.textContent = `${session.title}${suffix}`;
    btn.addEventListener("click", () => {
      currentSessionId = session.id;
      editingMessageId = null;
      renderAll();
    });
    sessionList.appendChild(btn);
  }
}

function createBranchSwitcher(session, node) {
  const info = siblingInfo(session, node);
  if (!info) return null;
  const wrap = document.createElement("div");
  wrap.className = "branch-switcher";

  const label = document.createElement("span");
  label.className = "branch-label";
  label.textContent = `Branch ${info.index + 1}/${info.total}`;
  wrap.appendChild(label);

  const prevBtn = document.createElement("button");
  prevBtn.type = "button";
  prevBtn.className = "ghost small";
  prevBtn.textContent = "Prev";
  prevBtn.disabled = isRequestRunning(session);
  prevBtn.addEventListener("click", () => switchBranch(node.id, -1));
  wrap.appendChild(prevBtn);

  const nextBtn = document.createElement("button");
  nextBtn.type = "button";
  nextBtn.className = "ghost small";
  nextBtn.textContent = "Next";
  nextBtn.disabled = isRequestRunning(session);
  nextBtn.addEventListener("click", () => switchBranch(node.id, 1));
  wrap.appendChild(nextBtn);

  return wrap;
}

function createActionBar(session, node) {
  const bar = document.createElement("div");
  bar.className = "msg-actions";
  const isEditingThisNode = editingMessageId === node.id;

  const switcher = createBranchSwitcher(session, node);
  if (switcher) {
    bar.appendChild(switcher);
  }

  if (node.role === "user" && !isEditingThisNode) {
    const editBtn = document.createElement("button");
    editBtn.type = "button";
    editBtn.className = "ghost small";
    editBtn.textContent = "Edit";
    editBtn.disabled = isRequestRunning(session);
    editBtn.addEventListener("click", () => startEditingUserNode(node.id));
    bar.appendChild(editBtn);
  }

  if (node.role === "assistant" && !isEditingThisNode) {
    const regenBtn = document.createElement("button");
    regenBtn.type = "button";
    regenBtn.className = "ghost small";
    regenBtn.textContent = "Regenerate";
    regenBtn.disabled = isRequestRunning(session);
    regenBtn.addEventListener("click", () => {
      void regenerateAssistant(node.id);
    });
    bar.appendChild(regenBtn);
  }

  return bar.childNodes.length > 0 ? bar : null;
}

function createInlineEditor(session, node) {
  if (editingMessageId !== node.id) return null;
  const wrap = document.createElement("div");
  wrap.className = "inline-editor";

  const input = document.createElement("textarea");
  input.className = "inline-editor-input";
  input.rows = 3;
  input.value = node.editDraft || node.text || "";
  input.addEventListener("input", () => {
    node.editDraft = input.value;
    saveState();
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      void submitEditedUserNode(node.id);
    }
  });
  wrap.appendChild(input);

  const actions = document.createElement("div");
  actions.className = "inline-editor-actions";

  const submitBtn = document.createElement("button");
  submitBtn.type = "button";
  submitBtn.className = "ghost small";
  submitBtn.textContent = "Submit";
  submitBtn.disabled = isRequestRunning(session);
  submitBtn.addEventListener("click", () => {
    void submitEditedUserNode(node.id);
  });
  actions.appendChild(submitBtn);

  const cancelBtnEl = document.createElement("button");
  cancelBtnEl.type = "button";
  cancelBtnEl.className = "ghost small";
  cancelBtnEl.textContent = "Cancel";
  cancelBtnEl.disabled = isRequestRunning(session);
  cancelBtnEl.addEventListener("click", () => cancelInlineEdit());
  actions.appendChild(cancelBtnEl);

  wrap.appendChild(actions);
  setTimeout(() => input.focus(), 0);
  return wrap;
}

function renderChat() {
  const session = currentSession();
  if (!session) return;
  const prevTop = chatLog.scrollTop;
  const stickToBottom = chatLog.scrollHeight - chatLog.clientHeight - chatLog.scrollTop < 24;
  chatLog.innerHTML = "";

  for (const node of getActivePathNodes(session)) {
    const el = document.createElement("div");
    el.className = `msg ${node.role}`;

    if (node.role === "assistant") {
      const details = document.createElement("details");
      details.className = "assistant-reasoning";
      details.open = node.reasoningOpen !== false;
      const summary = document.createElement("summary");
      summary.textContent = node.reasoningRunning ? "思考中..." : "思考过程";
      const body = document.createElement("pre");
      body.className = "assistant-reasoning-body";
      body.textContent = typeof node.reasoning === "string" && node.reasoning.length > 0
        ? node.reasoning
        : node.reasoningRunning ? "思考中..." : "（无思考内容）";
      details.addEventListener("toggle", () => {
        node.reasoningOpen = details.open;
        saveState();
      });
      details.appendChild(summary);
      details.appendChild(body);
      el.appendChild(details);

      const answer = document.createElement("div");
      answer.className = "assistant-answer";
      answer.textContent = node.text || "";
      el.appendChild(answer);
    } else {
      const text = document.createElement("div");
      text.textContent = node.text || "";
      el.appendChild(text);
      const inlineEditor = createInlineEditor(session, node);
      if (inlineEditor) {
        el.appendChild(inlineEditor);
      }
    }

    const actionBar = createActionBar(session, node);
    if (actionBar) {
      el.appendChild(actionBar);
    }
    chatLog.appendChild(el);
  }

  if (stickToBottom) {
    chatLog.scrollTop = chatLog.scrollHeight;
  } else {
    chatLog.scrollTop = prevTop;
  }
}

async function refreshKvStats(session, reason) {
  try {
    const url = `${serverBase.value.replace(/\/$/, "")}/debug/kv_cache_stats`;
    const resp = await fetch(url);
    if (!resp.ok) {
      if (resp.status !== 404) {
        logDebug(session, `kv stats http error status=${resp.status}`);
      }
      return;
    }
    const stats = await resp.json();
    const allocator = stats?.allocator || {};
    const runtime = stats?.runtime || {};
    const line = `kv stats (${reason}) prefix_hits=${allocator.prefix_hits ?? "-"} prefix_misses=${allocator.prefix_misses ?? "-"} prefix_saved_tokens=${allocator.prefix_saved_tokens ?? "-"} used_tokens=${runtime.used_tokens ?? "-"} peak_used_tokens=${runtime.peak_used_tokens ?? "-"}`;
    lastKvStats = stats;
    logDebug(session, line);
    if (session.id === currentSessionId) {
      renderDebugStats();
    }
  } catch (err) {
    logDebug(session, `kv stats fetch error: ${String(err)}`);
  }
}

function renderDebugStats() {
  const session = currentSession();
  if (!session) return;
  const ttft = session.firstChunkAtMs > 0 ? `${session.firstChunkAtMs - session.reqStartedAtMs}ms` : "-";
  const elapsed = session.reqStartedAtMs > 0 ? `${nowMs() - session.reqStartedAtMs}ms` : "-";
  const allocator = lastKvStats?.allocator || {};
  debugStats.textContent = `status=${session.status} request_id=${session.activeRequestId || "-"} chunks=${session.chunkCount} ttft=${ttft} elapsed=${elapsed} prefix_hits=${allocator.prefix_hits ?? "-"} prefix_misses=${allocator.prefix_misses ?? "-"}`;
}

function renderDebug() {
  const session = currentSession();
  if (!session) return;
  debugLog.textContent = session.debugLines.join("\n");
  debugLog.scrollTop = debugLog.scrollHeight;
  renderDebugStats();
}

function renderControls() {
  const session = currentSession();
  if (!session) return;
  const running = isRequestRunning(session);
  sendBtn.disabled = false;
  sendBtn.classList.toggle("is-running", running);
  sendBtn.setAttribute("aria-label", running ? "Cancel" : "Send");
  sendBtn.title = running ? "Cancel running request" : "Send";
  promptInput.disabled = running;
}

function renderAll() {
  const session = currentSession();
  sessionTitle.textContent = session ? `NovaInfer Chat - ${session.title}` : "NovaInfer Chat";
  renderSessions();
  renderChat();
  renderDebug();
  renderControls();
}

function buildSamplingPayload(payload) {
  if (useServerDefaults.checked) {
    return payload;
  }
  const maxTokens = Number.parseInt(maxTokensInput.value, 10);
  const topK = Number.parseInt(topKInput.value, 10);
  const topP = Number.parseFloat(topPInput.value);
  const temperature = Number.parseFloat(temperatureInput.value);
  if (Number.isFinite(maxTokens) && maxTokens > 0) payload.max_tokens = maxTokens;
  if (Number.isFinite(topK) && topK >= 0) payload.top_k = topK;
  if (Number.isFinite(topP) && topP >= 0) payload.top_p = topP;
  if (Number.isFinite(temperature) && temperature >= 0) payload.temperature = temperature;
  return payload;
}

async function streamAssistantNode(session, assistantNodeId) {
  const assistantNode = getNode(session, assistantNodeId);
  if (!assistantNode) return;

  const controller = new AbortController();
  liveControllers.set(session.id, controller);

  assistantNode.text = "";
  assistantNode.reasoning = "";
  assistantNode.reasoningOpen = true;
  assistantNode.reasoningRunning = true;
  assistantNode.requestId = null;
  session.reqStartedAtMs = nowMs();
  session.firstChunkAtMs = 0;
  session.chunkCount = 0;
  session.activeRequestId = PENDING_REQUEST_ID;
  setSessionStatus(session, "request_sent");
  renderSessions();
  if (isVisibleSession(session)) {
    renderChat();
  }

  const payload = buildSamplingPayload({
    model: "qwen2",
    stream: true,
    include_reasoning: true,
    messages: buildRequestMessages(session, assistantNodeId),
  });

  logDebug(session, `send branch payload messages=${payload.messages.length}`);

  try {
    const url = `${serverBase.value.replace(/\/$/, "")}/v1/chat/completions`;
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!resp.ok || !resp.body) {
      assistantNode.text = `request failed: http ${resp.status}`;
      assistantNode.reasoningRunning = false;
      logDebug(session, `http error status=${resp.status}`);
      setSessionStatus(session, "http_error");
      return;
    }

    logDebug(session, `http ok status=${resp.status}`);
    setSessionStatus(session, "stream_open");

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
          logDebug(session, "stream done");
          assistantNode.reasoningRunning = false;
          session.activeRequestId = null;
          setSessionStatus(session, "done");
          renderSessions();
          if (isVisibleSession(session)) {
            renderChat();
          }
          void refreshKvStats(session, "done");
          continue;
        }

        const obj = JSON.parse(payloadText);
        session.activeRequestId = obj.request_id || session.activeRequestId;
        assistantNode.requestId = session.activeRequestId;
        if (session.firstChunkAtMs === 0) {
          session.firstChunkAtMs = nowMs();
          logDebug(session, `first chunk ttft=${session.firstChunkAtMs - session.reqStartedAtMs}ms`);
        }
        session.chunkCount += 1;
        const delta = obj?.choices?.[0]?.delta?.content || "";
        const reasoningDelta = obj?.choices?.[0]?.delta?.reasoning || "";
        const doneFlag = Boolean(obj?.is_finished);
        const tokenId = obj?.token_id;
        logDebug(
          session,
          `chunk#${session.chunkCount} req=${session.activeRequestId} token_id=${tokenId} done=${doneFlag} delta_len=${delta.length} reasoning_len=${reasoningDelta.length}`
        );
        setSessionStatus(session, doneFlag ? "finishing" : "streaming");
        if (delta.length > 0) {
          assistantNode.text += delta;
        }
        if (reasoningDelta.length > 0) {
          assistantNode.reasoning += reasoningDelta;
        }
        if (doneFlag) {
          assistantNode.reasoningRunning = false;
        }
        saveState();
        if (isVisibleSession(session)) {
          renderChat();
        }
      }
    }

    if (session.activeRequestId === null) {
      setSessionStatus(session, "idle");
      void refreshKvStats(session, "stream_closed");
    } else {
      setSessionStatus(session, "stream_closed_without_done");
    }
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      logDebug(session, `request aborted request_id=${session.activeRequestId}`);
      setSessionStatus(session, "cancelled");
      void refreshKvStats(session, "cancel");
    } else {
      throw err;
    }
  } finally {
    assistantNode.reasoningRunning = false;
    session.activeRequestId = null;
    liveControllers.delete(session.id);
    updateSessionTitle(session);
    saveState();
    if (isVisibleSession(session)) {
      renderAll();
    } else {
      renderSessions();
    }
  }
}

async function regenerateAssistant(nodeId) {
  const session = currentSession();
  if (!session || isRequestRunning(session)) return;
  const node = getNode(session, nodeId);
  if (!node || node.role !== "assistant" || !node.parentId) return;
  editingMessageId = null;
  const regenNode = createChildNode(
    session,
    node.parentId,
    "assistant",
    "",
    { reasoning: "", reasoningOpen: true, reasoningRunning: true },
    true
  );
  renderAll();
  await streamAssistantNode(session, regenNode.id);
}

async function submitEditedUserNode(nodeId) {
  const session = currentSession();
  if (!session || isRequestRunning(session)) return;
  const node = getNode(session, nodeId);
  if (!node || node.role !== "user" || !node.parentId) return;
  const text = (node.editDraft || "").trim();
  if (!text) return;

  const userNode = createChildNode(session, node.parentId, "user", text, {}, true);
  const assistantNode = createChildNode(
    session,
    userNode.id,
    "assistant",
    "",
    { reasoning: "", reasoningOpen: true, reasoningRunning: true },
    true
  );
  editingMessageId = null;
  updateSessionTitle(session);
  renderAll();
  await streamAssistantNode(session, assistantNode.id);
}

async function sendPrompt() {
  const session = currentSession();
  if (!session || isRequestRunning(session)) return;
  const prompt = promptInput.value.trim();
  if (!prompt) return;

  promptInput.value = "";
  resizePromptInput();

  const parentId = session.activeLeafId || session.rootId;
  const userNode = createChildNode(session, parentId, "user", prompt, {}, true);
  const assistantNode = createChildNode(
    session,
    userNode.id,
    "assistant",
    "",
    { reasoning: "", reasoningOpen: true, reasoningRunning: true },
    true
  );
  editingMessageId = null;
  updateSessionTitle(session);
  renderAll();
  await streamAssistantNode(session, assistantNode.id);
}

async function cancelRequest() {
  const session = currentSession();
  if (!session || !isRequestRunning(session)) return;
  const reqId = session.activeRequestId;
  const controller = liveControllers.get(session.id);
  if (controller) {
    controller.abort();
  }
  if (reqId && reqId !== PENDING_REQUEST_ID) {
    const url = `${serverBase.value.replace(/\/$/, "")}/v1/requests/${reqId}/cancel`;
    try {
      await fetch(url, { method: "POST" });
      logDebug(session, `cancel ok request_id=${reqId}`);
    } catch (err) {
      logDebug(session, `cancel transport error: ${String(err)}`);
    }
  } else {
    logDebug(session, "cancel before request_id assigned");
  }
}

function sendWithErrorHandling() {
  const session = currentSession();
  return sendPrompt().catch((err) => {
    if (!session) return;
    logDebug(session, `request error: ${String(err)}`);
    setSessionStatus(session, "request_error");
    const activeLeaf = getNode(session, session.activeLeafId);
    if (activeLeaf && activeLeaf.role === "assistant" && !activeLeaf.text) {
      activeLeaf.text = `request error: ${String(err)}`;
      activeLeaf.reasoningRunning = false;
    }
    session.activeRequestId = null;
    liveControllers.delete(session.id);
    saveState();
    if (isVisibleSession(session)) {
      renderAll();
    } else {
      renderSessions();
    }
  });
}

function cancelWithErrorHandling() {
  const session = currentSession();
  return cancelRequest().catch((err) => {
    if (!session) return;
    logDebug(session, `cancel error: ${String(err)}`);
    setSessionStatus(session, "cancel_error");
  });
}

newSessionBtn.addEventListener("click", () => {
  const session = createSession();
  sessions.push(session);
  currentSessionId = session.id;
  editingMessageId = null;
  logDebug(session, `session created id=${session.id}`);
  saveState();
  renderAll();
});

sendBtn.addEventListener("click", () => {
  const session = currentSession();
  if (!session) return;
  if (isRequestRunning(session)) {
    void cancelWithErrorHandling();
    return;
  }
  void sendWithErrorHandling();
});

promptInput.addEventListener("keydown", (e) => {
  if (e.key !== "Enter" || e.shiftKey || e.isComposing) {
    return;
  }
  e.preventDefault();
  const session = currentSession();
  if (!session || isRequestRunning(session)) return;
  void sendWithErrorHandling();
});

promptInput.addEventListener("input", () => {
  resizePromptInput();
});

if (!restoreState()) {
  const initial = createSession();
  sessions.push(initial);
  currentSessionId = initial.id;
  saveState();
}

bindExclusivePanels();
resizePromptInput();
renderAll();
