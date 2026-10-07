import React, { useCallback, useEffect, useRef, useState } from "react";
import { api, downloadFile, downloadSvgAsPng, fileAsBase64, postJson, streamChat, transcribeRecording } from "./api.js";

const NAV = [
  ["assistant", "⌂", "Assistant"],
  ["library", "▤", "Knowledge Library"],
  ["lab", "⌁", "Schematic & Prototype Lab"],
  ["notebook", "▧", "Research Notebook"],
  ["experiments", "⌖", "Experiment Log"],
  ["memory", "✦", "Memory"],
];

function newConversationId() {
  return window.crypto?.randomUUID?.().replaceAll("-", "") || `${Date.now()}_${Math.random().toString(36).slice(2, 12)}`;
}

function conversationPayload(draft) {
  const firstQuestion = draft.find((message) => message.role === "user")?.text || "Saved conversation";
  return {
    title: firstQuestion.replace(/\s+/g, " ").trim().slice(0, 100) || "Saved conversation",
    messages: draft.map((message) => ({
      id: message.id || "",
      role: message.role,
      text: message.text || "",
      image_name: message.imageName || null,
      sources: message.sources || null,
    })),
  };
}

function ErrorBanner({ children }) {
  return children ? <div className="alert error" role="alert">{String(children)}</div> : null;
}

function renderInlineMarkdown(value, prefix) {
  const source = String(value ?? "");
  const tokenPattern = /(\[[^\]]+\]\(https?:\/\/[^)\s]+\)|`[^`\n]+`|\*\*\*.+?\*\*\*|\*\*.+?\*\*|__.+?__|\*[^*\n]+?\*|_[^_\n]+?_)/g;
  const parts = [];
  let cursor = 0;
  let match;
  let index = 0;

  while ((match = tokenPattern.exec(source))) {
    if (match.index > cursor) parts.push(source.slice(cursor, match.index));
    const token = match[0];
    const key = `${prefix}-${index++}`;
    const link = token.match(/^\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)$/);
    if (link) {
      parts.push(<a key={key} href={link[2]} target="_blank" rel="noreferrer">{renderInlineMarkdown(link[1], key)}</a>);
    } else if (token.startsWith("`")) {
      parts.push(<code key={key}>{token.slice(1, -1)}</code>);
    } else if (token.startsWith("***")) {
      parts.push(<strong key={key}><em>{renderInlineMarkdown(token.slice(3, -3), key)}</em></strong>);
    } else if (token.startsWith("**") || token.startsWith("__")) {
      parts.push(<strong key={key}>{renderInlineMarkdown(token.slice(2, -2), key)}</strong>);
    } else {
      parts.push(<em key={key}>{renderInlineMarkdown(token.slice(1, -1), key)}</em>);
    }
    cursor = tokenPattern.lastIndex;
  }
  if (cursor < source.length) parts.push(source.slice(cursor));
  return parts;
}

function MarkdownText({ text, className = "" }) {
  const lines = String(text ?? "").replace(/\r\n?/g, "\n").split("\n");
  const blocks = [];
  let index = 0;
  const isBlockStart = (line) => /^\s*(```|#{1,6}\s+|>|[-+*]\s+|\d+[.)]\s+|(?:---+|\*\*\*+|___+)\s*$)/.test(line);

  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) { index += 1; continue; }

    const fence = line.match(/^\s*```(.*)$/);
    if (fence) {
      const code = [];
      index += 1;
      while (index < lines.length && !/^\s*```/.test(lines[index])) code.push(lines[index++]);
      if (index < lines.length) index += 1;
      blocks.push(<pre key={`block-${blocks.length}`}><code>{code.join("\n")}</code></pre>);
      continue;
    }

    const heading = line.match(/^\s*(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (heading) {
      const level = Math.min(6, heading[1].length + 1);
      const Heading = `h${level}`;
      blocks.push(<Heading className="md-heading" key={`block-${blocks.length}`}>{renderInlineMarkdown(heading[2], `heading-${blocks.length}`)}</Heading>);
      index += 1;
      continue;
    }

    if (/^\s*(?:---+|\*\*\*+|___+)\s*$/.test(line)) {
      blocks.push(<hr key={`block-${blocks.length}`} />);
      index += 1;
      continue;
    }

    if (/^\s*>/.test(line)) {
      const quote = [];
      while (index < lines.length && /^\s*>/.test(lines[index])) quote.push(lines[index++].replace(/^\s*>\s?/, ""));
      blocks.push(<blockquote key={`block-${blocks.length}`}>{quote.map((part, i) => <React.Fragment key={i}>{i > 0 && <br />}{renderInlineMarkdown(part, `quote-${blocks.length}-${i}`)}</React.Fragment>)}</blockquote>);
      continue;
    }

    const bullet = line.match(/^\s*([-+*])\s+(.*)$/);
    const numbered = line.match(/^\s*(\d+)[.)]\s+(.*)$/);
    if (bullet || numbered) {
      const ordered = !!numbered;
      const items = [];
      while (index < lines.length) {
        const item = ordered ? lines[index].match(/^\s*\d+[.)]\s+(.*)$/) : lines[index].match(/^\s*[-+*]\s+(.*)$/);
        if (!item) break;
        items.push(item[1]);
        index += 1;
      }
      const List = ordered ? "ol" : "ul";
      blocks.push(<List className="md-list" key={`block-${blocks.length}`}>{items.map((item, i) => <li key={i}>{renderInlineMarkdown(item, `list-${blocks.length}-${i}`)}</li>)}</List>);
      continue;
    }

    const paragraph = [line];
    index += 1;
    while (index < lines.length && lines[index].trim() && !isBlockStart(lines[index])) paragraph.push(lines[index++]);
    blocks.push(<p key={`block-${blocks.length}`}>{paragraph.map((part, i) => <React.Fragment key={i}>{i > 0 && <br />}{renderInlineMarkdown(part, `paragraph-${blocks.length}-${i}`)}</React.Fragment>)}</p>);
  }

  return <div className={`markdown-body ${className}`.trim()}>{blocks}</div>;
}

function SourcePanel({ sources }) {
  if (!sources || (!sources.local?.length && !sources.web?.length && !sources.experiments?.length)) return null;
  return (
    <details className="source-panel">
      <summary>Sources and evidence used</summary>
      {sources.local?.length > 0 && <section>
        <h4>Local library</h4>
        {sources.local.map((item, index) => {
          const meta = item.metadata || {};
          const citation = `${meta.filename || "local file"} · page ${meta.page || 1}`;
          return <article className="source-item" key={`${citation}-${index}`}>
            <strong>{citation}</strong>{meta.source_url && <a href={meta.source_url} target="_blank" rel="noreferrer">Original web page ↗</a>}
            <p>{item.text}</p>
          </article>;
        })}
      </section>}
      {sources.web?.length > 0 && <section>
        <h4>Web results</h4>
        {sources.web.map((item, index) => <article className="source-item" key={`${item.url}-${index}`}>
          <a href={item.url} target="_blank" rel="noreferrer"><strong>{item.title}</strong> ↗</a>
          <small className="evidence-status">{item.fetched_chars ? `Fetched page text · ${item.fetched_chars.toLocaleString()} characters` : "Preliminary search snippet · full page not fetched"}</small>
          <p>{item.content ? `${item.content.slice(0, 700)}${item.content.length > 700 ? "…" : ""}` : item.snippet}</p>
        </article>)}
      </section>}
      {sources.experiments?.length > 0 && <section>
        <h4>Your experiment log</h4>
        {sources.experiments.map((item, index) => <article className="source-item" key={`${item.id || item.project}-${index}`}>
          <strong>{item.project} · {item.experiment} · Trial {item.trial} · {item.logged_on}</strong>
          <p>{item.metric}: {item.value}{item.unit ? ` ${item.unit}` : ""}{item.independent_variable ? ` · ${item.independent_variable}${item.setting ? ` = ${item.setting}` : ""}` : ""}</p>
          {item.source && <small className="evidence-status">Reference: {item.source}</small>}
        </article>)}
      </section>}
    </details>
  );
}

function Toggle({ checked, onChange, label, help }) {
  return <label className="toggle-row" title={help}>
    <span><b>{label}</b><small>{help}</small></span>
    <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
    <i aria-hidden="true" />
  </label>;
}

function ChatPage({ useWeb }) {
  const [messages, setMessages] = useState([]);
  const [conversationId, setConversationId] = useState(newConversationId);
  const [savedChats, setSavedChats] = useState([]);
  const [historyStatus, setHistoryStatus] = useState("");
  const [question, setQuestion] = useState("");
  const [image, setImage] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [taskSteps, setTaskSteps] = useState([]);
  const [speaking, setSpeaking] = useState(() => localStorage.getItem("akito-speak-replies") === "true");
  const [dictating, setDictating] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [voiceMessage, setVoiceMessage] = useState("");
  const controllerRef = useRef(null);
  const recorderRef = useRef(null);
  const audioStreamRef = useRef(null);
  const audioChunksRef = useRef([]);
  const voiceControllerRef = useRef(null);
  const voiceTimeoutRef = useRef(null);
  const currentConversationRef = useRef(null);
  currentConversationRef.current = { id: conversationId, messages };

  const refreshSavedChats = useCallback(async () => {
    try { setSavedChats((await api("/api/conversations")).items || []); }
    catch (err) { setHistoryStatus(`History unavailable: ${err.message}`); }
  }, []);

  const persistConversation = useCallback(async (id, draft) => {
    if (!draft.length) return null;
    const response = await api(`/api/conversations/${encodeURIComponent(id)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(conversationPayload(draft)),
    });
    if (response?.item) setSavedChats((items) => [response.item, ...items.filter((item) => item.id !== id)].slice(0, 200));
    return response;
  }, []);

  useEffect(() => { refreshSavedChats(); }, [refreshSavedChats]);
  useEffect(() => {
    if (!messages.length || busy) return undefined;
    const id = conversationId;
    const draft = messages;
    const timer = window.setTimeout(async () => {
      setHistoryStatus("Saving chat locally…");
      try {
        const response = await persistConversation(id, draft);
        setHistoryStatus("Chat saved locally");
      } catch (err) { setHistoryStatus(`Could not save chat: ${err.message}`); }
    }, 700);
    return () => window.clearTimeout(timer);
  }, [messages, conversationId, busy, persistConversation]);

  useEffect(() => () => {
    const current = currentConversationRef.current;
    if (current?.messages.length) {
      fetch(`/api/conversations/${encodeURIComponent(current.id)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(conversationPayload(current.messages)),
      }).catch(() => {});
    }
    controllerRef.current?.abort();
    voiceControllerRef.current?.abort();
    window.clearTimeout(voiceTimeoutRef.current);
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.onstop = null;
      recorderRef.current.stop();
    }
    audioStreamRef.current?.getTracks().forEach((track) => track.stop());
    window.speechSynthesis?.cancel();
  }, []);

  const toggleDictation = async () => {
    if (dictating) {
      window.clearTimeout(voiceTimeoutRef.current);
      if (recorderRef.current?.state === "recording") {
        setDictating(false);
        setTranscribing(true);
        setVoiceMessage("Transcribing on this computer with local Whisper…");
        recorderRef.current.stop();
      }
      return;
    }
    if (transcribing) return;
    setVoiceMessage("");
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      setVoiceMessage("This browser does not support microphone recording. Try a current version of Chrome or Edge.");
      return;
    }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
      audioStreamRef.current = stream;
      const mimeType = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/mp4"].find((type) => MediaRecorder.isTypeSupported?.(type));
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      audioChunksRef.current = [];
      recorderRef.current = recorder;
      recorder.ondataavailable = (event) => { if (event.data?.size) audioChunksRef.current.push(event.data); };
      recorder.onstop = async () => {
        window.clearTimeout(voiceTimeoutRef.current);
        stream.getTracks().forEach((track) => track.stop());
        audioStreamRef.current = null;
        recorderRef.current = null;
        setDictating(false);
        const recording = new Blob(audioChunksRef.current, { type: recorder.mimeType || "audio/webm" });
        audioChunksRef.current = [];
        if (!recording.size) {
          setTranscribing(false);
          setVoiceMessage("No audio was captured. Check microphone permission and try again.");
          return;
        }
        const voiceController = new AbortController();
        voiceControllerRef.current = voiceController;
        setTranscribing(true);
        setVoiceMessage("Transcribing on this computer with local Whisper…");
        try {
          const result = await transcribeRecording(recording, voiceController.signal);
          const transcript = result.text?.trim();
          if (transcript) {
            setQuestion((current) => current ? `${current.trimEnd()} ${transcript}` : transcript);
            setVoiceMessage("Transcript added to your message. Review it, then send when ready.");
          } else setVoiceMessage("I couldn't detect speech in that recording. Try speaking closer to the microphone.");
        } catch (err) {
          if (err.name !== "AbortError") setVoiceMessage(err.message || "Local speech transcription failed.");
        } finally {
          voiceControllerRef.current = null;
          setTranscribing(false);
        }
      };
      recorder.onerror = () => setVoiceMessage("Microphone recording stopped unexpectedly. Check browser microphone permission and try again.");
      recorder.start();
      setDictating(true);
      setVoiceMessage("Recording… select Stop & transcribe when you’re done (30-second limit).");
      voiceTimeoutRef.current = window.setTimeout(() => {
        if (recorder.state === "recording") {
          setVoiceMessage("30-second limit reached. Transcribing your recording…");
          setDictating(false);
          setTranscribing(true);
          recorder.stop();
        }
      }, 30000);
    } catch (err) {
      stream?.getTracks().forEach((track) => track.stop());
      audioStreamRef.current = null;
      const reason = err.name === "NotAllowedError" || err.name === "SecurityError"
        ? "Allow microphone access for AKITO in your browser, then try again."
        : err.name === "NotFoundError"
          ? "No microphone was found. Connect one and try again."
          : err.message || "Could not start microphone recording.";
      setVoiceMessage(reason);
    }
  };

  const cancelTask = () => {
    controllerRef.current?.abort();
    voiceControllerRef.current?.abort();
    window.speechSynthesis?.cancel();
  };

  const startNewChat = async () => {
    if (busy) return;
    try {
      await persistConversation(conversationId, messages);
      setConversationId(newConversationId());
      setMessages([]);
      setQuestion("");
      setImage(null);
      setError("");
      setHistoryStatus("Previous chat saved");
    } catch (err) { setError(`Could not save the current chat before starting a new one: ${err.message}`); }
  };

  const openSavedChat = async (id) => {
    if (busy) return;
    try {
      await persistConversation(conversationId, messages);
      const record = await api(`/api/conversations/${encodeURIComponent(id)}`);
      setConversationId(record.id);
      setMessages((record.messages || []).map((message, index) => ({
        ...message,
        id: message.id || `saved-${id}-${index}`,
        imageName: message.image_name || "",
        imageUrl: null,
      })));
      setQuestion("");
      setImage(null);
      setError("");
      setHistoryStatus("Loaded saved chat");
    } catch (err) { setError(`Could not open the saved chat: ${err.message}`); }
  };

  const send = async (event) => {
    event.preventDefault();
    const ask = question.trim() || (image ? "Describe this image and identify visible contents. State uncertainty." : "");
    if (!ask || busy) return;
    setError("");
    setQuestion("");
    const imageUrl = image ? URL.createObjectURL(image) : null;
    const userMessage = { role: "user", text: ask, imageUrl, imageName: image?.name || "" };
    const answerId = `${Date.now()}-${Math.random()}`;
    setMessages((items) => [...items, userMessage, { id: answerId, role: "assistant", text: "", sources: null }]);
    setBusy(true);
    setTaskSteps([]);
    const controller = new AbortController();
    controllerRef.current = controller;
    let fullAnswer = "";
    try {
      const imageBase64 = image ? await fileAsBase64(image) : null;
      const history = messages.slice(-12).filter((item) => item.text?.trim()).map(({ role, text }) => ({ role, content: text }));
      await streamChat({ question: ask, use_web: useWeb, image_base64: imageBase64, image_name: image?.name, history },
        (sources) => setMessages((items) => items.map((item) => item.id === answerId ? { ...item, sources } : item)),
        (text) => { fullAnswer += text; setMessages((items) => items.map((item) => item.id === answerId ? { ...item, text: item.text + text } : item)); },
        (step) => setTaskSteps((items) => [...items.filter((item) => item.stage !== step.stage), step]),
        controller.signal,
      );
      if (!fullAnswer.trim()) {
        throw new Error("The local model returned an empty answer. Check that Ollama is running and the configured chat model is available.");
      }
      if (speaking && fullAnswer.trim()) {
        if (window.speechSynthesis && window.SpeechSynthesisUtterance) {
          const utterance = new SpeechSynthesisUtterance(fullAnswer);
          window.speechSynthesis.speak(utterance);
        } else setVoiceMessage("Speech output is not available in this browser.");
      }
    } catch (err) {
      if (err.name === "AbortError") {
        setTaskSteps((items) => [...items, { stage: "cancelled", status: "done", label: "Stopped by you" }]);
        setMessages((items) => items.map((item) => item.id === answerId ? { ...item, text: item.text || "Request cancelled." } : item));
      } else {
        setError(err.message);
        setMessages((items) => items.map((item) => item.id === answerId ? { ...item, text: item.text || "I couldn't complete that request." } : item));
      }
    } finally {
      controllerRef.current = null;
      setImage(null);
      setBusy(false);
    }
  };

  return <div className="page chat-page">
    <div className="page-heading"><div><span className="eyebrow">AKITO AI by Devon</span><p>Ask from your local library, search the web, or inspect an image.</p></div><div className="voice-controls"><button className={`button secondary ${dictating ? "recording" : ""}`} type="button" onClick={toggleDictation} disabled={transcribing}>{transcribing ? "Transcribing…" : dictating ? "■ Stop & transcribe" : "🎙 Push to talk"}</button><label className="voice-toggle"><input type="checkbox" checked={speaking} onChange={(event) => { setSpeaking(event.target.checked); localStorage.setItem("akito-speak-replies", String(event.target.checked)); if (!event.target.checked) window.speechSynthesis?.cancel(); }} /> Speak replies</label><small className="voice-privacy">Audio is transcribed locally. First use downloads Whisper Tiny to this computer.</small></div></div>
    <div className="chat-history-bar"><button className="button secondary" type="button" onClick={startNewChat} disabled={busy}>＋ New chat</button><details className="saved-history"><summary>◷ Saved chats <span>{savedChats.length}</span></summary><div className="saved-history-popover">{savedChats.length ? savedChats.map((item) => <button type="button" className="saved-chat-item" key={item.id} onClick={() => openSavedChat(item.id)} disabled={busy}><b>{item.title}</b><small>{item.message_count} messages · {item.updated_at ? new Date(item.updated_at).toLocaleString() : "saved locally"}</small><span>{item.preview}</span></button>) : <div className="empty-state">Chats are saved here after you send a message.</div>}</div></details><span className="history-status" role="status">{historyStatus}</span></div>
    {voiceMessage && <div className="alert info">{voiceMessage}</div>}
    <ErrorBanner>{error}</ErrorBanner>
    <div className="conversation">
      {messages.length === 0 && <div className="welcome-card"><span className="welcome-orb">A</span><h3>What are you researching today?</h3><p>Your local files stay in this project. Turn on web search when you want current sources.</p><div className="suggestion-row"><span>ESP32 design</span><span>Research methods</span><span>Analyze a circuit</span></div></div>}
      {messages.map((message, index) => <div className={`message ${message.role}`} key={message.id || index}>
        <div className="avatar">{message.role === "assistant" ? "A" : "Y"}</div>
        <div className="message-body"><div className="message-label">{message.role === "assistant" ? "Akito" : "You"}</div>
          {message.imageUrl && <img className="chat-image" src={message.imageUrl} alt="Uploaded for analysis" />}
          {message.imageName && !message.imageUrl && <small className="stored-image-note">Image attached: {message.imageName} · image data is not included in chat history.</small>}
          <div className={`message-text ${message.role === "user" ? "user-text" : ""}`}>
            {message.role === "assistant"
              ? (message.text ? <MarkdownText text={message.text} /> : (busy ? <span className="typing">Akito is responding…</span> : ""))
              : message.text}
          </div>
          {message.role === "assistant" && <SourcePanel sources={message.sources} />}
        </div>
      </div>)}
      {busy && <div className="task-tracker" role="status"><div><span className="spinner"/><b>AKITO task</b><span>{taskSteps.at(-1)?.label || "Starting…"}</span><button type="button" className="button secondary" onClick={cancelTask}>Cancel</button></div>{taskSteps.length > 0 && <ol>{taskSteps.map((step) => <li className={`task-${step.status}`} key={step.stage}><span>{step.label}</span>{step.detail && <small>{step.detail}</small>}</li>)}</ol>}</div>}
    </div>
    <form className="composer card" onSubmit={send}>
      <textarea value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask a research question or describe a prototype…" rows="3" disabled={busy} />
      <div className="composer-bottom">
        <label className="upload-chip">＋ Upload image<input type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => setImage(event.target.files?.[0] || null)} /></label>
        {image && <span className="file-chip">{image.name}<button type="button" aria-label="Remove image" onClick={() => setImage(null)}>×</button></span>}
        <span className="composer-hint">{useWeb ? "Web search on" : "Local library"}</span>
        <button className="button primary send-button" disabled={busy || (!question.trim() && !image)}>{busy ? "Working…" : "Send to Akito ↑"}</button>
      </div>
    </form>
  </div>;
}

function KnowledgePage({ health, refreshHealth }) {
  const [topic, setTopic] = useState("");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [selected, setSelected] = useState(new Set());
  const [searchOffset, setSearchOffset] = useState(0);
  const [hasMoreResults, setHasMoreResults] = useState(false);
  const [activeSearchTerm, setActiveSearchTerm] = useState("");
  const [files, setFiles] = useState([]);
  const [localResults, setLocalResults] = useState([]);
  const [working, setWorking] = useState("");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [learning, setLearning] = useState(null);
  const [learningEnabled, setLearningEnabled] = useState(true);
  const [learningTopics, setLearningTopics] = useState("");
  const [learningWorking, setLearningWorking] = useState(false);
  const learningInitialized = useRef(false);

  const refreshLearning = async () => {
    try {
      const state = await api("/api/automation/learning");
      setLearning(state);
      if (!learningInitialized.current) {
        setLearningEnabled(state.enabled);
        setLearningTopics((state.topics || []).join("\n"));
        learningInitialized.current = true;
      }
    } catch (err) { setError(`Could not load background learning status: ${err.message}`); }
  };

  useEffect(() => {
    refreshLearning();
    const timer = window.setInterval(refreshLearning, 5000);
    return () => window.clearInterval(timer);
  }, []);

  const saveLearningSettings = async () => {
    const topics = learningTopics.split("\n").map((value) => value.trim()).filter(Boolean);
    if (!topics.length) { setError("Add at least one learning topic."); return; }
    setLearningWorking(true); setError(""); setNotice("");
    try {
      const state = await api("/api/automation/learning", {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: learningEnabled, topics }),
      });
      setLearning(state);
      setNotice(state.enabled ? "Continuous web discovery is enabled. It will keep searching until you pause it." : "Web discovery is paused.");
      refreshHealth();
    } catch (err) { setError(err.message); }
    finally { setLearningWorking(false); }
  };

  const runLearningNow = async () => {
    setLearningWorking(true); setError(""); setNotice("");
    try {
      const response = await api("/api/automation/learning/run", { method: "POST" });
      if (response.accepted) setNotice("Continuous web discovery started. It keeps searching until you pause it.");
      else setNotice(response.detail || "Continuous discovery is already running.");
      refreshLearning();
    } catch (err) { setError(err.message); }
    finally { setLearningWorking(false); }
  };

  const search = async (append = false) => {
    const fallbackTopic = (health?.model_topics || []).slice(0, 3).join(", ");
    const searchTerm = topic.trim() || fallbackTopic;
    if (!searchTerm) return setError("Enter a topic or add expertise areas to Modelfile.");
    const queryText = `${searchTerm} technical documentation research`;
    const offset = append && activeSearchTerm === queryText ? searchOffset : 0;
    setWorking(append ? "more" : "search"); setError(""); setNotice("");
    try {
      const response = await postJson("/api/web/search", { query: queryText, offset });
      const page = response.results || [];
      if (append && activeSearchTerm === queryText) {
        setResults((current) => {
          const known = new Set(current.map((item) => item.url));
          return [...current, ...page.filter((item) => !known.has(item.url))];
        });
        setSelected((current) => new Set([...current, ...page.map((item) => item.url)]));
      } else {
        setResults(page);
        setSelected(new Set(page.map((item) => item.url)));
      }
      setActiveSearchTerm(queryText);
      setSearchOffset(response.next_offset ?? offset + 10);
      setHasMoreResults(Boolean(response.has_more));
      if (!page.length) setNotice("No more search results were returned for this query.");
    } catch (err) { setError(err.message); }
    finally { setWorking(""); }
  };

  const searchAllPages = async () => {
    const fallbackTopic = (health?.model_topics || []).slice(0, 3).join(", ");
    const searchTerm = topic.trim() || fallbackTopic;
    if (!searchTerm) return setError("Enter a topic or add expertise areas to Modelfile.");
    const queryText = `${searchTerm} technical documentation research`;
    let offset = 0;
    let collected = [];
    const selectedUrls = new Set();
    setWorking("all"); setError(""); setNotice("Searching every available result page…");
    setActiveSearchTerm(queryText); setHasMoreResults(false); setResults([]); setSelected(new Set());
    try {
      while (true) {
        const response = await postJson("/api/web/search", { query: queryText, offset });
        const page = response.results || [];
        const known = new Set(collected.map((item) => item.url));
        const newResults = page.filter((item) => item.url && !known.has(item.url));
        if (!newResults.length) break;
        collected = [...collected, ...newResults];
        newResults.forEach((item) => selectedUrls.add(item.url));
        setResults(collected);
        setSelected(new Set(selectedUrls));
        setNotice(`Searching every result page… ${collected.length} distinct results loaded.`);
        offset = response.next_offset ?? offset + 10;
        setSearchOffset(offset);
        if (!response.has_more) break;
      }
      setHasMoreResults(false);
      setNotice(collected.length
        ? `Loaded ${collected.length} distinct results. All are selected for training; uncheck any you do not want.`
        : "No search results were returned for this topic.");
    } catch (err) {
      setError(collected.length
        ? `Search stopped after loading ${collected.length} results: ${err.message}`
        : err.message);
      setResults(collected); setSelected(new Set(selectedUrls));
    } finally { setWorking(""); }
  };

  const train = async () => {
    const chosen = results.filter((item) => selected.has(item.url));
    setWorking("train"); setError(""); setNotice("Searching, downloading selected pages, then indexing them…");
    try {
      const response = await postJson("/api/learn", { topic: topic.trim() || (health?.model_topics || []).slice(0, 3).join(", "), results: chosen });
      const added = response.added || [];
      setNotice(added.length ? `Added ${added.length} source page(s). The local library now has ${response.indexed_chunks} indexed chunks.` : response.skipped || "No readable pages were added.");
      refreshHealth();
    } catch (err) { setError(err.message); setNotice(""); }
    finally { setWorking(""); }
  };

  const upload = async () => {
    if (!files.length) return;
    setWorking("upload"); setError(""); setNotice("Saving files and rebuilding the local index…");
    const data = new FormData();
    files.forEach((file) => data.append("files", file));
    try {
      const response = await api("/api/knowledge/upload", { method: "POST", body: data });
      setNotice(`Added ${response.saved.length} file(s). ${response.indexed_chunks} chunks are indexed.`);
      setFiles([]);
      refreshHealth();
    } catch (err) { setError(err.message); setNotice(""); }
    finally { setWorking(""); }
  };

  const searchLocal = async (event) => {
    event.preventDefault(); setError("");
    if (!query.trim()) return;
    try {
      const response = await postJson("/api/knowledge/search", { query, limit: 8 });
      setLocalResults(response.results || []);
    } catch (err) { setError(err.message); }
  };

  return <div className="page">
    <div className="page-heading"><div><span className="eyebrow">LOCAL KNOWLEDGE</span><h2>Knowledge Library</h2><p>Find, curate, and index materials that Akito can cite in later answers.</p></div><div className="stat-card"><strong>{health?.indexed_chunks ?? "—"}</strong><span>indexed chunks</span></div></div>
    <ErrorBanner>{error}</ErrorBanner>{notice && <div className={`alert ${working ? "info" : "success"}`}>{working && <span className="spinner" />}{notice}</div>}
    <div className="two-column">
      <section className="card trainer-card"><div className="card-title"><span className="icon-box gold">⌕</span><div><h3>Web Trainer</h3><p>Search, page through results, and train on every selected source.</p></div></div>
        <label className="field-label" htmlFor="topic">Research topic</label>
        <input id="topic" className="text-input" value={topic} onChange={(event) => setTopic(event.target.value)} placeholder="e.g. ESP32 low-power sensor design" />
        <div className="button-row"><button className="button secondary" disabled={!!working} onClick={() => search(false)}>{working === "search" ? "Searching…" : "Search Web"}</button><button className="button secondary" disabled={!!working} onClick={searchAllPages}>{working === "all" ? "Searching all pages…" : "Search all pages"}</button><button className="button primary" disabled={!!working || !results.length || !selected.size} onClick={train}>{working === "train" ? "Adding knowledge…" : `Train Model · Add ${selected.size} selected`}</button></div>
        <p className="subtle">This grows the local retrieval library; it does not fine-tune model weights. Blank topic uses Modelfile expertise areas.</p>
        {results.length > 0 && <div className="result-list"><div className="button-row result-actions"><h4>{results.length} results loaded</h4><button className="button secondary" type="button" onClick={() => setSelected(new Set(results.map((item) => item.url)))}>Select all loaded</button><button className="button secondary" type="button" onClick={() => setSelected(new Set())}>Clear selection</button></div>{results.map((item) => <label className="web-result" key={item.url}>
          <input type="checkbox" checked={selected.has(item.url)} onChange={(event) => setSelected((old) => {
            const next = new Set(old);
            if (event.target.checked) next.add(item.url);
            else if (!event.target.checked) next.delete(item.url);
            return next;
          })} />
          <span><b>{item.title}</b><a href={item.url} target="_blank" rel="noreferrer">{item.url}</a><small>{item.snippet || "No search excerpt available."}</small></span>
        </label>)}{hasMoreResults && <button className="button secondary full" type="button" disabled={!!working} onClick={() => search(true)}>{working === "more" ? "Loading more results…" : "Load more results"}</button>}</div>}
      </section>
      <section className="card"><div className="card-title"><span className="icon-box sage">＋</span><div><h3>Add local files or datasets</h3><p>PDF, TXT, Markdown, and CSV files are stored below documents/.</p></div></div>
        <label className="drop-zone"><span className="drop-icon">↥</span><b>{files.length ? `${files.length} file(s) selected` : "Choose research files or datasets"}</b><small>PDF · TXT · MD · CSV · up to 50 MB each</small><input type="file" multiple accept=".pdf,.txt,.md,.csv,text/csv" onChange={(event) => setFiles(Array.from(event.target.files || []))} /></label>
        {files.length > 0 && <ul className="file-list">{files.map((file) => <li key={`${file.name}-${file.size}`}>{file.name} <span>{(file.size / 1024 / 1024).toFixed(1)} MB</span></li>)}</ul>}
        <button className="button primary full" disabled={!!working || !files.length} onClick={upload}>{working === "upload" ? "Indexing files…" : "Add files to local knowledge"}</button>
      </section>
    </div>
    <section className="card learning-automation-card">
      <div className="card-title"><span className="icon-box sage">↻</span><div><h3>Continuous Web Discovery &amp; Learning</h3><p>Akito keeps searching new public-web sources and saves accepted pages with citations into its research library.</p></div></div>
      <div className="learning-settings-grid">
        <Toggle checked={learningEnabled} onChange={setLearningEnabled} label="Keep searching until I pause it" help="No site-count or session-time cap. The API and computer must stay on and connected to the Internet." />
      </div>
      <label className="field-label learning-topics-label">Topics to rotate through<textarea className="text-input learning-topics" rows="5" value={learningTopics} onChange={(event) => setLearningTopics(event.target.value)} placeholder="One topic per line; defaults include current embedded research trends." /></label>
      <p className="subtle">Discovery advances through result pages without wrapping back to earlier pages. It skips sites already learned, but a failed page only gets a temporary URL-specific retry cooldown. It pauses briefly between requests and retries provider failures. Web pages and citations are stored locally for retrieval; this does not retrain model weights.</p>
      <div className="button-row"><button className="button secondary" type="button" disabled={learningWorking} onClick={saveLearningSettings}>{learningWorking ? "Saving…" : learningEnabled ? "Save & keep searching" : "Save & pause"}</button><button className="button primary" type="button" disabled={learningWorking || !learningEnabled} onClick={runLearningNow}>{learningWorking ? "Working…" : "Start / resume now"}</button></div>
      {learning && <div className={`learning-status status-${learning.status || "idle"}`} role="status"><div><b>{(learning.status || "idle").replaceAll("_", " ")}</b><span>{learning.progress}</span></div><p>{learning.last_summary}</p><small>Unique sites learned: {learning.unique_sites_learned ?? 0}</small>{learning.last_run_at && <small>{learning.enabled ? "Running since" : "Last session ended"}: {new Date(learning.last_run_at).toLocaleString()}</small>}{learning.last_activity_at && <small>Last activity: {new Date(learning.last_activity_at).toLocaleString()}</small>}{learning.last_error && <details><summary>Run warnings or errors</summary><pre>{learning.last_error}</pre></details>}</div>}
    </section>
    <section className="card search-card"><div><span className="eyebrow">CITABLE SOURCES</span><h3>Search local references</h3></div>
      <form className="search-bar" onSubmit={searchLocal}><input className="text-input" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search indexed files…"/><button className="button primary">Search Library</button></form>
      <div className="local-results">{localResults.map((item, index) => <details className="local-result" key={`${item.metadata.filename}-${index}`}><summary><b>{item.metadata.filename}</b><span>page {item.metadata.page || 1}</span></summary><p>{item.text}</p>{item.metadata.source_url && <a href={item.metadata.source_url} target="_blank" rel="noreferrer">Original web source ↗</a>}</details>)}</div>
    </section>
  </div>;
}

function SchematicReader({ refreshHealth, useWeb }) {
  const [file, setFile] = useState(null);
  const [focus, setFocus] = useState("");
  const [analysisText, setAnalysisText] = useState("");
  const [notes, setNotes] = useState("");
  const [reviewed, setReviewed] = useState(false);
  const [crosscheck, setCrosscheck] = useState(null);
  const [selectedWebUrls, setSelectedWebUrls] = useState(new Set());
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const read = async () => {
    if (!file) return;
    setBusy("analyze"); setError(""); setNotice("");
    const data = new FormData(); data.append("file", file); data.append("focus", focus);
    try {
      const response = await api("/api/schematic/analyze", { method: "POST", body: data });
      setAnalysisText(JSON.stringify(response.analysis, null, 2)); setCrosscheck(null); setSelectedWebUrls(new Set()); setReviewed(false);
    } catch (err) { setError(err.message); }
    finally { setBusy(""); }
  };

  const compare = async () => {
    setBusy("crosscheck"); setError("");
    try {
      const response = await postJson("/api/schematic/crosscheck", { analysis: JSON.parse(analysisText), use_web: useWeb });
      setCrosscheck(response);
      setSelectedWebUrls(new Set((response.sources?.web || []).map((item) => item.url)));
    } catch (err) { setError(err instanceof SyntaxError ? "Fix the extracted JSON before checking it." : err.message); }
    finally { setBusy(""); }
  };

  const addWebSources = async () => {
    const results = (crosscheck?.sources?.web || []).filter((item) => selectedWebUrls.has(item.url));
    if (!results.length) return;
    setBusy("curate"); setError(""); setNotice("");
    try {
      const response = await postJson("/api/learn", { topic: focus.trim() || "schematic component datasheets", results });
      const added = response.added || [];
      setNotice(added.length
        ? `Added ${added.length} web reference(s) to the local library (${response.indexed_chunks} indexed chunks).`
        : response.skipped || "No readable web pages were added.");
      refreshHealth();
    } catch (err) { setError(err.message); }
    finally { setBusy(""); }
  };

  const save = async () => {
    if (!file || !reviewed) return;
    setBusy("save"); setError("");
    try {
      const response = await postJson("/api/schematic/save", {
        filename: file.name,
        image_base64: await fileAsBase64(file),
        analysis: JSON.parse(analysisText),
        reviewer_notes: notes,
      });
      setNotice(`Reviewed schematic saved and indexed at ${response.path} (${response.indexed_chunks} chunks).`);
      refreshHealth();
    } catch (err) { setError(err.message); }
    finally { setBusy(""); }
  };

  return <div className="two-column schematic-grid">
    <section className="card"><div className="card-title"><span className="icon-box coral">⌗</span><div><h3>Read a schematic</h3><p>Extract visible components, labels, and wiring from an image.</p></div></div>
      <div className="notice-callout">AI extraction is a draft. Verify every part and net against the original diagram before adding it to your knowledge base.</div>
      <label className="drop-zone compact"><span className="drop-icon">▧</span><b>{file?.name || "Choose a schematic image"}</b><small>PNG · JPEG · WebP</small><input type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => { setFile(event.target.files?.[0] || null); setAnalysisText(""); setCrosscheck(null); setReviewed(false); }} /></label>
      {file && <img className="schematic-preview" src={URL.createObjectURL(file)} alt="Schematic preview" />}
      <label className="field-label">Focus area</label><input className="text-input" value={focus} onChange={(event) => setFocus(event.target.value)} placeholder="Power path, MCU pins, sensor connections…" />
      <button className="button primary full" disabled={!file || !!busy} onClick={read}>{busy === "analyze" ? "Reading image…" : "Analyze schematic"}</button>
      <p className="subtle">Confidence is the model’s estimate, not measured accuracy. Web Search in the sidebar adds internet results to datasheet curation.</p>
    </section>
    <section className="card extraction-card"><div className="card-title"><span className="icon-box gold">✓</span><div><h3>Researcher review</h3><p>Correct the extraction and compare it with local references and optional web results.</p></div></div>
      <ErrorBanner>{error}</ErrorBanner>{notice && <div className="alert success">{notice}</div>}
      {analysisText ? <>
        <label className="field-label">Extracted JSON · edit labels and connections directly</label>
        <textarea className="code-input" value={analysisText} onChange={(event) => { setAnalysisText(event.target.value); setCrosscheck(null); }} spellCheck="false" />
        <label className="field-label">Additional corrections / unclear regions</label><textarea className="text-input notes-input" value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="Record any corrections you made or details you could not verify…" />
        <label className="review-check"><input type="checkbox" checked={reviewed} onChange={(event) => setReviewed(event.target.checked)} /> I reviewed this extraction against the image.</label>
        <div className="button-row"><button className="button secondary" disabled={!!busy} onClick={compare}>{busy === "crosscheck" ? "Curating sources…" : useWeb ? "Cross-check local + web sources" : "Cross-check local datasheets"}</button><button className="button primary" disabled={!reviewed || !!busy} onClick={save}>{busy === "save" ? "Saving…" : "Save reviewed to knowledge"}</button></div>
        {crosscheck && <div className="crosscheck-box"><h4>Datasheet cross-check</h4><MarkdownText text={crosscheck.comparison} /><SourcePanel sources={{ local: crosscheck.sources?.local || [], web: [] }} />
          {crosscheck.web_error && <p className="web-search-note">Web curation was unavailable: {crosscheck.web_error}</p>}
          {crosscheck.sources?.web?.length > 0 && <div className="curated-results"><h4>Curated web references</h4><p className="subtle">All returned sources are selected by default. Select any subset to add their readable page text to your local library. {selectedWebUrls.size} selected.</p>
            {crosscheck.sources.web.map((item) => <label className="web-result" key={item.url}>
              <input type="checkbox" checked={selectedWebUrls.has(item.url)} onChange={(event) => setSelectedWebUrls((current) => {
                const next = new Set(current);
                if (event.target.checked) next.add(item.url);
                else if (!event.target.checked) next.delete(item.url);
                return next;
              })} />
              <span><b>{item.title}</b><a href={item.url} target="_blank" rel="noreferrer">{item.url}</a><small>{item.snippet || "No search excerpt available."}</small></span>
            </label>)}
            <button className="button primary" disabled={!selectedWebUrls.size || !!busy} onClick={addWebSources}>{busy === "curate" ? "Adding references…" : "Add selected references to local library"}</button>
          </div>}
        </div>}
      </> : <div className="empty-state">Your component and connection extraction will appear here for review.</div>}
    </section>
  </div>;
}

function PrototypeDesigner({ useWeb }) {
  const [description, setDescription] = useState("");
  const [board, setBoard] = useState("ESP32");
  const [constraints, setConstraints] = useState("");
  const [design, setDesign] = useState(null);
  const [diagramSvg, setDiagramSvg] = useState("");
  const [graph, setGraph] = useState(null);
  const [verification, setVerification] = useState(null);
  const [sources, setSources] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const generate = async (event) => {
    event.preventDefault(); setBusy(true); setError("");
    try {
      const response = await postJson("/api/prototype", { description, board, constraints, use_web: useWeb });
      setDesign(response.design); setDiagramSvg(response.svg || ""); setGraph(response.graph); setVerification(response.verification); setSources(response.sources);
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  };

  const csv = (items) => {
    if (!items?.length) return "";
    const fields = [...new Set(items.flatMap((item) => Object.keys(item)))];
    const cell = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
    return [fields.map(cell).join(","), ...items.map((item) => fields.map((key) => cell(item[key])).join(","))].join("\n");
  };

  return <div className="prototype-layout">
    <form className="card prototype-form" onSubmit={generate}><div className="card-title"><span className="icon-box gold">⌘</span><div><h3>Embedded-system prototype designer</h3><p>Use your research context to create a sourced concept and labeled schematic image.</p></div></div>
      <label className="field-label" htmlFor="project-description">Research problem or prototype goal</label><textarea id="project-description" className="text-input large-input" rows="3" value={description} onChange={(event) => setDescription(event.target.value)} placeholder="Describe the problem, what to measure, and the desired outcome…" required />
      <label className="field-label" htmlFor="target-board">Target platform</label><select className="text-input" id="target-board" value={board} onChange={(event) => setBoard(event.target.value)}>{["ESP32", "Arduino", "Raspberry Pi", "FPGA", "Other / undecided"].map((item) => <option key={item}>{item}</option>)}</select>
      <label className="field-label" htmlFor="design-constraints">Constraints</label><textarea id="design-constraints" className="text-input" rows="2" value={constraints} onChange={(event) => setConstraints(event.target.value)} placeholder="Budget, power, available sensors, connectivity…" />
      <button className="button primary full" disabled={busy}>{busy ? "Preparing sourced draft…" : "Design prototype + generate labeled diagram"}</button>
      <ErrorBanner>{error}</ErrorBanner>
    </form>
    {design && <section className="card design-result">
      <div className="design-heading"><div><span className="eyebrow">LABELED SCHEMATIC IMAGE</span><h2>{design.title}</h2></div><div className="download-group"><button className="button secondary" onClick={() => downloadFile("akito_prototype.json", JSON.stringify(design, null, 2), "application/json")}>Draft JSON ↓</button><button className="button secondary" onClick={() => downloadFile("akito_connectivity_graph.json", JSON.stringify(graph, null, 2), "application/json")}>EDA graph JSON ↓</button><button className="button secondary" onClick={() => downloadFile("akito_connections.csv", csv(design.connections), "text/csv")}>Connections CSV ↓</button><button className="button secondary" onClick={() => downloadFile("akito_components.csv", csv(design.components), "text/csv")}>BOM CSV ↓</button><button className="button secondary" onClick={async () => { try { await downloadSvgAsPng("akito_labeled_schematic.png", diagramSvg); } catch (err) { setError(err.message); } }}>PNG ↓</button><button className="button secondary" onClick={() => downloadFile("akito_labeled_schematic.svg", diagramSvg, "image/svg+xml")}>SVG ↓</button></div></div>
      <div className="notice-callout">This labeled block schematic is generated as a vector image and can be exported as PNG or SVG. It is a conceptual diagram, not a verified electrical schematic or manufacturing-ready netlist. Verify all wiring and limits against primary datasheets and an EDA tool.</div>
      <div className="diagram-frame"><img src={`data:image/svg+xml;charset=utf-8,${encodeURIComponent(diagramSvg)}`} alt="Labeled embedded-system block schematic draft" /></div>
      <div className="design-summary"><div><h3>Research framing</h3><p><b>Problem</b><br/>{design.problem}</p><p><b>Research gap hypothesis</b><br/>{design.research_gap_hypothesis}</p><p><b>Potential novelty hypothesis</b><br/>{design.novelty_hypothesis}</p><small>Novelty needs a systematic literature review; this is a starting hypothesis.</small></div>
        <div><h3>Draft components</h3><div className="component-list">{design.components.map((item, index) => <article key={`${item.id}-${index}`}><span>{item.id}</span><div><b>{item.name}</b><small>{item.role}</small><small>Interface: {item.interface}</small><small>Pin/limit check: {item.pins_or_limits}</small></div></article>)}</div></div></div>
      {verification && <div className="verification-card"><h3>Structural review · {verification.status.replaceAll("_", " ")}</h3><p>{verification.message}</p>{verification.structural_errors.length > 0 && <ul className="verification-errors">{verification.structural_errors.map((item, index) => <li key={index}>{item}</li>)}</ul>}{verification.review_warnings.length > 0 && <ul>{verification.review_warnings.map((item, index) => <li key={index}>{item}</li>)}</ul>}<small>Electrical-rule checking is not run. Use the graph export as draft interchange data, then validate it in an EDA tool.</small></div>}
      {design.connections?.length > 0 && <><h3>Draft connections</h3><div className="table-scroll"><table><thead><tr><th>From</th><th>To</th><th>Signal</th><th>Review notes</th></tr></thead><tbody>{design.connections.map((edge, index) => <tr key={index}><td>{edge.from}</td><td>{edge.to}</td><td>{edge.signal}</td><td>{edge.notes}</td></tr>)}</tbody></table></div></>}
      <div className="design-summary lists">{[["Firmware plan", design.firmware_plan], ["Validation plan", design.validation_plan], ["Risks and open questions", design.risks_and_open_questions]].map(([title, values]) => <div key={title}><h3>{title}</h3><ul>{(values || []).map((value, index) => <li key={index}>{value}</li>)}</ul></div>)}</div>
      <SourcePanel sources={sources} />
    </section>}
  </div>;
}

function localDateValue() {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
}

function emptyExperimentDraft() {
  return {
    project: "",
    experiment: "",
    trial: "1",
    logged_on: localDateValue(),
    independent_variable: "",
    setting: "",
    metric: "",
    value: "",
    unit: "",
    conditions: "",
    notes: "",
    source: "",
  };
}

function ExperimentPage() {
  const [records, setRecords] = useState([]);
  const [draft, setDraft] = useState(emptyExperimentDraft);
  const [projectFilter, setProjectFilter] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const load = useCallback(async () => {
    try { setRecords((await api("/api/experiments")).items || []); }
    catch (err) { setError(err.message); }
  }, []);
  useEffect(() => { load(); }, [load]);

  const update = (field, value) => setDraft((current) => ({ ...current, [field]: value }));
  const save = async (event) => {
    event.preventDefault(); setBusy(true); setError(""); setNotice("");
    try {
      const response = await postJson("/api/experiments", draft);
      setRecords((items) => [response.item, ...items]);
      setDraft((current) => ({ ...current, trial: /^\d+$/.test(current.trial) ? String(Number(current.trial) + 1) : current.trial, value: "", conditions: "", notes: "", source: "" }));
      setNotice("Measurement saved locally as a new record.");
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  };

  const exportRecords = async () => {
    setError(""); setNotice("");
    try {
      const csv = await api("/api/experiments/export.csv");
      downloadFile("akito-experiments.csv", csv, "text/csv;charset=utf-8");
      setNotice("Exported every saved experiment record as CSV.");
    } catch (err) { setError(err.message); }
  };

  const projects = [...new Set(records.map((item) => item.project).filter(Boolean))].sort((a, b) => a.localeCompare(b));
  const visibleRecords = records.filter((item) => !projectFilter || item.project === projectFilter).slice(0, 100);

  return <div className="page">
    <div className="page-heading"><div><span className="eyebrow">REPRODUCIBLE RESEARCH</span><h2>Experiment Log</h2><p>Capture actual settings and measurements as structured, locally stored trial records.</p></div><div className="stat-card"><strong>{records.length}</strong><span>measurements</span></div></div>
    <ErrorBanner>{error}</ErrorBanner>{notice && <div className="alert success">{notice}</div>}
    <div className="experiment-toolbar"><label className="filter-control"><span>Project filter</span><select className="text-input" value={projectFilter} onChange={(event) => setProjectFilter(event.target.value)}><option value="">All projects</option>{projects.map((project) => <option key={project}>{project}</option>)}</select></label><button className="button secondary" type="button" onClick={exportRecords}>↓ Export all records as CSV</button></div>
    <div className="two-column experiment-layout">
      <form className="card experiment-form" onSubmit={save}>
        <div className="card-title"><span className="icon-box gold">⌖</span><div><h3>Log a measurement</h3><p>Enter observed values as measured; AKITO will not fill in or modify your data.</p></div></div>
        <div className="experiment-form-grid">
          <label className="experiment-field"><span>Project</span><input className="text-input" maxLength="120" value={draft.project} onChange={(event) => update("project", event.target.value)} placeholder="e.g. Smart irrigation" required /></label>
          <label className="experiment-field"><span>Experiment</span><input className="text-input" maxLength="160" value={draft.experiment} onChange={(event) => update("experiment", event.target.value)} placeholder="e.g. Soil sensor calibration" required /></label>
          <label className="experiment-field"><span>Trial</span><input className="text-input" maxLength="60" value={draft.trial} onChange={(event) => update("trial", event.target.value)} placeholder="1" /></label>
          <label className="experiment-field"><span>Date measured</span><input className="text-input" type="date" value={draft.logged_on} onChange={(event) => update("logged_on", event.target.value)} required /></label>
          <label className="experiment-field"><span>Independent variable</span><input className="text-input" maxLength="160" value={draft.independent_variable} onChange={(event) => update("independent_variable", event.target.value)} placeholder="e.g. Soil moisture setting" /></label>
          <label className="experiment-field"><span>Setting / value</span><input className="text-input" maxLength="160" value={draft.setting} onChange={(event) => update("setting", event.target.value)} placeholder="e.g. 35%" /></label>
          <label className="experiment-field"><span>Measured metric</span><input className="text-input" maxLength="160" value={draft.metric} onChange={(event) => update("metric", event.target.value)} placeholder="e.g. Sensor voltage" required /></label>
          <div className="experiment-measurement"><label className="experiment-field"><span>Observed value</span><input className="text-input" maxLength="160" value={draft.value} onChange={(event) => update("value", event.target.value)} placeholder="Enter measured result" required /></label><label className="experiment-field"><span>Unit</span><input className="text-input" maxLength="40" value={draft.unit} onChange={(event) => update("unit", event.target.value)} placeholder="V, ms, %" /></label></div>
          <label className="experiment-field experiment-wide"><span>Conditions</span><input className="text-input" maxLength="500" value={draft.conditions} onChange={(event) => update("conditions", event.target.value)} placeholder="Ambient conditions, hardware revision, supply voltage…" /></label>
          <label className="experiment-field experiment-wide"><span>Notes</span><textarea className="text-input experiment-notes" maxLength="3000" value={draft.notes} onChange={(event) => update("notes", event.target.value)} placeholder="Observation, anomalies, or protocol deviations" /></label>
          <label className="experiment-field experiment-wide"><span>Source / protocol reference</span><input className="text-input" maxLength="500" value={draft.source} onChange={(event) => update("source", event.target.value)} placeholder="Notebook entry, filename/page, or URL" /></label>
        </div>
        <button className="button primary full" disabled={busy}>{busy ? "Saving measurement…" : "＋ Save trial record"}</button>
        <p className="subtle">Records are append-only in <code>documents/experiments.json</code>. Add a new row to document a correction; CSV export includes every record.</p>
      </form>
      <section className="card experiment-records"><div className="card-title"><span className="icon-box sage">▤</span><div><h3>Recorded trials</h3><p>{projectFilter ? `Showing ${projectFilter}` : "Newest measurements first"} · latest 100 shown</p></div></div>
        <div className="experiment-record-list">{visibleRecords.length ? visibleRecords.map((item) => <article className="experiment-record" key={item.id}>
          <div className="experiment-record-heading"><div><span className="eyebrow">{item.project} · {item.trial || "Trial"}</span><h4>{item.experiment}</h4></div><time>{item.logged_on}</time></div>
          <div className="experiment-result"><span>{item.metric}</span><strong>{item.value}{item.unit ? ` ${item.unit}` : ""}</strong></div>
          {(item.independent_variable || item.setting) && <p className="experiment-context">{item.independent_variable}{item.setting ? `: ${item.setting}` : ""}</p>}
          {item.conditions && <p className="experiment-context">Conditions: {item.conditions}</p>}
          {item.notes && <p className="experiment-record-notes">{item.notes}</p>}
          {item.source && <small className="experiment-source">Source: {item.source}</small>}
        </article>) : <div className="empty-state">No measurement records yet. Log your first trial to start a reproducible dataset.</div>}</div>
      </section>
    </div>
  </div>;
}

function LabPage({ useWeb, refreshHealth }) {
  const [tool, setTool] = useState("prototype");
  return <div className="page"><div className="page-heading"><div><span className="eyebrow">DESIGN WORKBENCH</span><h2>Schematic & Prototype Lab</h2><p>Read a diagram with uncertainty review, or turn a research goal into a draft system.</p></div></div>
    <div className="subnav"><button className={tool === "read" ? "selected" : ""} onClick={() => setTool("read")}>⌗ Read a schematic</button><button className={tool === "prototype" ? "selected" : ""} onClick={() => setTool("prototype")}>⌘ Prototype designer</button></div>
    {tool === "read" ? <SchematicReader refreshHealth={refreshHealth} useWeb={useWeb} /> : <PrototypeDesigner useWeb={useWeb} />}
  </div>;
}

function MemoryPage() {
  const [items, setItems] = useState([]);
  const [editing, setEditing] = useState(null);
  const [title, setTitle] = useState("");
  const [category, setCategory] = useState("Project context");
  const [content, setContent] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const load = useCallback(async () => {
    try { setItems((await api("/api/memory")).items || []); }
    catch (err) { setError(err.message); }
  }, []);
  useEffect(() => { load(); }, [load]);
  const reset = () => { setEditing(null); setTitle(""); setCategory("Project context"); setContent(""); };
  const save = async (event) => {
    event.preventDefault(); setBusy(true); setError(""); setNotice("");
    try {
      const payload = { title, category, content };
      if (editing) await api(`/api/memory/${editing}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      else await postJson("/api/memory", payload);
      setNotice(editing ? "Memory updated." : "Memory saved locally. Matching notes may be used as context in later chats."); reset(); await load();
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  };
  const edit = (item) => { setEditing(item.id); setTitle(item.title); setCategory(item.category || "Project context"); setContent(item.content); setError(""); setNotice(""); };
  const remove = async (item) => {
    if (!window.confirm(`Delete memory “${item.title}”? It will be removed from future chats.`)) return;
    try { await api(`/api/memory/${item.id}`, { method: "DELETE" }); if (editing === item.id) reset(); setNotice("Memory deleted."); await load(); }
    catch (err) { setError(err.message); }
  };
  return <div className="page"><div className="page-heading"><div><span className="eyebrow">USER-CONTROLLED CONTEXT</span><h2>Memory</h2><p>AKITO only remembers notes you add here. Review, edit, or delete them whenever you want.</p></div><div className="stat-card"><strong>{items.length}</strong><span>saved notes</span></div></div>
    <ErrorBanner>{error}</ErrorBanner>{notice && <div className="alert success">{notice}</div>}
    <div className="two-column memory-grid"><form className="card" onSubmit={save}><div className="card-title"><span className="icon-box blush">✦</span><div><h3>{editing ? "Edit memory" : "Add memory"}</h3><p>Save a project fact, preference, or current goal. AKITO does not infer or save memories by itself.</p></div></div>
      <label className="field-label">Title</label><input className="text-input" maxLength="120" value={title} onChange={(event) => setTitle(event.target.value)} required />
      <label className="field-label">Type</label><select className="text-input" value={category} onChange={(event) => setCategory(event.target.value)}>{["Project context", "Preference", "Current goal", "Constraint"].map((value) => <option key={value}>{value}</option>)}</select>
      <label className="field-label">What should AKITO remember?</label><textarea className="text-input memory-content" maxLength="2500" value={content} onChange={(event) => setContent(event.target.value)} required />
      <div className="button-row"><button className="button primary" disabled={busy}>{busy ? "Saving…" : editing ? "Update memory" : "Save memory"}</button>{editing && <button className="button secondary" type="button" onClick={reset}>Cancel edit</button>}</div>
      <p className="subtle">Only a few notes that match your question are added to chat context. Memory stays in documents/memory.json.</p>
    </form><section className="card"><div className="card-title"><span className="icon-box sage">≡</span><div><h3>Saved memory</h3><p>Notes are stored locally and can be edited or removed.</p></div></div>
      <div className="memory-list">{items.length ? items.map((item) => <article className="memory-item" key={item.id}><div><span className="eyebrow">{item.category}</span><h4>{item.title}</h4><p>{item.content}</p><small>Updated {new Date(item.updated_at).toLocaleString()}</small></div><div className="memory-actions"><button className="button secondary" type="button" onClick={() => edit(item)}>Edit</button><button className="button secondary" type="button" onClick={() => remove(item)}>Delete</button></div></article>) : <div className="empty-state">No saved memory yet. Add a research goal or preference to get started.</div>}</div>
    </section></div>
  </div>;
}

function NotebookPage() {
  const [entries, setEntries] = useState([]);
  const [title, setTitle] = useState("");
  const [category, setCategory] = useState("Research idea");
  const [body, setBody] = useState("");
  const [sources, setSources] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try { setEntries((await api("/api/notebook")).entries || []); }
    catch (err) { setError(err.message); }
  }, []);
  useEffect(() => { load(); }, [load]);

  const save = async (event) => {
    event.preventDefault(); setBusy(true); setError(""); setNotice("");
    try {
      const response = await postJson("/api/notebook", { title, category, body, sources });
      setNotice(`Saved and indexed ${response.path}.`); setTitle(""); setBody(""); setSources(""); await load();
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  };

  return <div className="page"><div className="page-heading"><div><span className="eyebrow">PROJECT MEMORY</span><h2>Research Notebook</h2><p>Record ideas, assumptions, design choices, and real experiment results with their sources.</p></div></div>
    <ErrorBanner>{error}</ErrorBanner>{notice && <div className="alert success">{notice}</div>}
    <div className="two-column notebook-grid"><form className="card" onSubmit={save}><div className="card-title"><span className="icon-box blush">✎</span><div><h3>New entry</h3><p>Notebook entries become searchable local knowledge.</p></div></div>
      <label className="field-label">Entry title</label><input className="text-input" value={title} onChange={(event) => setTitle(event.target.value)} placeholder="e.g. Low-power ESP32 sensing approach" required />
      <label className="field-label">Entry type</label><select className="text-input" value={category} onChange={(event) => setCategory(event.target.value)}>{["Research idea", "Hypothesis", "Design decision", "Experiment plan", "Experiment result", "Meeting note"].map((item) => <option key={item}>{item}</option>)}</select>
      <label className="field-label">Note</label><textarea className="text-input note-body" value={body} onChange={(event) => setBody(event.target.value)} placeholder="Separate measured observations from assumptions and next steps…" required />
      <label className="field-label">Linked sources · one file/page or URL per line</label><textarea className="text-input sources-input" value={sources} onChange={(event) => setSources(event.target.value)} />
      <button className="button primary full" disabled={busy}>{busy ? "Saving…" : "Save to local notebook"}</button>
    </form>
    <section className="card"><div className="card-title"><span className="icon-box sage">▧</span><div><h3>Recent entries</h3><p>Saved notes are indexed alongside your research files.</p></div></div>
      <div className="notebook-list">{entries.length ? entries.map((entry) => <details key={entry.filename}><summary>{entry.filename}</summary><pre>{entry.content}</pre></details>) : <div className="empty-state">Your research notes will appear here.</div>}</div>
    </section></div>
  </div>;
}

export default function App() {
  const [page, setPage] = useState("assistant");
  const [useWeb, setUseWeb] = useState(false);
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState("");

  const refreshHealth = useCallback(async () => {
    try { setHealth(await api("/api/health")); setHealthError(""); }
    catch (err) { setHealthError(err.message); }
  }, []);
  useEffect(() => {
    refreshHealth();
    const timer = window.setInterval(refreshHealth, 20000);
    return () => window.clearInterval(timer);
  }, [refreshHealth]);

  return <div className="app-shell">
    <aside className="sidebar"><div className="brand"><div className="brand-orb">A</div><div><strong>AKITO</strong></div></div>
      <div className="sidebar-controls"><Toggle checked={useWeb} onChange={setUseWeb} label="Web Search" help="Search web sources alongside local files."/></div>
      <nav className="main-nav" aria-label="Main navigation">{NAV.map(([id, icon, label]) => <button className={page === id ? "nav-item active" : "nav-item"} key={id} onClick={() => setPage(id)}><span className="nav-icon">{icon}</span>{label}<span className="nav-arrow">›</span></button>)}</nav>
      <div className="sidebar-bottom"><div className={`service-status ${health?.ollama === "connected" ? "online" : "offline"}`}><i/>{health?.ollama === "connected" ? "Ollama connected" : "Ollama not detected"}</div><small>{health?.indexed_chunks ?? "—"} chunks in local library</small><small>Chat: {health?.chat_model || "loading…"}</small><small>Vision: {health?.vision_model || "loading…"}</small>{healthError && <small className="health-error">API: {healthError}</small>}</div>
      <div className="palette-strip" aria-label="Theme palette"><i/><i/><i/><i/><i/></div>
    </aside>
    <main className="main-shell"><header className="topbar"><div><span className="eyebrow">COMPUTER ENGINEERING · RESEARCH COPILOT</span></div><div className="top-status"><span className="status-dot"/>Local workspace</div></header>
      {page === "assistant" && <ChatPage useWeb={useWeb} />}
      {page === "library" && <KnowledgePage health={health} refreshHealth={refreshHealth} />}
      {page === "lab" && <LabPage useWeb={useWeb} refreshHealth={refreshHealth} />}
      {page === "notebook" && <NotebookPage />}
      {page === "experiments" && <ExperimentPage />}
      {page === "memory" && <MemoryPage />}
    </main>
  </div>;
}
