import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  BookOpen,
  CheckCircle2,
  ChevronDown,
  Loader2,
  PanelRightOpen,
  RefreshCw,
  Search,
  Send,
  Upload,
  X,
} from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const API_KEY = import.meta.env.VITE_APP_API_KEY || "";

function sourceLabel(source) {
  return source.source_file;
}

function clampScore(score) {
  if (!Number.isFinite(score)) return 0;
  if (score <= 1) return Math.round(score * 100);
  return Math.round(score);
}

function groupSourcesByDocument(sources = []) {
  const groups = new Map();

  sources.forEach((source) => {
    const existing = groups.get(source.source_file);
    if (existing) {
      existing.chunks.push(source);
      existing.score = Math.max(existing.score, source.score || 0);
      existing.end_char = Math.max(existing.end_char || 0, source.end_char || 0);
      existing.start_char = Math.min(existing.start_char || source.start_char || 0, source.start_char || 0);
      return;
    }

    groups.set(source.source_file, {
      ...source,
      id: `doc::${source.source_file}`,
      citation_number: groups.size + 1,
      chunks: [source],
      score: source.score || 0,
    });
  });

  return Array.from(groups.values());
}

function App() {
  const [status, setStatus] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState("hybrid");
  const [sourceFile, setSourceFile] = useState("");
  const [messages, setMessages] = useState([]);
  const [activeSource, setActiveSource] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [authToken, setAuthToken] = useState(() => localStorage.getItem("auth_token") || "");
  const [authUser, setAuthUser] = useState(null);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  async function api(path, options, overrideToken = null) {
    const headers = new Headers(options?.headers || {});
    const token = overrideToken ?? authToken;
    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    } else if (API_KEY) {
      headers.set("X-API-Key", API_KEY);
    }

    const response = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers,
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || body.error || response.statusText);
    }
    return response.json();
  }

  async function refresh(nextToken = authToken) {
    try {
      const nextStatus = nextToken
        ? await api("/status", undefined, nextToken)
        : await api("/status");
      let docs = { documents: [] };
      if (nextToken) {
        docs = await api("/documents", undefined, nextToken);
      }
      setStatus(nextStatus);
      setDocuments(docs.documents || []);
      setError("");
    } catch (err) {
      setStatus(null);
      setDocuments([]);
      setError(`API offline: ${err.message}`);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    localStorage.setItem("auth_token", authToken);
  }, [authToken]);

  useEffect(() => {
    if (!authToken) {
      setAuthUser(null);
      return;
    }

    api("/auth/me")
      .then(setAuthUser)
      .catch(() => {
        setAuthToken("");
        setAuthUser(null);
      });
  }, [authToken]);

  async function login(event) {
    event.preventDefault();
    try {
      const result = await api("/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      const token = result.access_token;
      setAuthToken(token);
      setAuthUser(result.user);
      setPassword("");
      setError("");
      await refresh(token);
    } catch (err) {
      setError(err.message);
    }
  }

  async function register(event) {
    event.preventDefault();
    try {
      const result = await api("/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      const token = result.access_token;
      setAuthToken(token);
      setAuthUser(result.user);
      setPassword("");
      setError("");
      await refresh(token);
    } catch (err) {
      setError(err.message);
    }
  }

  async function logout() {
    try {
      await api("/auth/logout", { method: "POST" });
    } catch (_) {
      // Ignore logout errors and clear local state.
    } finally {
      setAuthToken("");
      setAuthUser(null);
      setDocuments([]);
      setSourceFile("");
      setMessages([]);
      setActiveSource(null);
    }
  }

  async function askQuestion(event) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || loading) return;

    setLoading(true);
    setError("");
    setQuestion("");
    const userMessage = { role: "user", content: trimmed };
    setMessages((current) => [...current, userMessage]);

    try {
      const result = await api("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: trimmed,
          mode,
          source_file: sourceFile || null,
        }),
      });
      const assistantMessage = {
        role: "assistant",
        answer: result.answer,
        sources: result.sources || [],
        citations: result.citations || [],
        confidence: result.confidence,
        confidenceLabel: result.confidence_label,
        mode: result.mode,
        chunks: result.chunks_retrieved,
      };
      setMessages((current) => [...current, assistantMessage]);
      const groupedSources = assistantMessage.citations.length
        ? assistantMessage.citations
        : groupSourcesByDocument(assistantMessage.sources);
      if (groupedSources.length) {
        setActiveSource(groupedSources[0]);
      }
    } catch (err) {
      setError(err.message);
      setMessages((current) => [
        ...current,
        { role: "assistant", answer: "I could not complete that request.", sources: [] },
      ]);
    } finally {
      setLoading(false);
    }
  }

  async function uploadFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      await api("/upload", { method: "POST", body: form });
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  }

  const latestSources = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i].citations?.length) return messages[i].citations;
      if (messages[i].sources?.length) return groupSourcesByDocument(messages[i].sources);
    }
    return [];
  }, [messages]);

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <BookOpen size={22} />
          </div>
          <div>
            <h1>Document Intelligence</h1>
            <p>Advanced RAG workspace</p>
          </div>
        </div>

        <section className="panel">
          <label className="field-label">Account</label>
          {authUser ? (
            <div className="status-ready" style={{ justifyContent: "space-between" }}>
              <span>{authUser.username} ({authUser.role})</span>
              <button className="icon-button" onClick={logout} title="Logout">
                <X size={14} />
              </button>
            </div>
          ) : (
            <form onSubmit={login}>
              <input
                className="auth-input"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Username"
                autoComplete="username"
              />
              <input
                className="auth-input"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Password"
                autoComplete="current-password"
              />
              <div className="auth-actions">
                <button type="submit" className="auth-button">Login</button>
                <button type="button" className="auth-button secondary" onClick={register}>
                  Register
                </button>
              </div>
            </form>
          )}
        </section>

        <section className="panel compact">
          <div className="panel-title">
            <span>API status</span>
            <button className="icon-button" onClick={refresh} title="Refresh status">
              <RefreshCw size={16} />
            </button>
          </div>
          {status ? (
            <div className="status-ready">
              <CheckCircle2 size={16} />
              <span>{status.num_documents} docs / {status.total_chunks} chunks</span>
            </div>
          ) : (
            <div className="status-error">Start FastAPI on port 8000</div>
          )}
        </section>

        <section className="panel">
          <label className="field-label" htmlFor="mode">Search mode</label>
          <div className="select-wrap">
            <select id="mode" value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="hybrid">Hybrid</option>
              <option value="dense">Dense</option>
              <option value="multi_doc">Multi document</option>
            </select>
            <ChevronDown size={16} />
          </div>

              <div className="field-row">
                <label className="field-label" htmlFor="source">Scope</label>
                <label className="mini-upload" title="Upload document">
                  {uploading ? <Loader2 className="spin" size={15} /> : <Upload size={15} />}
                  Upload
              <input type="file" accept=".pdf,.txt,.docx,.text" onChange={uploadFile} disabled={!authToken} />
                </label>
              </div>
              <div className="select-wrap">
            <select id="source" value={sourceFile} onChange={(e) => setSourceFile(e.target.value)} disabled={!authToken}>
              <option value="">All documents</option>
              {documents.map((doc) => (
                <option key={doc} value={doc}>{doc}</option>
              ))}
            </select>
            <ChevronDown size={16} />
          </div>
        </section>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Grounded answers</p>
            <h2>Ask your documents and inspect every citation.</h2>
          </div>
          <button
            className="source-toggle"
            onClick={() => setActiveSource(latestSources[0] || null)}
            disabled={!latestSources.length}
          >
            <PanelRightOpen size={17} />
            Sources
          </button>
        </header>

        {error && <div className="error-banner">{error}</div>}

        <div className="chat-surface">
          {messages.length === 0 ? (
            <div className="empty-state">
              <Search size={34} />
              <h3>Search across your knowledge base</h3>
              <p>Ask a question, then click a citation to open the exact retrieved text.</p>
            </div>
          ) : messages.map((message, index) => (
            <Message
              key={index}
              message={message}
              onCitationClick={setActiveSource}
            />
          ))}
          {loading && (
            <div className="message assistant">
              <Loader2 className="spin" size={18} />
              <span>Retrieving context and generating an answer...</span>
            </div>
          )}
        </div>

        <form className="composer" onSubmit={askQuestion}>
          <input
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask a question about your documents..."
            disabled={!status || loading || !authToken}
          />
          <button type="submit" disabled={!question.trim() || !status || loading || !authToken} title="Send question">
            {loading ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
          </button>
        </form>
      </section>

      <CitationPanel source={activeSource} onClose={() => setActiveSource(null)} />
    </main>
  );
}

function Message({ message, onCitationClick }) {
  if (message.role === "user") {
    return <div className="message user">{message.content}</div>;
  }

  const groupedSources = message.citations?.length
    ? message.citations
    : groupSourcesByDocument(message.sources);

  return (
    <article className="message assistant">
      <p>{message.answer}</p>
      {groupedSources.length > 0 && (
        <div className="citation-row">
          {groupedSources.map((source) => (
            <button
              key={source.id}
              className="citation-chip"
              onClick={() => onCitationClick(source)}
              title={`Open ${sourceLabel(source)}`}
            >
              [{source.citation_number}] {source.source_file}
              <span>{source.chunks.length} chunk{source.chunks.length === 1 ? "" : "s"}</span>
            </button>
          ))}
        </div>
      )}
      <div className="message-meta">
        {message.mode || "hybrid"} / {message.chunks || 0} chunks
        {typeof message.confidence === "number" && (
          <> / confidence {message.confidence.toFixed(3)} ({message.confidenceLabel || "unknown"})</>
        )}
      </div>
    </article>
  );
}

function CitationPanel({ source, onClose }) {
  const chunks = source?.chunks || (source ? [source] : []);

  return (
    <aside className={`citation-panel ${source ? "open" : ""}`}>
      <div className="citation-header">
        <div>
          <p className="eyebrow">Citation detail</p>
          <h2>{source ? `[${source.citation_number}] ${source.source_file}` : "No citation selected"}</h2>
        </div>
        <button className="icon-button" onClick={onClose} title="Close citations">
          <X size={17} />
        </button>
      </div>

      {source ? (
        <>
          <div className="source-stats">
            <span>{chunks.length} retrieved chunk{chunks.length === 1 ? "" : "s"}</span>
            <span>{source.start_char}-{source.end_char}</span>
            <span>{clampScore(source.score)}% match</span>
          </div>
          <div className="source-stack">
            {chunks.map((chunk) => (
              <section className="context-block" key={`${chunk.source_file}-${chunk.chunk_index}-${chunk.start_char}`}>
                <div className="context-meta">
                  <span>Chunk {chunk.chunk_index}</span>
                  <span>{chunk.start_char}-{chunk.end_char}</span>
                  <span>{clampScore(chunk.score)}% match</span>
                </div>
                <div className="source-text">{chunk.text}</div>
              </section>
            ))}
          </div>
        </>
      ) : (
        <p className="muted">Click any citation chip below an answer to inspect the retrieved chunk text.</p>
      )}
    </aside>
  );
}

createRoot(document.getElementById("root")).render(<App />);
