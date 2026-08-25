import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowUp,
  Bot,
  BrainCircuit,
  CheckCircle2,
  FileText,
  Loader2,
  MessageSquareText,
  RefreshCw,
  Search,
  Upload,
} from "lucide-react";
import ReactMarkdown from "react-markdown";

import { askQuestion, listDocuments, uploadDocuments } from "./api";
import type { ChatResponse, ChatTurn, DocumentInfo } from "./types";

type ConversationItem =
  | { kind: "user"; content: string }
  | { kind: "assistant"; response: ChatResponse };

function App() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [conversation, setConversation] = useState<ConversationItem[]>([]);
  const [question, setQuestion] = useState("");
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [isLoadingDocuments, setIsLoadingDocuments] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const totalChunks = useMemo(
    () => documents.reduce((sum, document) => sum + document.chunks, 0),
    [documents],
  );

  useEffect(() => {
    void refreshDocuments();
  }, []);

  async function refreshDocuments() {
    setIsLoadingDocuments(true);
    setError(null);

    try {
      setDocuments(await listDocuments());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load indexed documents");
    } finally {
      setIsLoadingDocuments(false);
    }
  }

  async function handleUpload(event: FormEvent) {
    event.preventDefault();

    if (!selectedFiles?.length) {
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      await uploadDocuments(selectedFiles);
      setSelectedFiles(null);

      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }

      await refreshDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setIsUploading(false);
    }
  }

  async function handleAsk(event: FormEvent) {
    event.preventDefault();

    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || isThinking) {
      return;
    }

    const history = buildHistory(conversation);

    setQuestion("");
    setError(null);
    setIsThinking(true);
    setConversation((items) => [...items, { kind: "user", content: trimmedQuestion }]);

    try {
      const response = await askQuestion(trimmedQuestion, history);
      setConversation((items) => [...items, { kind: "assistant", response }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Question failed");
    } finally {
      setIsThinking(false);
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <section className="brand-block">
          <div className="brand-mark">
            <BrainCircuit size={24} aria-hidden="true" />
          </div>
          <div>
            <p className="eyebrow">Agentic RAG</p>
            <h1>AI Financial Agent</h1>
          </div>
        </section>

        <section className="stats-grid" aria-label="Index stats">
          <div>
            <span>{documents.length}</span>
            <p>Documents</p>
          </div>
          <div>
            <span>{totalChunks}</span>
            <p>Chunks</p>
          </div>
        </section>

        <form className="upload-panel" onSubmit={handleUpload}>
          <div className="section-title">
            <Upload size={18} aria-hidden="true" />
            <h2>Upload Knowledge</h2>
          </div>

          <label className="drop-zone" htmlFor="files">
            <FileText size={24} aria-hidden="true" />
            <span>{selectedFiles?.length ? fileLabel(selectedFiles) : "Choose PDF, TXT, or MD files"}</span>
            <small>Files are chunked, embedded, and indexed in ChromaDB.</small>
          </label>

          <input
            ref={fileInputRef}
            id="files"
            type="file"
            multiple
            accept=".pdf,.txt,.md"
            onChange={(event) => setSelectedFiles(event.currentTarget.files)}
          />

          <button className="primary-button" type="submit" disabled={!selectedFiles?.length || isUploading}>
            {isUploading ? <Loader2 className="spin" size={18} aria-hidden="true" /> : <Upload size={18} aria-hidden="true" />}
            {isUploading ? "Indexing" : "Upload"}
          </button>
        </form>

        <section className="documents-panel">
          <div className="section-title">
            <FileText size={18} aria-hidden="true" />
            <h2>Indexed Documents</h2>
            <button className="icon-button" type="button" onClick={refreshDocuments} aria-label="Refresh indexed documents">
              {isLoadingDocuments ? <Loader2 className="spin" size={16} /> : <RefreshCw size={16} />}
            </button>
          </div>

          {documents.length === 0 ? (
            <p className="empty-copy">No indexed documents yet.</p>
          ) : (
            <ul className="document-list">
              {documents.map((document) => (
                <li key={document.source_id}>
                  <FileText size={17} aria-hidden="true" />
                  <div>
                    <strong>{document.filename}</strong>
                    <span>{document.chunks} chunks indexed</span>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Agentic Financial Analysis</p>
            <h2>Document Search + Web Search + Calculator</h2>
          </div>
          <div className="status-pill">
            <CheckCircle2 size={16} aria-hidden="true" />
            Backend ready
          </div>
        </header>

        <section className="pipeline-strip" aria-label="RAG pipeline">
          {["Plan", "Documents", "Web", "Calculate", "Evaluate", "Answer"].map((item) => (
            <div key={item}>{item}</div>
          ))}
        </section>

        <section className="chat-surface">
          <div className="messages">
            {conversation.length === 0 ? (
              <div className="empty-state">
                <MessageSquareText size={42} aria-hidden="true" />
                <h2>Ask the financial agent</h2>
                <p>Upload documents, then ask for analysis. The agent can search documents, use web search, calculate, and return a trace.</p>
              </div>
            ) : (
              conversation.map((item, index) =>
                item.kind === "user" ? (
                  <article className="message user-message" key={index}>
                    {item.content}
                  </article>
                ) : (
                  <AssistantMessage response={item.response} key={index} />
                ),
              )
            )}

            {isThinking && (
              <article className="message assistant-message loading-message">
                <Loader2 className="spin" size={18} aria-hidden="true" />
                Running agent tools and generating answer
              </article>
            )}
          </div>

          {error && <div className="error-panel">{error}</div>}

          <form className="question-form" onSubmit={handleAsk}>
            <Search size={20} aria-hidden="true" />
            <input
              value={question}
              onChange={(event) => setQuestion(event.currentTarget.value)}
              placeholder="Ask a financial analysis question"
              disabled={isThinking}
            />
            <button type="submit" aria-label="Ask question" disabled={isThinking || !question.trim()}>
              {isThinking ? <Loader2 className="spin" size={18} /> : <ArrowUp size={18} />}
            </button>
          </form>
        </section>
      </section>
    </main>
  );
}

function AssistantMessage({ response }: { response: ChatResponse }) {
  return (
    <article className="message assistant-message">
      <div className="assistant-heading">
        <Bot size={18} aria-hidden="true" />
        <span>Agent answer</span>
        <strong>{response.trace?.length ?? 0} steps</strong>
      </div>

      <div className="markdown-body">
        <ReactMarkdown>{response.answer}</ReactMarkdown>
      </div>

      {response.trace && response.trace.length > 0 && (
        <details className="source-details">
          <summary>Agent trace</summary>

          <div className="sources-grid">
            {response.trace.map((step, index) => (
              <section className="source-card" key={`${step.step}-${step.tool}-${index}`}>
                <div>
                  <strong>
                    [{step.step}] {step.tool}
                  </strong>
                  <span>tool call</span>
                </div>
                <pre>{JSON.stringify({ input: step.input, output: step.output }, null, 2)}</pre>
              </section>
            ))}
          </div>
        </details>
      )}
    </article>
  );
}

function buildHistory(items: ConversationItem[]): ChatTurn[] {
  return items.flatMap<ChatTurn>((item) => {
    if (item.kind === "user") {
      return [{ role: "user", content: item.content }];
    }

    return [{ role: "assistant", content: item.response.answer }];
  });
}

function fileLabel(files: FileList) {
  if (files.length === 1) {
    return files[0].name;
  }

  return `${files.length} files selected`;
}

export default App;
