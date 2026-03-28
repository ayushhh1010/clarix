"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
    listRepos,
    uploadRepo,
    getRepoStatus,
    getRepoFiles,
    chatStream,
    runAgentStream,
    listConversations,
    getChatHistory,
    deleteRepo,
    type RepoResponse,
    type FileNode,
    type ConversationResponse,
} from "@/lib/api";
import ProtectedRoute from "@/components/ProtectedRoute";
import { useAuth } from "@/components/AuthProvider";

// ── Types ───────────────────────────────────────────────────

interface ChatMessage {
    id: string;
    role: "user" | "assistant" | "system";
    content: string;
    sources?: any[];
    agentSteps?: any[];
}

// ── Main Page Component ─────────────────────────────────────

function DashboardContent() {
    const { user, logout } = useAuth();
    // State
    const [repos, setRepos] = useState<RepoResponse[]>([]);
    const [activeRepo, setActiveRepo] = useState<RepoResponse | null>(null);
    const [files, setFiles] = useState<FileNode[]>([]);
    const [conversations, setConversations] = useState<ConversationResponse[]>([]);
    const [activeConvId, setActiveConvId] = useState<string | undefined>();
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [showUpload, setShowUpload] = useState(false);
    const [uploadUrl, setUploadUrl] = useState("");
    const [isUploading, setIsUploading] = useState(false);
    const [useAgent, setUseAgent] = useState(false);

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLTextAreaElement>(null);

    // Scroll to bottom on new messages
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    // Load repos on mount
    useEffect(() => {
        loadRepos();
    }, []);

    // Load files and conversations when active repo changes
    useEffect(() => {
        if (activeRepo?.status === "ready") {
            loadFiles(activeRepo.id);
            loadConversations(activeRepo.id);
        }
    }, [activeRepo?.id, activeRepo?.status]);

    // ── Data Loading ────────────────────────────────────────

    const loadRepos = async () => {
        try {
            const data = await listRepos();
            setRepos(data);
        } catch (err) {
            console.error("Failed to load repos:", err);
        }
    };

    const loadFiles = async (repoId: string) => {
        try {
            const data = await getRepoFiles(repoId);
            setFiles(data);
        } catch (err) {
            console.error("Failed to load files:", err);
        }
    };

    const loadConversations = async (repoId: string) => {
        try {
            const data = await listConversations(repoId);
            setConversations(data);
        } catch (err) {
            console.error("Failed to load conversations:", err);
        }
    };

    const loadConversationHistory = async (convId: string) => {
        try {
            const data = await getChatHistory(convId);
            setMessages(
                data.map((m) => ({
                    id: m.id,
                    role: m.role as "user" | "assistant",
                    content: m.content,
                }))
            );
            setActiveConvId(convId);
        } catch (err) {
            console.error("Failed to load history:", err);
        }
    };

    // ── Repo Upload ─────────────────────────────────────────

    const handleUpload = async () => {
        if (!uploadUrl.trim()) return;
        setIsUploading(true);
        try {
            const repo = await uploadRepo(uploadUrl.trim());
            setRepos((prev) => [repo, ...prev]);
            setShowUpload(false);
            setUploadUrl("");
            setActiveRepo(repo);

            // Poll for ingestion status
            pollRepoStatus(repo.id);
        } catch (err) {
            console.error("Upload failed:", err);
            alert("Failed to upload repository. Check the URL and try again.");
        } finally {
            setIsUploading(false);
        }
    };

    const pollRepoStatus = useCallback(async (repoId: string) => {
        const poll = async () => {
            try {
                const status = await getRepoStatus(repoId);
                setRepos((prev) =>
                    prev.map((r) =>
                        r.id === repoId ? { ...r, ...status } : r
                    )
                );
                setActiveRepo((prev) =>
                    prev?.id === repoId ? { ...prev, ...status } : prev
                );

                if (status.status === "pending" || status.status === "ingesting") {
                    setTimeout(poll, 3000);
                } else if (status.status === "ready") {
                    loadFiles(repoId);
                }
            } catch (err) {
                console.error("Poll failed:", err);
            }
        };
        poll();
    }, []);

    // ── Chat ────────────────────────────────────────────────

    const handleSend = async () => {
        if (!input.trim() || !activeRepo || isLoading) return;

        const userMsg: ChatMessage = {
            id: Date.now().toString(),
            role: "user",
            content: input.trim(),
        };

        setMessages((prev) => [...prev, userMsg]);
        setInput("");
        setIsLoading(true);

        const assistantMsg: ChatMessage = {
            id: (Date.now() + 1).toString(),
            role: "assistant",
            content: "",
            sources: [],
            agentSteps: [],
        };
        setMessages((prev) => [...prev, assistantMsg]);

        const streamFn = useAgent ? runAgentStream : chatStream;

        streamFn(
            activeRepo.id,
            userMsg.content,
            activeConvId,
            (data: any) => {
                if (data.type === "metadata") {
                    setActiveConvId(data.conversation_id);
                } else if (data.type === "content") {
                    setMessages((prev) =>
                        prev.map((m) =>
                            m.id === assistantMsg.id
                                ? { ...m, content: m.content + data.content }
                                : m
                        )
                    );
                } else if (data.type === "agent_step") {
                    setMessages((prev) =>
                        prev.map((m) =>
                            m.id === assistantMsg.id
                                ? {
                                    ...m,
                                    agentSteps: [...(m.agentSteps || []), data],
                                    ...(data.type === "final_answer"
                                        ? { content: data.content }
                                        : {}),
                                }
                                : m
                        )
                    );
                } else if (data.type === "final_answer") {
                    setMessages((prev) =>
                        prev.map((m) =>
                            m.id === assistantMsg.id
                                ? { ...m, content: data.content }
                                : m
                        )
                    );
                }
            },
            () => {
                setIsLoading(false);
                if (activeRepo) loadConversations(activeRepo.id);
            },
            (err: Error) => {
                console.error("Stream error:", err);
                setMessages((prev) =>
                    prev.map((m) =>
                        m.id === assistantMsg.id
                            ? { ...m, content: `Error: ${err.message}` }
                            : m
                    )
                );
                setIsLoading(false);
            }
        );
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleDeleteRepo = async (repoId: string) => {
        if (!confirm("Are you sure you want to delete this repository?")) return;
        try {
            await deleteRepo(repoId);
            setRepos((prev) => prev.filter((r) => r.id !== repoId));
            if (activeRepo?.id === repoId) {
                setActiveRepo(null);
                setMessages([]);
                setFiles([]);
                setConversations([]);
            }
        } catch (err) {
            console.error("Delete failed:", err);
        }
    };

    const startNewConversation = () => {
        setActiveConvId(undefined);
        setMessages([]);
    };

    // ── Render ──────────────────────────────────────────────

    return (
        <div className="app-layout">
            {/* ── Sidebar ──────────────────────── */}
            <aside className="sidebar">
                <div className="sidebar-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: "var(--accent-light)" }}>
                            <polyline points="16 18 22 12 16 6" />
                            <polyline points="8 6 2 12 8 18" />
                        </svg>
                        <h1>Clarix</h1>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        {user && (
                            <span style={{ fontSize: 11, color: "var(--text-muted)", maxWidth: 80, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                                {user.name || user.email}
                            </span>
                        )}
                        <button
                            onClick={logout}
                            title="Logout"
                            style={{
                                background: "rgba(255,255,255,0.05)",
                                border: "1px solid rgba(255,255,255,0.1)",
                                borderRadius: 6,
                                padding: "4px 8px",
                                fontSize: 11,
                                color: "var(--text-secondary)",
                                cursor: "pointer",
                            }}
                        >
                            Logout
                        </button>
                    </div>
                </div>

                <div className="sidebar-section">
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                        <span className="sidebar-section-title" style={{ margin: 0 }}>Repositories</span>
                        <button className="btn btn-primary btn-sm" onClick={() => setShowUpload(true)}>
                            + Add
                        </button>
                    </div>
                    <ul className="sidebar-list">
                        {repos.map((repo) => (
                            <li
                                key={repo.id}
                                className={`sidebar-item ${activeRepo?.id === repo.id ? "active" : ""}`}
                                onClick={() => setActiveRepo(repo)}
                            >
                                <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis" }}>
                                    {repo.name}
                                </span>
                                <span className={`status-badge ${repo.status}`}>
                                    {repo.status}
                                </span>
                            </li>
                        ))}
                        {repos.length === 0 && (
                            <li className="sidebar-item" style={{ color: "var(--text-muted)", cursor: "default" }}>
                                No repositories yet
                            </li>
                        )}
                    </ul>
                </div>

                {activeRepo?.status === "ready" && (
                    <>
                        <div className="sidebar-section">
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                                <span className="sidebar-section-title" style={{ margin: 0 }}>Conversations</span>
                                <button className="btn btn-secondary btn-sm" onClick={startNewConversation}>
                                    New
                                </button>
                            </div>
                            <ul className="sidebar-list">
                                {conversations.map((conv) => (
                                    <li
                                        key={conv.id}
                                        className={`sidebar-item ${activeConvId === conv.id ? "active" : ""}`}
                                        onClick={() => loadConversationHistory(conv.id)}
                                    >
                                        {conv.title.slice(0, 40)}
                                    </li>
                                ))}
                            </ul>
                        </div>

                        <div className="sidebar-section sidebar-scroll">
                            <span className="sidebar-section-title">Files</span>
                            <div className="file-tree">
                                {files.map((f) => (
                                    <div key={f.path} className="file-node">
                                        <span className="file-node-icon">
                                            {f.type === "directory" ? "📁" : "📄"}
                                        </span>
                                        {f.name}
                                    </div>
                                ))}
                            </div>
                        </div>
                    </>
                )}
            </aside>

            {/* ── Main Content ────────────────── */}
            <main className="main-content">
                {activeRepo ? (
                    <>
                        <header className="main-header">
                            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                                <h2>{activeRepo.name}</h2>
                                <span className={`status-badge ${activeRepo.status}`}>
                                    {activeRepo.status}
                                </span>
                                {activeRepo.status === "ready" && (
                                    <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
                                        {activeRepo.file_count} files · {activeRepo.chunk_count} chunks
                                    </span>
                                )}
                            </div>
                            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                                <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13, cursor: "pointer", color: "var(--text-secondary)" }}>
                                    <input
                                        type="checkbox"
                                        checked={useAgent}
                                        onChange={(e) => setUseAgent(e.target.checked)}
                                        style={{ accentColor: "var(--accent)" }}
                                    />
                                    Agent Mode
                                </label>
                                <button
                                    className="btn btn-secondary btn-sm"
                                    onClick={() => handleDeleteRepo(activeRepo.id)}
                                    style={{ color: "var(--error)" }}
                                >
                                    Delete
                                </button>
                            </div>
                        </header>

                        {activeRepo.status === "ready" ? (
                            <div className="chat-container">
                                <div className="chat-messages">
                                    {messages.length === 0 && (
                                        <div className="empty-state">
                                            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ color: "var(--accent-light)" }}>
                                                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                                            </svg>
                                            <h3>Ask anything about this codebase</h3>
                                            <p>
                                                I can explain code, debug issues, trace flows,
                                                suggest improvements, and even modify code.
                                                {useAgent && " Agent Mode is ON — I'll plan, research, and use tools."}
                                            </p>
                                        </div>
                                    )}

                                    {messages.map((msg) => (
                                        <div key={msg.id} className={`message ${msg.role}`}>
                                            {/* Agent steps */}
                                            {msg.agentSteps && msg.agentSteps.length > 0 && (
                                                <div className="agent-steps">
                                                    {msg.agentSteps.map((step: any, idx: number) => (
                                                        <div key={idx} className="agent-step">
                                                            <span className={`agent-step-badge ${step.agent}`}>
                                                                {step.agent}
                                                            </span>
                                                            <span style={{ color: "var(--text-secondary)", fontSize: 12 }}>
                                                                {step.plan
                                                                    ? `Plan: ${step.plan.join(" → ")}`
                                                                    : step.chunks_found !== undefined
                                                                        ? `Found ${step.chunks_found} code chunks`
                                                                        : step.tools_called
                                                                            ? `Called: ${step.tools_called.join(", ")}`
                                                                            : "Processing..."}
                                                            </span>
                                                        </div>
                                                    ))}
                                                </div>
                                            )}
                                            {/* Message content */}
                                            <div
                                                dangerouslySetInnerHTML={{
                                                    __html: formatMarkdown(msg.content),
                                                }}
                                            />
                                            {/* Sources */}
                                            {msg.sources && msg.sources.length > 0 && (
                                                <div className="message-sources">
                                                    <div className="message-sources-title">Sources</div>
                                                    {msg.sources.map((s: any, i: number) => (
                                                        <span key={i} className="source-tag">
                                                            {s.file_path}:{s.start_line}
                                                        </span>
                                                    ))}
                                                </div>
                                            )}
                                            {/* Loading indicator */}
                                            {msg.role === "assistant" && msg.content === "" && isLoading && (
                                                <div className="typing-indicator">
                                                    <div className="typing-dot" />
                                                    <div className="typing-dot" />
                                                    <div className="typing-dot" />
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                    <div ref={messagesEndRef} />
                                </div>

                                <div className="chat-input-container">
                                    <div className="chat-input-wrapper">
                                        <textarea
                                            ref={inputRef}
                                            className="chat-input"
                                            value={input}
                                            onChange={(e) => setInput(e.target.value)}
                                            onKeyDown={handleKeyDown}
                                            placeholder={
                                                useAgent
                                                    ? "Describe a task for the agent... (e.g., 'Find and fix the login bug')"
                                                    : "Ask about the codebase... (e.g., 'Explain the authentication flow')"
                                            }
                                            rows={1}
                                            disabled={isLoading}
                                        />
                                        <button
                                            className="btn btn-primary btn-icon"
                                            onClick={handleSend}
                                            disabled={!input.trim() || isLoading}
                                            title="Send"
                                        >
                                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                                <line x1="22" y1="2" x2="11" y2="13" />
                                                <polygon points="22 2 15 22 11 13 2 9 22 2" />
                                            </svg>
                                        </button>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="empty-state">
                                <div className="spinner spinner-lg" />
                                <h3>
                                    {activeRepo.status === "pending"
                                        ? "Queued for processing..."
                                        : activeRepo.status === "ingesting"
                                            ? "Ingesting codebase..."
                                            : "Ingestion failed"}
                                </h3>
                                <p>
                                    {activeRepo.status === "failed"
                                        ? activeRepo.error_message || "An error occurred during ingestion."
                                        : activeRepo.status === "ingesting"
                                            ? "⚙️ Cloning → Parsing → Chunking → Embedding... This takes 1-3 minutes."
                                            : "Queued — starting shortly..."}
                                </p>
                            </div>
                        )}
                    </>
                ) : (
                    <div className="empty-state">
                        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" style={{ color: "var(--accent-light)", opacity: 0.5 }}>
                            <polyline points="16 18 22 12 16 6" />
                            <polyline points="8 6 2 12 8 18" />
                        </svg>
                        <h3>Clarix</h3>
                        <p>
                            Add a Git repository to get started. I&apos;ll analyze the codebase
                            and answer your questions with full semantic understanding.
                        </p>
                        <button className="btn btn-primary" onClick={() => setShowUpload(true)}>
                            + Add Repository
                        </button>
                    </div>
                )}
            </main>

            {/* ── Upload Modal ────────────────── */}
            {showUpload && (
                <div className="modal-overlay" onClick={() => !isUploading && setShowUpload(false)}>
                    <div className="modal" onClick={(e) => e.stopPropagation()}>
                        <h3>Add Repository</h3>
                        <p>Enter a Git repository URL to clone and analyze.</p>
                        <input
                            className="modal-input"
                            type="text"
                            placeholder="https://github.com/user/repo.git"
                            value={uploadUrl}
                            onChange={(e) => setUploadUrl(e.target.value)}
                            onKeyDown={(e) => e.key === "Enter" && handleUpload()}
                            autoFocus
                            disabled={isUploading}
                        />
                        <div className="modal-actions">
                            <button
                                className="btn btn-secondary"
                                onClick={() => setShowUpload(false)}
                                disabled={isUploading}
                            >
                                Cancel
                            </button>
                            <button
                                className="btn btn-primary"
                                onClick={handleUpload}
                                disabled={!uploadUrl.trim() || isUploading}
                            >
                                {isUploading ? (
                                    <>
                                        <div className="spinner" /> Uploading...
                                    </>
                                ) : (
                                    "Clone & Analyze"
                                )}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

// ── Markdown Formatter (lightweight) ────────────────────────

function formatMarkdown(text: string): string {
    if (!text) return "";

    let html = text
        // Code blocks
        .replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
            return `<pre><code class="language-${lang}">${escapeHtml(code.trim())}</code></pre>`;
        })
        // Inline code
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        // Bold
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        // Italic
        .replace(/\*(.+?)\*/g, "<em>$1</em>")
        // Headers
        .replace(/^### (.+)$/gm, "<h4>$1</h4>")
        .replace(/^## (.+)$/gm, "<h3>$1</h3>")
        .replace(/^# (.+)$/gm, "<h2>$1</h2>")
        // Lists
        .replace(/^- (.+)$/gm, "<li>$1</li>")
        .replace(/^(\d+)\. (.+)$/gm, "<li>$2</li>")
        // Line breaks
        .replace(/\n\n/g, "</p><p>")
        .replace(/\n/g, "<br/>");

    // Wrap list items
    html = html.replace(/(<li>.*?<\/li>)+/gs, "<ul>$&</ul>");

    return `<p>${html}</p>`;
}

function escapeHtml(text: string): string {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

// ── Wrapped Export ──────────────────────────────────────────

export default function Dashboard() {
    return (
        <ProtectedRoute>
            <DashboardContent />
        </ProtectedRoute>
    );
}
