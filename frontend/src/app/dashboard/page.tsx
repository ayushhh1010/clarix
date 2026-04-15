"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import DOMPurify from "dompurify";
import { toast, Toaster } from "sonner";
import { Settings, Trash2, Edit2, ChevronRight, ChevronDown, File, Folder, FolderTree, MoreVertical, Check, X, MessageSquare, LogOut, User } from "lucide-react";
import {
    listRepos,
    uploadRepo,
    getRepoStatus,
    getRepoFiles,
    getFileContent,
    chatStream,
    runAgentStream,
    listConversations,
    getChatHistory,
    deleteRepo,
    deleteConversation,
    renameConversation,
    type RepoResponse,
    type FileNode,
    type ConversationResponse,
} from "@/lib/api";
import ProtectedRoute from "@/components/ProtectedRoute";
import { useAuth } from "@/components/AuthProvider";
import SettingsModal from "@/components/SettingsModal";
import FileViewer from "@/components/FileViewer";
import { ErrorBoundary } from "@/components/ErrorBoundary";

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
    const [expandedDirs, setExpandedDirs] = useState<Set<string>>(new Set());
    const [fileContents, setFileContents] = useState<Map<string, FileNode[]>>(new Map());
    // Per-repo conversations map & which repos are expanded in the sidebar
    const [repoConversations, setRepoConversations] = useState<Map<string, ConversationResponse[]>>(new Map());
    const [expandedRepos, setExpandedRepos] = useState<Set<string>>(new Set());
    const [activeConvId, setActiveConvId] = useState<string | undefined>();
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [showUpload, setShowUpload] = useState(false);
    const [uploadUrl, setUploadUrl] = useState("");
    const [isUploading, setIsUploading] = useState(false);
    const [useAgent, setUseAgent] = useState(false);
    // New state for file viewer
    const [viewingFile, setViewingFile] = useState<{ path: string; content: string; language: string } | null>(null);
    // New state for conversation management
    const [editingConvId, setEditingConvId] = useState<string | null>(null);
    const [editingConvTitle, setEditingConvTitle] = useState("");
    // Settings modal
    const [showSettings, setShowSettings] = useState(false);
    const [showSettingsMenu, setShowSettingsMenu] = useState(false);
    const [activeTray, setActiveTray] = useState<"repos" | "files" | null>("repos");

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLTextAreaElement>(null);
    const settingsMenuRef = useRef<HTMLDivElement>(null);

    // Close settings menu on click outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (settingsMenuRef.current && !settingsMenuRef.current.contains(event.target as Node)) {
                setShowSettingsMenu(false);
            }
        };

        if (showSettingsMenu) {
            document.addEventListener("mousedown", handleClickOutside);
        }
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, [showSettingsMenu]);

    // Scroll to bottom on new messages
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    // Load repos when user changes (login/logout/switch)
    useEffect(() => {
        // Reset all state on user change for clean isolation
        setActiveRepo(null);
        setFiles([]);
        setRepoConversations(new Map());
        setMessages([]);
        setActiveConvId(undefined);
        if (user) {
            loadRepos();
        } else {
            setRepos([]);
        }
    }, [user?.id]);

    // Persist active repo ID to localStorage
    useEffect(() => {
        if (activeRepo) {
            localStorage.setItem("clarix_active_repo_id", activeRepo.id);
        }
    }, [activeRepo?.id]);

    // Load files and conversations when active repo changes
    useEffect(() => {
        if (activeRepo?.status === "ready") {
            loadFiles(activeRepo.id);
            loadConversationsForRepo(activeRepo.id);
        }
    }, [activeRepo?.id, activeRepo?.status]);

    // ── Data Loading ────────────────────────────────────────

    const loadRepos = async () => {
        try {
            const data = await listRepos();
            setRepos(data.items);
            // Restore previously selected repo from localStorage
            const savedRepoId = localStorage.getItem("clarix_active_repo_id");
            if (savedRepoId && data.items.length > 0) {
                const savedRepo = data.items.find((r: RepoResponse) => r.id === savedRepoId);
                if (savedRepo) {
                    setActiveRepo(savedRepo);
                    setExpandedRepos(prev => new Set(prev).add(savedRepo.id));
                }
            }
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
            setFiles([]);
        }
    };

    const loadConversationsForRepo = async (repoId: string) => {
        try {
            const data = await listConversations(repoId);
            setRepoConversations(prev => new Map(prev).set(repoId, data.items));
        } catch (err) {
            console.error("Failed to load conversations:", err);
        }
    };

    const loadConversationHistory = async (convId: string) => {
        try {
            const data = await getChatHistory(convId);
            setMessages(
                data.items.map((m) => ({
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
            toast.success("Repository added! Starting analysis...");

            // Poll for ingestion status
            pollRepoStatus(repo.id);
        } catch (err) {
            console.error("Upload failed:", err);
            toast.error("Failed to upload repository. Check the URL and try again.");
        } finally {
            setIsUploading(false);
        }
    };

    const pollRepoStatus = useCallback(async (repoId: string) => {
        const MAX_POLL_DURATION_MS = 60 * 60 * 1000; // 60 minutes — large repos can take a while
        const startTime = Date.now();

        const poll = async () => {
            try {
                // Timeout guard — stop polling after MAX_POLL_DURATION_MS
                if (Date.now() - startTime > MAX_POLL_DURATION_MS) {
                    toast.error("Ingestion timed out. Please delete and re-add the repository.");
                    setRepos((prev) =>
                        prev.map((r) =>
                            r.id === repoId ? { ...r, status: "failed", error_message: "Ingestion timed out" } : r
                        )
                    );
                    setActiveRepo((prev) =>
                        prev?.id === repoId ? { ...prev, status: "failed", error_message: "Ingestion timed out" } : prev
                    );
                    return;
                }

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
                    setTimeout(poll, 2000);
                } else if (status.status === "ready") {
                    toast.success("Repository indexed successfully!");
                    loadFiles(repoId);
                } else if (status.status === "failed") {
                    toast.error(status.error_message || "Ingestion failed. Please try again.");
                }
                // For any other status (or failed/ready), polling stops naturally
            } catch (err) {
                console.error("Poll failed:", err);
                // Don't stop polling on transient network errors — retry
                setTimeout(poll, 3000);
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
                if (activeRepo) loadConversationsForRepo(activeRepo.id);
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
                localStorage.removeItem("clarix_active_repo_id");
            }
            setRepoConversations(prev => { const next = new Map(prev); next.delete(repoId); return next; });
            toast.success("Repository deleted");
        } catch (err) {
            console.error("Delete failed:", err);
            toast.error("Failed to delete repository");
        }
    };

    const startNewConversation = () => {
        setActiveConvId(undefined);
        setMessages([]);
    };

    // Toggle repo expansion in sidebar (loads conversations on first expand)
    const toggleRepoExpanded = async (repo: RepoResponse) => {
        const repoId = repo.id;
        if (expandedRepos.has(repoId)) {
            setExpandedRepos(prev => { const next = new Set(prev); next.delete(repoId); return next; });
        } else {
            setExpandedRepos(prev => new Set(prev).add(repoId));
            // Load conversations for this repo if not already loaded
            if (!repoConversations.has(repoId) && repo.status === "ready") {
                loadConversationsForRepo(repoId);
            }
        }
    };

    // ── File Explorer Functions ─────────────────────────────

    const toggleDirectory = async (path: string) => {
        if (!activeRepo) return;
        
        if (expandedDirs.has(path)) {
            setExpandedDirs(prev => {
                const next = new Set(prev);
                next.delete(path);
                return next;
            });
        } else {
            // Load directory contents if not cached
            if (!fileContents.has(path)) {
                try {
                    const contents = await getRepoFiles(activeRepo.id, path);
                    setFileContents(prev => new Map(prev).set(path, contents));
                } catch (err) {
                    console.error("Failed to load directory:", err);
                    toast.error("Failed to load directory contents");
                    return;
                }
            }
            setExpandedDirs(prev => new Set(prev).add(path));
        }
    };

    const openFile = async (path: string) => {
        if (!activeRepo) return;
        try {
            const { content, language } = await getFileContent(activeRepo.id, path);
            setViewingFile({ path, content, language });
        } catch (err) {
            console.error("Failed to load file:", err);
            toast.error("Failed to load file contents");
        }
    };

    // ── Conversation Management Functions ───────────────────

    const handleDeleteConversation = async (convId: string, repoId: string) => {
        if (!confirm("Delete this conversation?")) return;
        try {
            await deleteConversation(convId);
            setRepoConversations(prev => {
                const next = new Map(prev);
                const existing = next.get(repoId) || [];
                next.set(repoId, existing.filter(c => c.id !== convId));
                return next;
            });
            if (activeConvId === convId) {
                setActiveConvId(undefined);
                setMessages([]);
            }
            toast.success("Conversation deleted");
        } catch (err) {
            console.error("Delete conversation failed:", err);
            toast.error("Failed to delete conversation");
        }
    };

    const handleRenameConversation = async (convId: string, repoId: string) => {
        if (!editingConvTitle.trim()) return;
        try {
            await renameConversation(convId, editingConvTitle.trim());
            setRepoConversations(prev => {
                const next = new Map(prev);
                const existing = next.get(repoId) || [];
                next.set(repoId, existing.map(c => c.id === convId ? { ...c, title: editingConvTitle.trim() } : c));
                return next;
            });
            setEditingConvId(null);
            setEditingConvTitle("");
            toast.success("Conversation renamed");
        } catch (err) {
            console.error("Rename failed:", err);
            toast.error("Failed to rename conversation");
        }
    };

    const startEditingConversation = (conv: ConversationResponse) => {
        setEditingConvId(conv.id);
        setEditingConvTitle(conv.title);
    };

    // ── Render ──────────────────────────────────────────────

    return (
        <div className="app-layout">
            <Toaster position="top-right" richColors />
            
            {/* ── Activity Bar ─────────────────── */}
            <nav className="activity-bar">
                <div className="activity-bar-actions">
                    <button 
                        className={`activity-icon ${activeTray === "repos" ? "active" : ""}`}
                        onClick={() => setActiveTray(activeTray === "repos" ? null : "repos")}
                        data-tooltip="Repositories"
                    >
                        <MessageSquare size={22} strokeWidth={1.5} />
                    </button>
                    <button 
                        className={`activity-icon ${activeTray === "files" ? "active" : ""}`}
                        onClick={() => setActiveTray(activeTray === "files" ? null : "files")}
                        data-tooltip="File Explorer"
                    >
                        <FolderTree size={22} strokeWidth={1.5} />
                    </button>
                </div>

                <div className="activity-bar-actions">
                    <div className="activity-icon-wrapper" ref={settingsMenuRef}>
                        <button
                            className={`activity-icon ${showSettingsMenu ? "active" : ""}`}
                            onClick={() => setShowSettingsMenu(!showSettingsMenu)}
                            data-tooltip={showSettingsMenu ? undefined : "Settings"}
                        >
                            <Settings size={22} strokeWidth={1.5} />
                        </button>
                        {showSettingsMenu && (
                            <div className="activity-menu">
                                <button 
                                    className="activity-menu-item"
                                    onClick={() => {
                                        setShowSettingsMenu(false);
                                        setShowSettings(true);
                                    }}
                                >
                                    <User size={16} />
                                    Profile
                                </button>
                                <div className="activity-menu-divider" />
                                <button 
                                    className="activity-menu-item danger"
                                    onClick={() => {
                                        setShowSettingsMenu(false);
                                        logout();
                                    }}
                                >
                                    <LogOut size={16} />
                                    Logout
                                </button>
                            </div>
                        )}
                    </div>
                </div>
            </nav>

            {/* ── Sidebar ─────────────── */}
            <aside className={`collapsible-sidebar ${activeTray ? "open" : ""}`}>
                <div className="sidebar-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: "var(--accent-light)" }}>
                            <polyline points="16 18 22 12 16 6" />
                            <polyline points="8 6 2 12 8 18" />
                        </svg>
                        <h1>Clarix</h1>
                    </div>
                    {activeTray === "repos" && (
                        <button className="btn btn-primary btn-sm" onClick={() => setShowUpload(true)}>+ Add</button>
                    )}
                </div>

                {/* ── Repos + Conversations Tray ── */}
                <div style={{ display: activeTray === "repos" ? "flex" : "none", flexDirection: "column", flex: 1, minHeight: 0, overflowY: "auto", scrollbarWidth: "none" }}>
                    <div className="sidebar-section" style={{ borderBottom: "none" }}>
                        <span className="sidebar-section-title" style={{ marginBottom: 8 }}>Repositories</span>
                        <ul className="sidebar-list">
                            {repos.map((repo) => {
                                const isExpanded = expandedRepos.has(repo.id);
                                const isActive = activeRepo?.id === repo.id;
                                const convos = repoConversations.get(repo.id) || [];
                                const handleSelectRepo = () => {
                                    setActiveRepo(repo);
                                    toggleRepoExpanded(repo);
                                };
                                return (
                                    <li key={repo.id} style={{ marginBottom: 2 }}>
                                        <div
                                            className={`sidebar-item ${isActive ? "active" : ""}`}
                                            style={{ display: "flex", alignItems: "center", gap: 6 }}
                                            onClick={handleSelectRepo}
                                        >
                                            <span style={{ display: "inline-flex", alignItems: "center" }}>
                                                {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                                            </span>
                                            <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis" }}>
                                                {repo.name}
                                            </span>
                                            <span className={`status-badge ${repo.status}`}>
                                                {repo.status}
                                            </span>
                                        </div>
                                        {/* Expanded: show conversations for this repo */}
                                        {isExpanded && repo.status === "ready" && (
                                            <div style={{ paddingLeft: 20, borderLeft: "1px solid var(--border)", marginLeft: 10, marginTop: 2 }}>
                                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "4px 4px 4px 0" }}>
                                                    <span style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>Chats</span>
                                                    <button
                                                        className="btn btn-secondary btn-sm"
                                                        style={{ fontSize: 10, padding: "1px 6px" }}
                                                        onClick={(e) => { e.stopPropagation(); setActiveRepo(repo); startNewConversation(); }}
                                                    >
                                                        + New
                                                    </button>
                                                </div>
                                                {convos.length === 0 ? (
                                                    <div style={{ fontSize: 11, color: "var(--text-muted)", padding: "4px 0" }}>No conversations yet</div>
                                                ) : (
                                                    convos.map((conv) => (
                                                        <div
                                                            key={conv.id}
                                                            className={`sidebar-item ${activeConvId === conv.id ? "active" : ""}`}
                                                            style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, padding: "4px 6px" }}
                                                        >
                                                            {editingConvId === conv.id ? (
                                                                <div style={{ display: "flex", alignItems: "center", gap: 4, flex: 1 }}>
                                                                    <input
                                                                        type="text"
                                                                        value={editingConvTitle}
                                                                        onChange={(e) => setEditingConvTitle(e.target.value)}
                                                                        onKeyDown={(e) => {
                                                                            if (e.key === "Enter") handleRenameConversation(conv.id, repo.id);
                                                                            if (e.key === "Escape") setEditingConvId(null);
                                                                        }}
                                                                        autoFocus
                                                                        style={{
                                                                            flex: 1,
                                                                            background: "var(--bg)",
                                                                            border: "1px solid var(--accent)",
                                                                            borderRadius: 4,
                                                                            padding: "2px 6px",
                                                                            fontSize: 11,
                                                                            color: "var(--text-primary)",
                                                                        }}
                                                                        onClick={(e) => e.stopPropagation()}
                                                                    />
                                                                    <button
                                                                        onClick={(e) => { e.stopPropagation(); handleRenameConversation(conv.id, repo.id); }}
                                                                        style={{ background: "transparent", border: "none", cursor: "pointer", padding: 2, color: "var(--success)" }}
                                                                    >
                                                                        <Check size={12} />
                                                                    </button>
                                                                    <button
                                                                        onClick={(e) => { e.stopPropagation(); setEditingConvId(null); }}
                                                                        style={{ background: "transparent", border: "none", cursor: "pointer", padding: 2, color: "var(--text-muted)" }}
                                                                    >
                                                                        <X size={12} />
                                                                    </button>
                                                                </div>
                                                            ) : (
                                                                <>
                                                                    <MessageSquare size={12} style={{ color: "var(--text-muted)", flexShrink: 0 }} />
                                                                    <span
                                                                        onClick={() => { setActiveRepo(repo); loadConversationHistory(conv.id); }}
                                                                        style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", cursor: "pointer" }}
                                                                    >
                                                                        {conv.title.slice(0, 30)}
                                                                    </span>
                                                                    <div style={{ display: "flex", gap: 2, opacity: 0.5 }} className="conv-actions">
                                                                        <button
                                                                            onClick={(e) => { e.stopPropagation(); startEditingConversation(conv); }}
                                                                            style={{ background: "transparent", border: "none", cursor: "pointer", padding: 2, color: "var(--text-muted)" }}
                                                                            title="Rename"
                                                                        >
                                                                            <Edit2 size={10} />
                                                                        </button>
                                                                        <button
                                                                            onClick={(e) => { e.stopPropagation(); handleDeleteConversation(conv.id, repo.id); }}
                                                                            style={{ background: "transparent", border: "none", cursor: "pointer", padding: 2, color: "var(--error)" }}
                                                                            title="Delete"
                                                                        >
                                                                            <Trash2 size={10} />
                                                                        </button>
                                                                    </div>
                                                                </>
                                                            )}
                                                        </div>
                                                    ))
                                                )}
                                            </div>
                                        )}
                                    </li>
                                );
                            })}
                            {repos.length === 0 && (
                                <li className="sidebar-item" style={{ color: "var(--text-muted)", cursor: "default" }}>
                                    No repositories yet
                                </li>
                            )}
                        </ul>
                    </div>
                </div>

                {/* ── File Explorer Tray ── */}
                <div style={{ display: activeTray === "files" ? "flex" : "none", flexDirection: "column", flex: 1, minHeight: 0 }}>
                    {activeRepo?.status === "ready" ? (
                        <div className="sidebar-section sidebar-scroll" style={{ display: "flex", flexDirection: "column", flex: 1 }}>
                            <span className="sidebar-section-title">Files — {activeRepo.name}</span>
                            <div className="file-tree" style={{ flex: 1, overflowY: "auto", scrollbarWidth: "none" }}>
                                {files.length > 0 ? (
                                    <FileTreeNode 
                                        nodes={files} 
                                        expandedDirs={expandedDirs}
                                        fileContents={fileContents}
                                        onToggleDir={toggleDirectory}
                                        onOpenFile={openFile}
                                        depth={0}
                                    />
                                ) : (
                                    <div style={{ padding: "12px", color: "var(--text-muted)", fontSize: 12, lineHeight: 1.5 }}>
                                        Files unavailable — the repository may need to be re-ingested.
                                    </div>
                                )}
                            </div>
                        </div>
                    ) : (
                        <div style={{ padding: 16, color: "var(--text-muted)", fontSize: 13 }}>
                            {activeRepo ? `Repository is ${activeRepo.status}...` : "Select a repository to browse files."}
                        </div>
                    )}
                </div>
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
                        ) : activeRepo.status === "failed" ? (
                            <div className="empty-state">
                                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--error)" strokeWidth="1.5">
                                    <circle cx="12" cy="12" r="10" />
                                    <line x1="15" y1="9" x2="9" y2="15" />
                                    <line x1="9" y1="9" x2="15" y2="15" />
                                </svg>
                                <h3>Ingestion Failed</h3>
                                <p style={{ maxWidth: 480, lineHeight: 1.6 }}>
                                    {(() => {
                                        const raw = activeRepo.error_message || "An error occurred during ingestion.";
                                        // Extract a short user-friendly summary from verbose error
                                        const shortMsg = raw.length > 200 ? raw.slice(0, 200) + "…" : raw;
                                        return shortMsg;
                                    })()}
                                </p>
                                <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
                                    <button
                                        className="btn btn-primary btn-sm"
                                        onClick={async () => {
                                            // Delete failed repo and re-add
                                            try {
                                                const url = activeRepo.url;
                                                await deleteRepo(activeRepo.id);
                                                setRepos((prev) => prev.filter((r) => r.id !== activeRepo.id));
                                                if (url) {
                                                    setUploadUrl(url);
                                                    setShowUpload(true);
                                                }
                                                setActiveRepo(null);
                                                toast.info("Repository removed. You can re-add it.");
                                            } catch {
                                                toast.error("Failed to clean up. Try deleting manually.");
                                            }
                                        }}
                                    >
                                        Retry
                                    </button>
                                    <button
                                        className="btn btn-secondary btn-sm"
                                        onClick={() => handleDeleteRepo(activeRepo.id)}
                                        style={{ color: "var(--error)" }}
                                    >
                                        Delete
                                    </button>
                                </div>
                                {activeRepo.error_message && activeRepo.error_message.length > 200 && (
                                    <details style={{ marginTop: 12, maxWidth: 560, textAlign: "left", fontSize: 11, color: "var(--text-muted)" }}>
                                        <summary style={{ cursor: "pointer", fontSize: 12, color: "var(--text-secondary)" }}>Show full error</summary>
                                        <pre style={{ marginTop: 8, padding: 12, background: "var(--surface)", borderRadius: 8, overflowX: "auto", whiteSpace: "pre-wrap", wordBreak: "break-word", maxHeight: 200, overflowY: "auto", border: "1px solid var(--border)" }}>
                                            {activeRepo.error_message}
                                        </pre>
                                    </details>
                                )}
                            </div>
                        ) : (
                            <div className="ingestion-progress-container">
                                <div className="ingestion-progress-card">
                                    <div className="ingestion-icon-ring">
                                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="ingestion-code-icon">
                                            <polyline points="16 18 22 12 16 6" />
                                            <polyline points="8 6 2 12 8 18" />
                                        </svg>
                                    </div>
                                    <h3 className="ingestion-title">
                                        {activeRepo.status === "pending" ? "Preparing analysis..." : "Indexing codebase"}
                                    </h3>
                                    <div className="ingestion-steps">
                                        <IngestionStep 
                                            label="Clone" 
                                            active={activeRepo.ingestion_phase === "clone"} 
                                            done={["parse", "embed", "store", "done"].includes(activeRepo.ingestion_phase || "")} 
                                        />
                                        <div className="ingestion-step-connector" />
                                        <IngestionStep 
                                            label="Parse" 
                                            active={activeRepo.ingestion_phase === "parse"} 
                                            done={["embed", "store", "done"].includes(activeRepo.ingestion_phase || "")} 
                                        />
                                        <div className="ingestion-step-connector" />
                                        <IngestionStep 
                                            label="Embed" 
                                            active={activeRepo.ingestion_phase === "embed"} 
                                            done={["store", "done"].includes(activeRepo.ingestion_phase || "")} 
                                        />
                                        <div className="ingestion-step-connector" />
                                        <IngestionStep 
                                            label="Store" 
                                            active={activeRepo.ingestion_phase === "store"} 
                                            done={activeRepo.ingestion_phase === "done" || activeRepo.status === "ready"} 
                                        />
                                    </div>
                                    {activeRepo.status === "ingesting" && (
                                        <div className="ingestion-progress-section">
                                            <div className="ingestion-progress-bar-track">
                                                <div className="ingestion-progress-bar-fill" style={{ width: `${Math.max(activeRepo.ingestion_progress || 0, 2)}%` }} />
                                                <div className="ingestion-progress-bar-glow" style={{ width: `${Math.max(activeRepo.ingestion_progress || 0, 2)}%` }} />
                                            </div>
                                            <div className="ingestion-stats">
                                                <span className="ingestion-pct">{activeRepo.ingestion_progress || 0}%</span>
                                                {(activeRepo.ingestion_total_chunks || 0) > 0 && (
                                                    <span className="ingestion-detail">
                                                        {activeRepo.ingestion_total_chunks?.toLocaleString()} chunks
                                                        {(activeRepo.ingestion_cached_chunks || 0) > 0 && (
                                                            <span className="ingestion-cache-badge">⚡ {activeRepo.ingestion_cached_chunks?.toLocaleString()} cached</span>
                                                        )}
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                    )}
                                    <p className="ingestion-hint">
                                        {activeRepo.status === "pending" ? "Queued — starting shortly..." : (activeRepo.ingestion_cached_chunks || 0) > 0 ? "Using cached embeddings for unchanged code ⚡" : "Analyzing code structure and generating embeddings..."}
                                    </p>
                                </div>
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

            {/* ── Settings Modal ────────────────── */}
            <SettingsModal isOpen={showSettings} onClose={() => setShowSettings(false)} />

            {/* ── File Viewer ────────────────── */}
            <FileViewer file={viewingFile} onClose={() => setViewingFile(null)} />
        </div>
    );
}
// ── Ingestion Step Indicator Component ───────────────────────

function IngestionStep({ label, active, done }: { label: string; active: boolean; done: boolean }) {
    return (
        <div className={`ingestion-step-item ${active ? "active" : ""} ${done ? "done" : ""}`}>
            <div className="ingestion-step-dot">
                {done && (
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12" />
                    </svg>
                )}
            </div>
            <span className="ingestion-step-label">{label}</span>
        </div>
    );
}

// ── File Tree Node Component ────────────────────────────────

interface FileTreeNodeProps {
    nodes: FileNode[];
    expandedDirs: Set<string>;
    fileContents: Map<string, FileNode[]>;
    onToggleDir: (path: string) => void;
    onOpenFile: (path: string) => void;
    depth: number;
}

function FileTreeNode({ nodes, expandedDirs, fileContents, onToggleDir, onOpenFile, depth }: FileTreeNodeProps) {
    return (
        <>
            {nodes.map((node) => {
                const isExpanded = expandedDirs.has(node.path);
                const children = fileContents.get(node.path) || [];

                return (
                    <div key={node.path}>
                        <div
                            className="file-node"
                            onClick={() => {
                                if (node.type === "directory") {
                                    onToggleDir(node.path);
                                } else {
                                    onOpenFile(node.path);
                                }
                            }}
                            style={{
                                paddingLeft: `${12 + depth * 16}px`,
                                cursor: "pointer",
                                position: "relative",
                            }}
                        >
                            {/* Indent Guides */}
                            {Array.from({ length: depth }).map((_, i) => (
                                <div
                                    key={i}
                                    style={{
                                        width: 1,
                                        height: "100%",
                                        position: "absolute",
                                        left: 20 + i * 16,
                                        top: 0,
                                        background: "var(--border)",
                                    }}
                                />
                            ))}
                            {node.type === "directory" ? (
                                <>
                                    <span style={{ width: 16, display: "inline-flex", alignItems: "center", justifyContent: "center" }}>
                                        {isExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                                    </span>
                                    <Folder size={14} style={{ color: "var(--accent-light)", marginRight: 6 }} />
                                </>
                            ) : (
                                <>
                                    <span style={{ width: 16 }} />
                                    <File size={14} style={{ color: "var(--text-muted)", marginRight: 6 }} />
                                </>
                            )}
                            <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{node.name}</span>
                        </div>
                        {node.type === "directory" && isExpanded && children.length > 0 && (
                            <FileTreeNode
                                nodes={children}
                                expandedDirs={expandedDirs}
                                fileContents={fileContents}
                                onToggleDir={onToggleDir}
                                onOpenFile={onOpenFile}
                                depth={depth + 1}
                            />
                        )}
                    </div>
                );
            })}
        </>
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
    html = html.replace(/(<li>[\s\S]*?<\/li>)+/g, "<ul>$&</ul>");

    // Sanitize HTML with DOMPurify before returning
    return DOMPurify.sanitize(`<p>${html}</p>`, {
        ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'code', 'pre', 'h2', 'h3', 'h4', 'ul', 'li'],
        ALLOWED_ATTR: ['class'],
    });
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
        <ErrorBoundary>
            <ProtectedRoute>
                <DashboardContent />
            </ProtectedRoute>
        </ErrorBoundary>
    );
}
