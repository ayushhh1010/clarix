/**
 * API client for the AI Engineering Copilot backend.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

import { getToken } from "@/lib/auth";

/** Build auth headers if a token is present. */
function authHeaders(): Record<string, string> {
    const token = getToken();
    return token ? { Authorization: `Bearer ${token}` } : {};
}
export interface RepoResponse {
    id: string;
    name: string;
    url: string | null;
    status: string;
    file_count: number;
    chunk_count: number;
    error_message: string | null;
    created_at: string;
    updated_at: string;
}

export interface FileNode {
    name: string;
    path: string;
    type: "file" | "directory";
    children?: FileNode[];
}

export interface ChatResponse {
    conversation_id: string;
    message_id: string;
    content: string;
    sources: Source[];
}

export interface Source {
    file_path: string;
    language: string;
    start_line: number;
    end_line: number;
    chunk_type: string;
    name: string;
    relevance_score: number;
}

export interface AgentStep {
    step: string;
    agent: string;
    content: string;
}

export interface AgentRunResponse {
    conversation_id: string;
    steps: AgentStep[];
    final_answer: string;
}

export interface ConversationResponse {
    id: string;
    repo_id: string;
    title: string;
    created_at: string;
    updated_at: string;
}

export interface MessageResponse {
    id: string;
    role: string;
    content: string;
    metadata_json: string | null;
    created_at: string;
}

// Pagination wrapper
export interface PaginatedResponse<T> {
    items: T[];
    total: number;
    page: number;
    per_page: number;
    has_more: boolean;
}

// ── Repository APIs ────────────────────────────────────────

export async function uploadRepo(url: string): Promise<RepoResponse> {
    const res = await fetch(`${API_BASE}/api/repo/upload`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ url }),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

export async function getRepo(repoId: string): Promise<RepoResponse> {
    const res = await fetch(`${API_BASE}/api/repo/${repoId}`, { headers: authHeaders() });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

export async function getRepoStatus(repoId: string) {
    const res = await fetch(`${API_BASE}/api/repo/${repoId}/status`, { headers: authHeaders() });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

export async function listRepos(page = 1, perPage = 20): Promise<PaginatedResponse<RepoResponse>> {
    const res = await fetch(`${API_BASE}/api/repo/?page=${page}&per_page=${perPage}`, { headers: authHeaders() });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

export async function getRepoFiles(repoId: string, path = ""): Promise<FileNode[]> {
    const params = path ? `?path=${encodeURIComponent(path)}` : "";
    const res = await fetch(`${API_BASE}/api/repo/${repoId}/files${params}`, { headers: authHeaders() });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

export async function deleteRepo(repoId: string) {
    const res = await fetch(`${API_BASE}/api/repo/${repoId}`, { method: "DELETE", headers: authHeaders() });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

// ── Chat APIs ──────────────────────────────────────────────

export async function chat(
    repoId: string,
    message: string,
    conversationId?: string
): Promise<ChatResponse> {
    const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({
            repo_id: repoId,
            message,
            conversation_id: conversationId,
        }),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

export async function chatStream(
    repoId: string,
    message: string,
    conversationId: string | undefined,
    onChunk: (data: any) => void,
    onDone: () => void,
    onError: (err: Error) => void
) {
    try {
        const res = await fetch(`${API_BASE}/api/chat/stream`, {
            method: "POST",
            headers: { "Content-Type": "application/json", ...authHeaders() },
            body: JSON.stringify({
                repo_id: repoId,
                message,
                conversation_id: conversationId,
            }),
        });

        if (!res.ok) throw new Error(await res.text());
        if (!res.body) throw new Error("No response body");

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        onChunk(data);
                    } catch { }
                }
            }
        }

        onDone();
    } catch (err) {
        onError(err instanceof Error ? err : new Error(String(err)));
    }
}

// ── Agent APIs ─────────────────────────────────────────────

export async function runAgent(
    repoId: string,
    task: string,
    conversationId?: string
): Promise<AgentRunResponse> {
    const res = await fetch(`${API_BASE}/api/agent/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({
            repo_id: repoId,
            task,
            conversation_id: conversationId,
        }),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

export async function runAgentStream(
    repoId: string,
    task: string,
    conversationId: string | undefined,
    onEvent: (data: any) => void,
    onDone: () => void,
    onError: (err: Error) => void
) {
    try {
        const res = await fetch(`${API_BASE}/api/agent/run/stream`, {
            method: "POST",
            headers: { "Content-Type": "application/json", ...authHeaders() },
            body: JSON.stringify({
                repo_id: repoId,
                task,
                conversation_id: conversationId,
            }),
        });

        if (!res.ok) throw new Error(await res.text());
        if (!res.body) throw new Error("No response body");

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        onEvent(data);
                    } catch { }
                }
            }
        }

        onDone();
    } catch (err) {
        onError(err instanceof Error ? err : new Error(String(err)));
    }
}

// ── Conversation APIs ──────────────────────────────────────

export async function listConversations(repoId: string, page = 1, perPage = 20): Promise<PaginatedResponse<ConversationResponse>> {
    const res = await fetch(`${API_BASE}/api/chat/conversations/${repoId}?page=${page}&per_page=${perPage}`, { headers: authHeaders() });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

export async function getChatHistory(conversationId: string, page = 1, perPage = 50): Promise<PaginatedResponse<MessageResponse>> {
    const res = await fetch(`${API_BASE}/api/chat/${conversationId}/history?page=${page}&per_page=${perPage}`, { headers: authHeaders() });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

export async function deleteConversation(conversationId: string): Promise<void> {
    const res = await fetch(`${API_BASE}/api/chat/conversations/${conversationId}`, {
        method: "DELETE",
        headers: authHeaders(),
    });
    if (!res.ok) throw new Error(await res.text());
}

export async function renameConversation(conversationId: string, title: string): Promise<void> {
    const res = await fetch(`${API_BASE}/api/chat/conversations/${conversationId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ title }),
    });
    if (!res.ok) throw new Error(await res.text());
}

// ── File Content API ───────────────────────────────────────

export async function getFileContent(repoId: string, path: string): Promise<{ content: string; language: string }> {
    const res = await fetch(`${API_BASE}/api/repo/${repoId}/file-content?path=${encodeURIComponent(path)}`, {
        headers: authHeaders(),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

// ── Auth APIs ──────────────────────────────────────────────

export async function forgotPassword(email: string): Promise<void> {
    const res = await fetch(`${API_BASE}/api/auth/forgot-password`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
    });
    if (!res.ok) throw new Error(await res.text());
}

export async function resetPassword(token: string, newPassword: string): Promise<void> {
    const res = await fetch(`${API_BASE}/api/auth/reset-password`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token, new_password: newPassword }),
    });
    if (!res.ok) throw new Error(await res.text());
}

export async function updateProfile(name: string, email: string): Promise<void> {
    const res = await fetch(`${API_BASE}/api/auth/profile`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ name, email }),
    });
    if (!res.ok) throw new Error(await res.text());
}

export async function changePassword(currentPassword: string, newPassword: string): Promise<void> {
    const res = await fetch(`${API_BASE}/api/auth/change-password`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
    });
    if (!res.ok) throw new Error(await res.text());
}

export async function deleteAccount(): Promise<void> {
    const res = await fetch(`${API_BASE}/api/auth/account`, {
        method: "DELETE",
        headers: authHeaders(),
    });
    if (!res.ok) throw new Error(await res.text());
}
