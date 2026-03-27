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

export async function listRepos(): Promise<RepoResponse[]> {
    const res = await fetch(`${API_BASE}/api/repo/`, { headers: authHeaders() });
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

export async function listConversations(repoId: string): Promise<ConversationResponse[]> {
    const res = await fetch(`${API_BASE}/api/chat/conversations/${repoId}`, { headers: authHeaders() });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

export async function getChatHistory(conversationId: string): Promise<MessageResponse[]> {
    const res = await fetch(`${API_BASE}/api/chat/${conversationId}/history`, { headers: authHeaders() });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}
