/**
 * Auth utilities: token management, auth API calls, OAuth helpers.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const TOKEN_KEY = "copilot_token";
const USER_KEY = "copilot_user";

// ── Types ──────────────────────────────────────────────────

export interface AuthUser {
    id: string;
    email: string;
    name: string;
    avatar_url: string | null;
    oauth_provider: string | null;
    created_at: string;
}

export interface TokenResponse {
    access_token: string;
    token_type: string;
    user: AuthUser;
}

// ── Token Management ───────────────────────────────────────

export function getToken(): string | null {
    if (typeof window === "undefined") return null;
    return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
    localStorage.setItem(TOKEN_KEY, token);
}

export function removeToken(): void {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
}

export function isAuthenticated(): boolean {
    return !!getToken();
}

export function getCachedUser(): AuthUser | null {
    if (typeof window === "undefined") return null;
    const raw = localStorage.getItem(USER_KEY);
    if (!raw) return null;
    try {
        return JSON.parse(raw);
    } catch {
        return null;
    }
}

export function setCachedUser(user: AuthUser): void {
    localStorage.setItem(USER_KEY, JSON.stringify(user));
}

// ── Auth API Calls ─────────────────────────────────────────

export async function loginApi(email: string, password: string): Promise<TokenResponse> {
    const res = await fetch(`${API_BASE}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
    });
    if (!res.ok) {
        const data = await res.json().catch(() => ({ detail: "Login failed" }));
        throw new Error(data.detail || "Login failed");
    }
    return res.json();
}

export async function registerApi(
    email: string,
    password: string,
    name: string
): Promise<TokenResponse> {
    const res = await fetch(`${API_BASE}/api/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, name, password }),
    });
    if (!res.ok) {
        const data = await res.json().catch(() => ({ detail: "Registration failed" }));
        throw new Error(data.detail || "Registration failed");
    }
    return res.json();
}

export async function fetchCurrentUser(token: string): Promise<AuthUser> {
    const res = await fetch(`${API_BASE}/api/auth/me`, {
        headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) {
        throw new Error("Not authenticated");
    }
    return res.json();
}

// ── OAuth Helpers ──────────────────────────────────────────

export function getGithubOAuthUrl(): string {
    return `${API_BASE}/api/auth/github`;
}

export function getGoogleOAuthUrl(): string {
    return `${API_BASE}/api/auth/google`;
}
