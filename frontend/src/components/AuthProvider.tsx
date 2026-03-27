"use client";

import React, { createContext, useContext, useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
    AuthUser,
    getToken,
    setToken,
    removeToken,
    getCachedUser,
    setCachedUser,
    fetchCurrentUser,
    loginApi,
    registerApi,
    getGithubOAuthUrl,
    getGoogleOAuthUrl,
} from "@/lib/auth";

// ── Context Types ──────────────────────────────────────────

interface AuthContextType {
    user: AuthUser | null;
    loading: boolean;
    login: (email: string, password: string) => Promise<void>;
    register: (email: string, password: string, name: string) => Promise<void>;
    logout: () => void;
    loginWithGithub: () => void;
    loginWithGoogle: () => void;
    handleOAuthToken: (token: string) => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// ── Provider ───────────────────────────────────────────────

export function AuthProvider({ children }: { children: React.ReactNode }) {
    const [user, setUser] = useState<AuthUser | null>(null);
    const [loading, setLoading] = useState(true);
    const router = useRouter();

    // Load user on mount
    useEffect(() => {
        const init = async () => {
            const token = getToken();
            if (!token) {
                setLoading(false);
                return;
            }

            // Try cached user first for instant display
            const cached = getCachedUser();
            if (cached) setUser(cached);

            try {
                const freshUser = await fetchCurrentUser(token);
                setUser(freshUser);
                setCachedUser(freshUser);
            } catch {
                // Token invalid — clear
                removeToken();
                setUser(null);
            } finally {
                setLoading(false);
            }
        };
        init();
    }, []);

    const login = useCallback(async (email: string, password: string) => {
        const res = await loginApi(email, password);
        setToken(res.access_token);
        setCachedUser(res.user);
        setUser(res.user);
    }, []);

    const register = useCallback(async (email: string, password: string, name: string) => {
        const res = await registerApi(email, password, name);
        setToken(res.access_token);
        setCachedUser(res.user);
        setUser(res.user);
    }, []);

    const logout = useCallback(() => {
        removeToken();
        setUser(null);
        router.push("/");
    }, [router]);

    const loginWithGithub = useCallback(() => {
        window.location.href = getGithubOAuthUrl();
    }, []);

    const loginWithGoogle = useCallback(() => {
        window.location.href = getGoogleOAuthUrl();
    }, []);

    const handleOAuthToken = useCallback(async (token: string) => {
        setToken(token);
        try {
            const freshUser = await fetchCurrentUser(token);
            setUser(freshUser);
            setCachedUser(freshUser);
        } catch {
            removeToken();
            throw new Error("OAuth authentication failed");
        }
    }, []);

    return (
        <AuthContext.Provider
            value={{
                user,
                loading,
                login,
                register,
                logout,
                loginWithGithub,
                loginWithGoogle,
                handleOAuthToken,
            }}
        >
            {children}
        </AuthContext.Provider>
    );
}

// ── Hook ───────────────────────────────────────────────────

export function useAuth(): AuthContextType {
    const ctx = useContext(AuthContext);
    if (!ctx) throw new Error("useAuth must be used within AuthProvider");
    return ctx;
}
