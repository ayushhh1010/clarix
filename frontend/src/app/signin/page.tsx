"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useAuth } from "@/components/AuthProvider";
import { Mail, Lock, Eye, EyeOff, ArrowLeft, Loader2 } from "lucide-react";

export default function SignInPage() {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [showPassword, setShowPassword] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const router = useRouter();
    const { login, loginWithGithub, loginWithGoogle } = useAuth();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError("");
        setLoading(true);

        try {
            await login(email, password);
            router.push("/dashboard");
        } catch (err: any) {
            setError(err.message || "Invalid email or password");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="auth-page">
            {/* Animated background */}
            <div className="auth-bg">
                <div className="auth-orb auth-orb-1" />
                <div className="auth-orb auth-orb-2" />
                <div className="auth-orb auth-orb-3" />
            </div>

            <div className="auth-container">
                {/* Left: Branding */}
                <div className="auth-branding">
                    <Link href="/" className="auth-back-link">
                        <ArrowLeft size={16} />
                        Back to home
                    </Link>
                    <div className="auth-logo">
                        <h1>Clarix</h1>
                    </div>
                    <div className="auth-branding-content">

                        <h2>Welcome back</h2>
                        <p>
                            Sign in to access your codebases, conversations, and AI-powered
                            insights. Pick up right where you left off.
                        </p>
                        <div className="auth-stats">
                            <div className="auth-stat">
                                <div className="auth-stat-value">50K+</div>
                                <div className="auth-stat-label">Repos Analyzed</div>
                            </div>
                            <div className="auth-stat">
                                <div className="auth-stat-value">99.9%</div>
                                <div className="auth-stat-label">Uptime</div>
                            </div>
                            <div className="auth-stat">
                                <div className="auth-stat-value">&lt;2s</div>
                                <div className="auth-stat-label">Avg Response</div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Right: Form */}
                <div className="auth-form-panel">
                    <div className="auth-card">
                        <div className="auth-card-header">
                            <h2>Sign In</h2>
                            <p>Enter your credentials to continue</p>
                        </div>

                        {error && (
                            <div style={{
                                padding: "10px 14px",
                                background: "rgba(255, 85, 85, 0.1)",
                                border: "1px solid rgba(255, 85, 85, 0.3)",
                                borderRadius: "8px",
                                color: "#ff5555",
                                fontSize: "13px",
                                marginBottom: "20px",
                            }}>
                                {error}
                            </div>
                        )}

                        <form onSubmit={handleSubmit} className="auth-form">
                            <div className="auth-field">
                                <label>Email</label>
                                <div className="auth-input-wrapper">
                                    <Mail size={16} className="auth-input-icon" />
                                    <input
                                        type="email"
                                        placeholder="you@example.com"
                                        value={email}
                                        onChange={(e) => setEmail(e.target.value)}
                                        autoComplete="email"
                                        required
                                    />
                                </div>
                            </div>

                            <div className="auth-field">
                                <div className="auth-field-header">
                                    <label>Password</label>
                                    <a href="#" className="auth-forgot-link">Forgot password?</a>
                                </div>
                                <div className="auth-input-wrapper">
                                    <Lock size={16} className="auth-input-icon" />
                                    <input
                                        type={showPassword ? "text" : "password"}
                                        placeholder="••••••••"
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        autoComplete="current-password"
                                        required
                                    />
                                    <button
                                        type="button"
                                        className="auth-password-toggle"
                                        onClick={() => setShowPassword(!showPassword)}
                                    >
                                        {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                                    </button>
                                </div>
                            </div>

                            <button
                                type="submit"
                                className="auth-submit-btn"
                                disabled={loading}
                            >
                                {loading ? (
                                    <>
                                        <Loader2 size={16} className="spinning" />
                                        Signing in...
                                    </>
                                ) : (
                                    "Sign In"
                                )}
                            </button>
                        </form>

                        <div className="auth-divider" style={{ margin: "24px 0" }}>
                            <span>or continue with</span>
                        </div>

                        <div className="auth-social-buttons">
                            <button className="auth-social-btn" onClick={loginWithGithub} type="button">
                                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                                    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
                                </svg>
                                GitHub
                            </button>
                            <button className="auth-social-btn" onClick={loginWithGoogle} type="button">
                                <svg width="16" height="16" viewBox="0 0 16 16">
                                    <path d="M15.545 6.558a9.42 9.42 0 01.139 1.626c0 2.234-.636 4.131-1.946 5.543C12.486 15.143 10.57 16 8 16A8 8 0 018 0a7.689 7.689 0 015.352 2.082l-2.284 2.284A4.347 4.347 0 008 3.166c-2.087 0-3.86 1.408-4.492 3.304a4.792 4.792 0 000 3.063h.003c.635 1.893 2.405 3.301 4.492 3.301 1.078 0 2.004-.276 2.722-.764h-.003a3.702 3.702 0 001.599-2.431H8V6.558h7.545z" fill="#4285F4" />
                                </svg>
                                Google
                            </button>
                        </div>

                        <div className="auth-card-footer">
                            Don&apos;t have an account?{" "}
                            <Link href="/signup">Sign Up</Link>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
