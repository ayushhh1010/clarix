"use client";

import React, { useState, useEffect, Suspense } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useAuth } from "@/components/AuthProvider";
import { Mail, Lock, Eye, EyeOff, ArrowLeft, Loader2, CheckCircle, ArrowRight } from "lucide-react";
import { forgotPassword, resetPassword } from "@/lib/api";

type AuthMode = "signin" | "forgot" | "reset";

function SignInContent() {
    const searchParams = useSearchParams();
    const router = useRouter();
    const { login, loginWithGithub, loginWithGoogle } = useAuth();

    // Determine mode from URL params
    const modeParam = searchParams.get("mode");
    const tokenParam = searchParams.get("token");
    const initialMode: AuthMode = 
        modeParam === "reset" && tokenParam ? "reset" :
        modeParam === "forgot" ? "forgot" : "signin";

    const [mode, setMode] = useState<AuthMode>(initialMode);
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [newPassword, setNewPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");
    const [showPassword, setShowPassword] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [success, setSuccess] = useState("");

    // Update mode when URL changes
    useEffect(() => {
        const newMode: AuthMode = 
            modeParam === "reset" && tokenParam ? "reset" :
            modeParam === "forgot" ? "forgot" : "signin";
        setMode(newMode);
        setError("");
        setSuccess("");
    }, [modeParam, tokenParam]);

    const handleSignIn = async (e: React.FormEvent) => {
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

    const handleForgotPassword = async (e: React.FormEvent) => {
        e.preventDefault();
        setError("");
        setLoading(true);

        try {
            await forgotPassword(email);
            setSuccess("If an account exists with this email, you'll receive a reset link shortly.");
        } catch (err: any) {
            setError(err.message || "Failed to send reset email");
        } finally {
            setLoading(false);
        }
    };

    const handleResetPassword = async (e: React.FormEvent) => {
        e.preventDefault();
        setError("");

        if (newPassword.length < 6) {
            setError("Password must be at least 6 characters");
            return;
        }
        if (newPassword !== confirmPassword) {
            setError("Passwords do not match");
            return;
        }

        setLoading(true);
        try {
            await resetPassword(tokenParam!, newPassword);
            setSuccess("Password reset successful! Redirecting to sign in...");
            setTimeout(() => {
                router.push("/signin");
            }, 2000);
        } catch (err: any) {
            setError(err.message || "Failed to reset password. The link may have expired.");
        } finally {
            setLoading(false);
        }
    };

    const switchToForgot = () => {
        router.push("/signin?mode=forgot");
    };

    const switchToSignIn = () => {
        router.push("/signin");
    };

    // Render different forms based on mode
    const renderForm = () => {
        // Success state
        if (success) {
            return (
                <div style={{ textAlign: "center", padding: "20px 0" }}>
                    <CheckCircle size={48} style={{ color: "var(--success)", marginBottom: 16 }} />
                    <h3 style={{ marginBottom: 12, color: "var(--text-primary)" }}>{mode === "reset" ? "Password Reset!" : "Check Your Email"}</h3>
                    <p style={{ color: "var(--text-secondary)", marginBottom: 24, fontSize: 14 }}>
                        {success}
                    </p>
                    {mode === "forgot" && (
                        <button onClick={switchToSignIn} className="auth-submit-btn" style={{ maxWidth: 200, margin: "0 auto" }}>
                            Back to Sign In
                        </button>
                    )}
                </div>
            );
        }

        // Forgot password form
        if (mode === "forgot") {
            return (
                <>
                    <div className="auth-card-header">
                        <h2>Reset Password</h2>
                        <p>Enter your email to receive a reset link</p>
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

                    <form onSubmit={handleForgotPassword} className="auth-form">
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

                        <button type="submit" className="auth-submit-btn" disabled={loading}>
                            {loading ? (
                                <>
                                    <Loader2 size={16} className="spinning" />
                                    Sending...
                                </>
                            ) : (
                                <>
                                    Send Reset Link
                                    <ArrowRight size={16} />
                                </>
                            )}
                        </button>
                    </form>

                    <div className="auth-card-footer">
                        Remember your password?{" "}
                        <button onClick={switchToSignIn} style={{ background: "none", border: "none", color: "var(--accent)", cursor: "pointer", textDecoration: "underline" }}>
                            Sign In
                        </button>
                    </div>
                </>
            );
        }

        // Reset password form (with token)
        if (mode === "reset") {
            return (
                <>
                    <div className="auth-card-header">
                        <h2>Set New Password</h2>
                        <p>Enter your new password below</p>
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

                    <form onSubmit={handleResetPassword} className="auth-form">
                        <div className="auth-field">
                            <label>New Password</label>
                            <div className="auth-input-wrapper">
                                <Lock size={16} className="auth-input-icon" />
                                <input
                                    type={showPassword ? "text" : "password"}
                                    placeholder="Min. 6 characters"
                                    value={newPassword}
                                    onChange={(e) => setNewPassword(e.target.value)}
                                    autoComplete="new-password"
                                    required
                                    minLength={6}
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

                        <div className="auth-field">
                            <label>Confirm Password</label>
                            <div className="auth-input-wrapper">
                                <Lock size={16} className="auth-input-icon" />
                                <input
                                    type={showPassword ? "text" : "password"}
                                    placeholder="Re-enter password"
                                    value={confirmPassword}
                                    onChange={(e) => setConfirmPassword(e.target.value)}
                                    autoComplete="new-password"
                                    required
                                />
                            </div>
                        </div>

                        <button type="submit" className="auth-submit-btn" disabled={loading}>
                            {loading ? (
                                <>
                                    <Loader2 size={16} className="spinning" />
                                    Resetting...
                                </>
                            ) : (
                                "Reset Password"
                            )}
                        </button>
                    </form>

                    <div className="auth-card-footer">
                        <button onClick={switchToSignIn} style={{ background: "none", border: "none", color: "var(--accent)", cursor: "pointer", textDecoration: "underline" }}>
                            Back to Sign In
                        </button>
                    </div>
                </>
            );
        }

        // Default: Sign in form
        return (
            <>
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

                <form onSubmit={handleSignIn} className="auth-form">
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
                            <button 
                                type="button" 
                                onClick={switchToForgot} 
                                className="auth-forgot-link"
                                style={{ background: "none", border: "none", cursor: "pointer" }}
                            >
                                Forgot password?
                            </button>
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
            </>
        );
    };

    // Dynamic branding based on mode
    const getBrandingContent = () => {
        if (mode === "forgot") {
            return {
                title: "Password Recovery",
                description: "Forgot your password? No worries! Enter your email and we'll send you a link to reset it securely."
            };
        }
        if (mode === "reset") {
            return {
                title: "Almost There",
                description: "Create a strong new password to secure your account. Make sure it's at least 6 characters."
            };
        }
        return {
            title: "Welcome back",
            description: "Sign in to access your codebases, conversations, and AI-powered insights. Pick up right where you left off."
        };
    };

    const branding = getBrandingContent();

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
                        <h2>{branding.title}</h2>
                        <p>{branding.description}</p>
                        {mode === "signin" && (
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
                        )}
                    </div>
                </div>

                {/* Right: Form */}
                <div className="auth-form-panel">
                    <div className="auth-card">
                        {renderForm()}
                    </div>
                </div>
            </div>
        </div>
    );
}

export default function SignInPage() {
    return (
        <Suspense fallback={
            <div style={{
                minHeight: "100vh",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                background: "#050508",
                color: "rgba(255,255,255,0.5)",
            }}>
                Loading...
            </div>
        }>
            <SignInContent />
        </Suspense>
    );
}
