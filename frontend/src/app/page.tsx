"use client";

import Link from "next/link";
import { useEffect, useRef } from "react";

export default function LandingPage() {
    const heroRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        // Parallax effect on mouse move
        const handleMouseMove = (e: MouseEvent) => {
            const orbs = document.querySelectorAll<HTMLElement>(".landing-orb");
            const floats = document.querySelectorAll<HTMLElement>(".floating-point");
            const x = e.clientX / window.innerWidth;
            const y = e.clientY / window.innerHeight;

            orbs.forEach((orb, i) => {
                const speed = (i + 1) * 15;
                orb.style.transform = `translate(${x * speed}px, ${y * speed}px)`;
            });

            floats.forEach((el, i) => {
                const speed = (i + 1) * 8;
                el.style.transform = `translate(${x * speed - speed / 2}px, ${y * speed - speed / 2}px)`;
            });
        };

        window.addEventListener("mousemove", handleMouseMove);
        return () => window.removeEventListener("mousemove", handleMouseMove);
    }, []);

    return (
        <div className="landing-page">
            {/* ── Ambient Background ── */}
            <div className="landing-bg">
                <div className="landing-orb landing-orb-1" />
                <div className="landing-orb landing-orb-2" />
                <div className="landing-orb landing-orb-3" />
                <div className="landing-grid" />
            </div>

            {/* ── Navbar ── */}
            <nav className="landing-nav">
                <div className="landing-nav-inner">
                    <Link href="/" className="landing-logo">
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="16 18 22 12 16 6" />
                            <polyline points="8 6 2 12 8 18" />
                        </svg>
                        <span>Clarix</span>
                    </Link>

                    <div className="landing-nav-links">
                        <a href="#features">Features</a>
                        <a href="#how-it-works">How It Works</a>
                        <a href="#trusted">Trusted By</a>
                    </div>

                    <div className="landing-nav-actions">
                        <Link href="/signin" className="landing-nav-signin">
                            Sign In
                        </Link>
                        <Link href="/signup" className="landing-nav-cta">
                            Get Started
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                <path d="M5 12h14M12 5l7 7-7 7" />
                            </svg>
                        </Link>
                    </div>
                </div>
            </nav>

            {/* ── Hero Section ── */}
            <section className="landing-hero" ref={heroRef}>
                {/* Floating data points */}
                <div className="floating-point fp-1">
                    <div className="fp-icon">◇</div>
                    <div className="fp-label">Codebase</div>
                    <div className="fp-value">247K lines</div>
                </div>
                <div className="floating-point fp-2">
                    <div className="fp-icon">⬡</div>
                    <div className="fp-label">Chunks</div>
                    <div className="fp-value">12,845</div>
                </div>
                <div className="floating-point fp-3">
                    <div className="fp-icon">△</div>
                    <div className="fp-label">Accuracy</div>
                    <div className="fp-value">98.7%</div>
                </div>
                <div className="floating-point fp-4">
                    <div className="fp-icon">○</div>
                    <div className="fp-label">Latency</div>
                    <div className="fp-value">&lt;200ms</div>
                </div>

                <div className="landing-hero-content">
                    <div className="landing-badge">
                        <span className="landing-badge-dot" />
                        Unlock Your Code Superpowers →
                    </div>

                    <h1 className="landing-title">
                        One-click for
                        <br />
                        <span className="landing-title-gradient">Codebase Intelligence</span>
                    </h1>

                    <p className="landing-subtitle">
                        Dive into any codebase with AI-powered understanding. Ask questions,
                        debug issues, trace flows, and get instant answers with full semantic context.
                    </p>

                    <div className="landing-hero-actions">
                        <Link href="/dashboard" className="landing-btn-primary">
                            Open App
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                <path d="M7 17L17 7M17 7H7M17 7v10" />
                            </svg>
                        </Link>
                        <a href="#features" className="landing-btn-secondary">
                            Discover More
                        </a>
                    </div>
                </div>

                {/* Scroll indicator */}
                <div className="landing-scroll-indicator">
                    <div className="landing-scroll-icon">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M12 5v14M19 12l-7 7-7-7" />
                        </svg>
                    </div>
                    <span>Scroll down</span>
                </div>
            </section>

            {/* ── Trusted By ── */}
            <section className="landing-trusted" id="trusted">
                <div className="landing-trusted-track">
                    {["GitHub", "Vercel", "Google", "Microsoft", "Stripe", "Notion", "Linear", "Figma"].map((brand) => (
                        <div key={brand} className="landing-trusted-logo">
                            {brand}
                        </div>
                    ))}
                </div>
            </section>

            {/* ── Features ── */}
            <section className="landing-features" id="features">
                <div className="landing-features-inner">
                    <div className="landing-section-label">Features</div>
                    <h2 className="landing-section-title">
                        Everything you need to understand
                        <br />
                        <span className="landing-title-gradient">any codebase, instantly</span>
                    </h2>

                    <div className="landing-features-grid">
                        <div className="landing-feature-card">
                            <div className="landing-feature-icon">
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                    <circle cx="11" cy="11" r="8" />
                                    <path d="m21 21-4.35-4.35" />
                                </svg>
                            </div>
                            <h3>Semantic Code Search</h3>
                            <p>
                                Search by meaning, not keywords. Our RAG pipeline understands code
                                semantically and retrieves the most relevant chunks.
                            </p>
                        </div>

                        <div className="landing-feature-card">
                            <div className="landing-feature-icon">
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                    <path d="M12 2L2 7l10 5 10-5-10-5z" />
                                    <path d="M2 17l10 5 10-5" />
                                    <path d="M2 12l10 5 10-5" />
                                </svg>
                            </div>
                            <h3>Multi-Agent Architecture</h3>
                            <p>
                                Planner, Retrieval, Tool, and Executor agents work together to understand,
                                debug, and even modify code autonomously.
                            </p>
                        </div>

                        <div className="landing-feature-card">
                            <div className="landing-feature-icon">
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                    <polyline points="16 18 22 12 16 6" />
                                    <polyline points="8 6 2 12 8 18" />
                                </svg>
                            </div>
                            <h3>Code Understanding</h3>
                            <p>
                                Get deep explanations of code flows, architecture patterns,
                                and dependencies. Like having a senior engineer on demand.
                            </p>
                        </div>

                        <div className="landing-feature-card">
                            <div className="landing-feature-icon">
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                                    <line x1="3" y1="9" x2="21" y2="9" />
                                    <line x1="9" y1="21" x2="9" y2="9" />
                                </svg>
                            </div>
                            <h3>Full Repo Ingestion</h3>
                            <p>
                                Clone any Git repo, and we&apos;ll parse, chunk, embed, and index
                                the entire codebase automatically.
                            </p>
                        </div>

                        <div className="landing-feature-card">
                            <div className="landing-feature-icon">
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                                </svg>
                            </div>
                            <h3>Streaming Chat</h3>
                            <p>
                                Real-time streaming responses with full source attribution.
                                See exactly where in the code each answer comes from.
                            </p>
                        </div>

                        <div className="landing-feature-card">
                            <div className="landing-feature-icon">
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                                </svg>
                            </div>
                            <h3>Persistent Memory</h3>
                            <p>
                                Conversations are stored with full context. Return to any discussion
                                and pick up right where you left off.
                            </p>
                        </div>
                    </div>
                </div>
            </section>

            {/* ── How It Works ── */}
            <section className="landing-how" id="how-it-works">
                <div className="landing-how-inner">
                    <div className="landing-section-label">How It Works</div>
                    <h2 className="landing-section-title">
                        Three steps to
                        <span className="landing-title-gradient"> codebase mastery</span>
                    </h2>

                    <div className="landing-steps">
                        <div className="landing-step">
                            <div className="landing-step-number">01</div>
                            <h3>Add Your Repository</h3>
                            <p>Paste a Git URL and we&apos;ll clone, parse, and intelligently chunk your entire codebase.</p>
                        </div>
                        <div className="landing-step-divider" />
                        <div className="landing-step">
                            <div className="landing-step-number">02</div>
                            <h3>Ask Anything</h3>
                            <p>Ask questions in natural language. Our AI retrieves the most relevant code and generates accurate answers.</p>
                        </div>
                        <div className="landing-step-divider" />
                        <div className="landing-step">
                            <div className="landing-step-number">03</div>
                            <h3>Ship Faster</h3>
                            <p>Debug, refactor, and understand any codebase in minutes instead of hours. Let the agents handle the grunt work.</p>
                        </div>
                    </div>
                </div>
            </section>

            {/* ── CTA Section ── */}
            <section className="landing-cta-section">
                <div className="landing-cta-inner">
                    <h2>Ready to supercharge your<br /><span className="landing-title-gradient">development workflow?</span></h2>
                    <p>Start analyzing codebases with AI-powered understanding. Free to get started.</p>
                    <div className="landing-hero-actions" style={{ justifyContent: "center" }}>
                        <Link href="/signup" className="landing-btn-primary">
                            Create Free Account
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                <path d="M5 12h14M12 5l7 7-7 7" />
                            </svg>
                        </Link>
                    </div>
                </div>
            </section>

            {/* ── Footer ── */}
            <footer className="landing-footer">
                <div className="landing-footer-inner">
                    <div className="landing-footer-brand">
                        <div className="landing-logo" style={{ marginBottom: 12 }}>
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <polyline points="16 18 22 12 16 6" />
                                <polyline points="8 6 2 12 8 18" />
                            </svg>
                            <span>Clarix</span>
                        </div>
                        <p>AI-powered codebase understanding for modern engineering teams.</p>
                    </div>
                    <div className="landing-footer-links">
                        <div className="landing-footer-col">
                            <h4>Product</h4>
                            <a href="#features">Features</a>
                            <a href="#how-it-works">How It Works</a>
                            <Link href="/dashboard">Dashboard</Link>
                        </div>
                        <div className="landing-footer-col">
                            <h4>Company</h4>
                            <a href="#">About</a>
                            <a href="#">Blog</a>
                            <a href="#">Careers</a>
                        </div>
                        <div className="landing-footer-col">
                            <h4>Legal</h4>
                            <a href="#">Privacy</a>
                            <a href="#">Terms</a>
                            <a href="#">Security</a>
                        </div>
                    </div>
                </div>
                <div className="landing-footer-bottom">
                    <p>&copy; 2026 Clarix. All rights reserved.</p>
                </div>
            </footer>
        </div>
    );
}
