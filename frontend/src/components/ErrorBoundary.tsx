"use client";

import React, { Component, ErrorInfo, ReactNode } from "react";

interface ErrorBoundaryProps {
    children: ReactNode;
    fallback?: ReactNode;
    onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface ErrorBoundaryState {
    hasError: boolean;
    error: Error | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
    constructor(props: ErrorBoundaryProps) {
        super(props);
        this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error: Error): ErrorBoundaryState {
        return { hasError: true, error };
    }

    componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
        console.error("ErrorBoundary caught:", error, errorInfo);
        this.props.onError?.(error, errorInfo);
    }

    handleRetry = (): void => {
        this.setState({ hasError: false, error: null });
    };

    render(): ReactNode {
        if (this.state.hasError) {
            if (this.props.fallback) {
                return this.props.fallback;
            }

            return (
                <div className="error-boundary">
                    <div className="error-content">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                            <circle cx="12" cy="12" r="10" />
                            <line x1="12" y1="8" x2="12" y2="12" />
                            <line x1="12" y1="16" x2="12.01" y2="16" />
                        </svg>
                        <h3>Something went wrong</h3>
                        <p>{this.state.error?.message || "An unexpected error occurred"}</p>
                        <button className="btn btn-primary" onClick={this.handleRetry}>
                            Try Again
                        </button>
                    </div>
                    <style jsx>{`
                        .error-boundary {
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            min-height: 200px;
                            padding: 2rem;
                        }
                        .error-content {
                            text-align: center;
                            max-width: 400px;
                        }
                        .error-content svg {
                            color: var(--error, #ef4444);
                            margin-bottom: 1rem;
                        }
                        .error-content h3 {
                            margin: 0 0 0.5rem;
                            color: var(--text-primary, #fff);
                        }
                        .error-content p {
                            color: var(--text-muted, #888);
                            margin: 0 0 1.5rem;
                            font-size: 0.9rem;
                        }
                    `}</style>
                </div>
            );
        }

        return this.props.children;
    }
}

// Hook for functional components to trigger error boundary
export function useErrorHandler() {
    const [, setError] = React.useState<Error | null>(null);
    
    return React.useCallback((error: Error) => {
        setError(() => {
            throw error;
        });
    }, []);
}

export default ErrorBoundary;
