"use client";

import React from "react";
import { X, Copy, Download, ExternalLink } from "lucide-react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import { toast } from "sonner";

interface FileViewerProps {
    file: {
        path: string;
        content: string;
        language: string;
    } | null;
    onClose: () => void;
}

export default function FileViewer({ file, onClose }: FileViewerProps) {
    if (!file) return null;

    const fileName = file.path.split("/").pop() || file.path;
    const lineCount = file.content.split("\n").length;

    const handleCopy = () => {
        navigator.clipboard.writeText(file.content);
        toast.success("Copied to clipboard");
    };

    const handleDownload = () => {
        const blob = new Blob([file.content], { type: "text/plain" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = fileName;
        a.click();
        URL.revokeObjectURL(url);
        toast.success("Downloaded " + fileName);
    };

    // Map language names to Prism-compatible names
    const languageMap: Record<string, string> = {
        "javascript": "javascript",
        "typescript": "typescript",
        "python": "python",
        "java": "java",
        "go": "go",
        "rust": "rust",
        "c": "c",
        "cpp": "cpp",
        "csharp": "csharp",
        "ruby": "ruby",
        "php": "php",
        "swift": "swift",
        "kotlin": "kotlin",
        "scala": "scala",
        "html": "html",
        "css": "css",
        "scss": "scss",
        "json": "json",
        "yaml": "yaml",
        "xml": "xml",
        "markdown": "markdown",
        "bash": "bash",
        "sql": "sql",
        "plaintext": "text",
    };

    const prismLanguage = languageMap[file.language] || "text";

    return (
        <div 
            className="file-viewer-overlay"
            onClick={onClose}
            style={{
                position: "fixed",
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                background: "rgba(0, 0, 0, 0.7)",
                zIndex: 100,
                display: "flex",
                justifyContent: "flex-end",
            }}
        >
            <div 
                className="file-viewer-panel"
                onClick={(e) => e.stopPropagation()}
                style={{
                    width: "70%",
                    maxWidth: 900,
                    height: "100%",
                    background: "var(--surface)",
                    display: "flex",
                    flexDirection: "column",
                    animation: "slideInRight 0.2s ease-out",
                }}
            >
                {/* Header */}
                <div style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    padding: "12px 16px",
                    borderBottom: "1px solid var(--border)",
                    background: "var(--bg)",
                }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                        <button 
                            onClick={onClose}
                            style={{
                                background: "transparent",
                                border: "none",
                                cursor: "pointer",
                                padding: 4,
                                color: "var(--text-muted)",
                                display: "flex",
                            }}
                        >
                            <X size={18} />
                        </button>
                        <div>
                            <div style={{ 
                                fontSize: 14, 
                                fontWeight: 500, 
                                color: "var(--text-primary)",
                                fontFamily: "var(--font-mono)",
                            }}>
                                {fileName}
                            </div>
                            <div style={{ 
                                fontSize: 11, 
                                color: "var(--text-muted)",
                                marginTop: 2,
                            }}>
                                {file.path} • {lineCount} lines • {file.language}
                            </div>
                        </div>
                    </div>
                    <div style={{ display: "flex", gap: 8 }}>
                        <button
                            onClick={handleCopy}
                            title="Copy contents"
                            style={{
                                background: "rgba(255,255,255,0.05)",
                                border: "1px solid var(--border)",
                                borderRadius: 6,
                                padding: "6px 10px",
                                cursor: "pointer",
                                color: "var(--text-secondary)",
                                display: "flex",
                                alignItems: "center",
                                gap: 6,
                                fontSize: 12,
                            }}
                        >
                            <Copy size={14} />
                            Copy
                        </button>
                        <button
                            onClick={handleDownload}
                            title="Download file"
                            style={{
                                background: "rgba(255,255,255,0.05)",
                                border: "1px solid var(--border)",
                                borderRadius: 6,
                                padding: "6px 10px",
                                cursor: "pointer",
                                color: "var(--text-secondary)",
                                display: "flex",
                                alignItems: "center",
                                gap: 6,
                                fontSize: 12,
                            }}
                        >
                            <Download size={14} />
                            Download
                        </button>
                    </div>
                </div>

                {/* Code Content */}
                <div style={{ 
                    flex: 1, 
                    overflow: "auto",
                    fontSize: 13,
                }}>
                    <SyntaxHighlighter
                        language={prismLanguage}
                        style={vscDarkPlus}
                        showLineNumbers
                        customStyle={{
                            margin: 0,
                            padding: "16px",
                            background: "#1e1e1e",
                            minHeight: "100%",
                            fontSize: "13px",
                            lineHeight: 1.5,
                        }}
                        lineNumberStyle={{
                            minWidth: "3em",
                            paddingRight: "1em",
                            color: "#6e7681",
                            userSelect: "none",
                        }}
                    >
                        {file.content}
                    </SyntaxHighlighter>
                </div>
            </div>

            <style jsx global>{`
                @keyframes slideInRight {
                    from {
                        transform: translateX(100%);
                        opacity: 0;
                    }
                    to {
                        transform: translateX(0);
                        opacity: 1;
                    }
                }
            `}</style>
        </div>
    );
}
