import type { Metadata } from "next";
import "./globals.css";
import { AuthProvider } from "@/components/AuthProvider";

export const metadata: Metadata = {
    title: "Clarix — AI-Powered Codebase Intelligence",
    description:
        "Understand any codebase instantly with AI-powered semantic search, multi-agent architecture, and real-time streaming chat.",
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en">
            <head>
                <link rel="preconnect" href="https://fonts.googleapis.com" />
                <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
            </head>
            <body>
                <AuthProvider>{children}</AuthProvider>
            </body>
        </html>
    );
}
