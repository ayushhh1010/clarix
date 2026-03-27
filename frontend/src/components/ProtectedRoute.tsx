"use client";

import React, { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/components/AuthProvider";

/**
 * Wrap any page that requires authentication.
 * Redirects to /signin if not authenticated.
 */
export default function ProtectedRoute({ children }: { children: React.ReactNode }) {
    const { user, loading } = useAuth();
    const router = useRouter();

    useEffect(() => {
        if (!loading && !user) {
            router.replace("/signin");
        }
    }, [loading, user, router]);

    if (loading) {
        return (
            <div style={{
                minHeight: "100vh",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                background: "#050508",
                color: "rgba(255,255,255,0.5)",
                fontSize: "14px",
            }}>
                <div style={{ textAlign: "center" }}>
                    <div className="spinner" style={{ margin: "0 auto 16px" }} />
                    Loading...
                </div>
            </div>
        );
    }

    if (!user) return null;

    return <>{children}</>;
}
