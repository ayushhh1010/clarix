"use client";

import { useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useAuth } from "@/components/AuthProvider";
import { Suspense } from "react";

/**
 * OAuth callback page.
 * GitHub/Google OAuth redirects here with ?token=xxx
 * We store the token and redirect to /dashboard.
 */
function CallbackHandler() {
    const searchParams = useSearchParams();
    const router = useRouter();
    const { handleOAuthToken } = useAuth();

    useEffect(() => {
        const token = searchParams.get("token");
        if (token) {
            handleOAuthToken(token)
                .then(() => router.replace("/dashboard"))
                .catch(() => router.replace("/signin?error=oauth_failed"));
        } else {
            router.replace("/signin?error=no_token");
        }
    }, [searchParams, router, handleOAuthToken]);

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
                Completing sign in...
            </div>
        </div>
    );
}

export default function AuthCallbackPage() {
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
            <CallbackHandler />
        </Suspense>
    );
}
