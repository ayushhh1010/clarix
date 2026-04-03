"use client";

import React, { useState, useEffect } from "react";
import { X, User, Shield, AlertTriangle, Loader2, Check } from "lucide-react";
import { useAuth } from "@/components/AuthProvider";
import { updateProfile, changePassword, deleteAccount } from "@/lib/api";
import { toast } from "sonner";

interface SettingsModalProps {
    isOpen: boolean;
    onClose: () => void;
}

type Tab = "profile" | "security" | "danger";

export default function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
    const { user, logout } = useAuth();
    const [activeTab, setActiveTab] = useState<Tab>("profile");
    
    // Profile form
    const [name, setName] = useState("");
    const [email, setEmail] = useState("");
    const [profileLoading, setProfileLoading] = useState(false);
    
    // Security form
    const [currentPassword, setCurrentPassword] = useState("");
    const [newPassword, setNewPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");
    const [securityLoading, setSecurityLoading] = useState(false);
    
    // Danger zone
    const [deleteConfirm, setDeleteConfirm] = useState("");
    const [deleteLoading, setDeleteLoading] = useState(false);

    // Initialize form with user data
    useEffect(() => {
        if (user) {
            setName(user.name || "");
            setEmail(user.email || "");
        }
    }, [user]);

    // Reset form when modal closes
    useEffect(() => {
        if (!isOpen) {
            setActiveTab("profile");
            setCurrentPassword("");
            setNewPassword("");
            setConfirmPassword("");
            setDeleteConfirm("");
        }
    }, [isOpen]);

    if (!isOpen) return null;

    const handleProfileSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setProfileLoading(true);
        try {
            await updateProfile(name, email);
            toast.success("Profile updated successfully");
        } catch (err: any) {
            toast.error(err.message || "Failed to update profile");
        } finally {
            setProfileLoading(false);
        }
    };

    const handlePasswordSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        
        if (newPassword.length < 6) {
            toast.error("New password must be at least 6 characters");
            return;
        }
        if (newPassword !== confirmPassword) {
            toast.error("Passwords do not match");
            return;
        }

        setSecurityLoading(true);
        try {
            await changePassword(currentPassword, newPassword);
            toast.success("Password changed successfully");
            setCurrentPassword("");
            setNewPassword("");
            setConfirmPassword("");
        } catch (err: any) {
            toast.error(err.message || "Failed to change password");
        } finally {
            setSecurityLoading(false);
        }
    };

    const handleDeleteAccount = async () => {
        if (deleteConfirm !== "DELETE") {
            toast.error("Please type DELETE to confirm");
            return;
        }

        setDeleteLoading(true);
        try {
            await deleteAccount();
            toast.success("Account deleted");
            logout();
            onClose();
        } catch (err: any) {
            toast.error(err.message || "Failed to delete account");
        } finally {
            setDeleteLoading(false);
        }
    };

    const tabs = [
        { id: "profile" as Tab, label: "Profile", icon: User },
        { id: "security" as Tab, label: "Security", icon: Shield },
        { id: "danger" as Tab, label: "Danger Zone", icon: AlertTriangle },
    ];

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div 
                className="settings-modal" 
                onClick={(e) => e.stopPropagation()}
                style={{
                    background: "var(--surface)",
                    borderRadius: 12,
                    width: "100%",
                    maxWidth: 600,
                    maxHeight: "85vh",
                    overflow: "hidden",
                    display: "flex",
                    flexDirection: "column",
                }}
            >
                {/* Header */}
                <div style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    padding: "16px 20px",
                    borderBottom: "1px solid var(--border)",
                }}>
                    <h2 style={{ fontSize: 18, fontWeight: 600 }}>Settings</h2>
                    <button 
                        onClick={onClose}
                        style={{
                            background: "transparent",
                            border: "none",
                            cursor: "pointer",
                            padding: 4,
                            color: "var(--text-muted)",
                        }}
                    >
                        <X size={20} />
                    </button>
                </div>

                {/* Tabs */}
                <div style={{
                    display: "flex",
                    borderBottom: "1px solid var(--border)",
                    padding: "0 20px",
                }}>
                    {tabs.map((tab) => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            style={{
                                display: "flex",
                                alignItems: "center",
                                gap: 8,
                                padding: "12px 16px",
                                background: "transparent",
                                border: "none",
                                borderBottom: activeTab === tab.id ? "2px solid var(--accent)" : "2px solid transparent",
                                color: activeTab === tab.id ? "var(--text-primary)" : "var(--text-muted)",
                                cursor: "pointer",
                                fontSize: 13,
                                fontWeight: 500,
                                marginBottom: -1,
                            }}
                        >
                            <tab.icon size={16} />
                            {tab.label}
                        </button>
                    ))}
                </div>

                {/* Content */}
                <div style={{ padding: 20, overflowY: "auto", flex: 1 }}>
                    {/* Profile Tab */}
                    {activeTab === "profile" && (
                        <form onSubmit={handleProfileSubmit}>
                            <div style={{ marginBottom: 20 }}>
                                <label style={{ display: "block", marginBottom: 6, fontSize: 13, color: "var(--text-secondary)" }}>
                                    Display Name
                                </label>
                                <input
                                    type="text"
                                    value={name}
                                    onChange={(e) => setName(e.target.value)}
                                    placeholder="Your name"
                                    style={{
                                        width: "100%",
                                        padding: "10px 12px",
                                        background: "var(--bg)",
                                        border: "1px solid var(--border)",
                                        borderRadius: 8,
                                        color: "var(--text-primary)",
                                        fontSize: 14,
                                    }}
                                />
                            </div>
                            <div style={{ marginBottom: 20 }}>
                                <label style={{ display: "block", marginBottom: 6, fontSize: 13, color: "var(--text-secondary)" }}>
                                    Email Address
                                </label>
                                <input
                                    type="email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    placeholder="you@example.com"
                                    style={{
                                        width: "100%",
                                        padding: "10px 12px",
                                        background: "var(--bg)",
                                        border: "1px solid var(--border)",
                                        borderRadius: 8,
                                        color: "var(--text-primary)",
                                        fontSize: 14,
                                    }}
                                />
                            </div>
                            {user?.oauth_provider && (
                                <div style={{
                                    padding: "10px 12px",
                                    background: "rgba(99, 102, 241, 0.1)",
                                    border: "1px solid rgba(99, 102, 241, 0.2)",
                                    borderRadius: 8,
                                    fontSize: 13,
                                    color: "var(--text-secondary)",
                                    marginBottom: 20,
                                }}>
                                    Connected via {user.oauth_provider.charAt(0).toUpperCase() + user.oauth_provider.slice(1)}
                                </div>
                            )}
                            <button
                                type="submit"
                                disabled={profileLoading}
                                className="btn btn-primary"
                                style={{ display: "flex", alignItems: "center", gap: 8 }}
                            >
                                {profileLoading ? <Loader2 size={16} className="spinning" /> : <Check size={16} />}
                                Save Changes
                            </button>
                        </form>
                    )}

                    {/* Security Tab */}
                    {activeTab === "security" && (
                        <form onSubmit={handlePasswordSubmit}>
                            {user?.oauth_provider && !user?.email?.includes("@") ? (
                                <div style={{
                                    padding: "16px",
                                    background: "rgba(255, 193, 7, 0.1)",
                                    border: "1px solid rgba(255, 193, 7, 0.2)",
                                    borderRadius: 8,
                                    fontSize: 13,
                                    color: "var(--text-secondary)",
                                }}>
                                    You signed up with {user.oauth_provider}. Password management is handled by your OAuth provider.
                                </div>
                            ) : (
                                <>
                                    <div style={{ marginBottom: 20 }}>
                                        <label style={{ display: "block", marginBottom: 6, fontSize: 13, color: "var(--text-secondary)" }}>
                                            Current Password
                                        </label>
                                        <input
                                            type="password"
                                            value={currentPassword}
                                            onChange={(e) => setCurrentPassword(e.target.value)}
                                            placeholder="••••••••"
                                            required
                                            style={{
                                                width: "100%",
                                                padding: "10px 12px",
                                                background: "var(--bg)",
                                                border: "1px solid var(--border)",
                                                borderRadius: 8,
                                                color: "var(--text-primary)",
                                                fontSize: 14,
                                            }}
                                        />
                                    </div>
                                    <div style={{ marginBottom: 20 }}>
                                        <label style={{ display: "block", marginBottom: 6, fontSize: 13, color: "var(--text-secondary)" }}>
                                            New Password
                                        </label>
                                        <input
                                            type="password"
                                            value={newPassword}
                                            onChange={(e) => setNewPassword(e.target.value)}
                                            placeholder="Min. 6 characters"
                                            required
                                            minLength={6}
                                            style={{
                                                width: "100%",
                                                padding: "10px 12px",
                                                background: "var(--bg)",
                                                border: "1px solid var(--border)",
                                                borderRadius: 8,
                                                color: "var(--text-primary)",
                                                fontSize: 14,
                                            }}
                                        />
                                    </div>
                                    <div style={{ marginBottom: 20 }}>
                                        <label style={{ display: "block", marginBottom: 6, fontSize: 13, color: "var(--text-secondary)" }}>
                                            Confirm New Password
                                        </label>
                                        <input
                                            type="password"
                                            value={confirmPassword}
                                            onChange={(e) => setConfirmPassword(e.target.value)}
                                            placeholder="Re-enter new password"
                                            required
                                            style={{
                                                width: "100%",
                                                padding: "10px 12px",
                                                background: "var(--bg)",
                                                border: "1px solid var(--border)",
                                                borderRadius: 8,
                                                color: "var(--text-primary)",
                                                fontSize: 14,
                                            }}
                                        />
                                    </div>
                                    <button
                                        type="submit"
                                        disabled={securityLoading}
                                        className="btn btn-primary"
                                        style={{ display: "flex", alignItems: "center", gap: 8 }}
                                    >
                                        {securityLoading ? <Loader2 size={16} className="spinning" /> : <Shield size={16} />}
                                        Change Password
                                    </button>
                                </>
                            )}
                        </form>
                    )}

                    {/* Danger Zone Tab */}
                    {activeTab === "danger" && (
                        <div>
                            <div style={{
                                padding: 16,
                                background: "rgba(239, 68, 68, 0.1)",
                                border: "1px solid rgba(239, 68, 68, 0.2)",
                                borderRadius: 8,
                                marginBottom: 20,
                            }}>
                                <h3 style={{ color: "#ef4444", fontSize: 14, marginBottom: 8, display: "flex", alignItems: "center", gap: 8 }}>
                                    <AlertTriangle size={16} />
                                    Delete Account
                                </h3>
                                <p style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 16 }}>
                                    Once you delete your account, there is no going back. All your repositories, conversations, and data will be permanently deleted.
                                </p>
                                <div style={{ marginBottom: 12 }}>
                                    <label style={{ display: "block", marginBottom: 6, fontSize: 12, color: "var(--text-muted)" }}>
                                        Type <strong>DELETE</strong> to confirm
                                    </label>
                                    <input
                                        type="text"
                                        value={deleteConfirm}
                                        onChange={(e) => setDeleteConfirm(e.target.value)}
                                        placeholder="DELETE"
                                        style={{
                                            width: "100%",
                                            padding: "10px 12px",
                                            background: "var(--bg)",
                                            border: "1px solid rgba(239, 68, 68, 0.3)",
                                            borderRadius: 8,
                                            color: "var(--text-primary)",
                                            fontSize: 14,
                                        }}
                                    />
                                </div>
                                <button
                                    onClick={handleDeleteAccount}
                                    disabled={deleteLoading || deleteConfirm !== "DELETE"}
                                    style={{
                                        padding: "10px 16px",
                                        background: deleteConfirm === "DELETE" ? "#ef4444" : "rgba(239, 68, 68, 0.3)",
                                        border: "none",
                                        borderRadius: 8,
                                        color: "#fff",
                                        fontSize: 13,
                                        fontWeight: 500,
                                        cursor: deleteConfirm === "DELETE" ? "pointer" : "not-allowed",
                                        display: "flex",
                                        alignItems: "center",
                                        gap: 8,
                                    }}
                                >
                                    {deleteLoading ? <Loader2 size={16} className="spinning" /> : <AlertTriangle size={16} />}
                                    Delete My Account
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
