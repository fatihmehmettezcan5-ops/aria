"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { ChatSession, DocumentInfo, ModelInfo } from "@/lib/types";
import { useUI } from "@/lib/store";
import { Sidebar } from "@/components/sidebar/Sidebar";
import { ChatArea } from "@/components/chat/ChatArea";
import { SettingsModal } from "@/components/SettingsModal";

export default function Home() {
  const { lang, setLang } = useUI();
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const active = sessions.find((s) => s.id === activeId) || null;

  async function refresh() {
    try {
      const [s, d, m] = await Promise.all([
        api.listSessions(),
        api.listDocuments(),
        api.modelInfo(),
      ]);
      setSessions(s); setDocs(d); setInfo(m);
      if (!activeId && s.length > 0) setActiveId(s[0].id);
    } catch (e) { console.error(e); }
  }

  useEffect(() => { refresh(); }, []);

  async function newSession() {
    const s = await api.createSession(lang);
    setSessions((cs) => [s, ...cs]);
    setActiveId(s.id);
  }
  async function deleteSession(id: string) {
    await api.deleteSession(id);
    setSessions((cs) => cs.filter((s) => s.id !== id));
    if (activeId === id) setActiveId(null);
  }
  async function deleteDoc(id: string) {
    await api.deleteDocument(id);
    setDocs((ds) => ds.filter((d) => d.id !== id));
  }

  return (
    <div className="h-full flex">
      <Sidebar
        lang={lang}
        sessions={sessions}
        documents={docs}
        activeId={activeId}
        modelInfo={info}
        onSelect={setActiveId}
        onNew={newSession}
        onDelete={deleteSession}
        onDeleteDoc={deleteDoc}
        onOpenSettings={() => setSettingsOpen(true)}
      />
      <ChatArea
        lang={lang}
        session={active}
        onSessionUpdated={(s) => setSessions((cs) => cs.map((x) => x.id === s.id ? s : x))}
        onUploaded={() => api.listDocuments().then(setDocs).catch(() => {})}
      />
      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        lang={lang}
        onLang={setLang}
      />
    </div>
  );
}
