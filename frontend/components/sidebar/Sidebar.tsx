"use client";
import { Plus, Trash2, MessageSquare, FileText, Settings as SettingsIcon, Cpu } from "lucide-react";
import clsx from "clsx";
import type { ChatSession, DocumentInfo, ModelInfo } from "@/lib/types";
import { Lang, t } from "@/lib/i18n";

export function Sidebar({
  lang, sessions, documents, activeId, modelInfo,
  onSelect, onNew, onDelete, onDeleteDoc, onOpenSettings,
}: {
  lang: Lang;
  sessions: ChatSession[];
  documents: DocumentInfo[];
  activeId: string | null;
  modelInfo: ModelInfo | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onDeleteDoc: (id: string) => void;
  onOpenSettings: () => void;
}) {
  return (
    <aside className="w-72 shrink-0 h-full border-r border-border bg-panel flex flex-col">
      <div className="p-3 border-b border-border">
        <button onClick={onNew}
          className="w-full flex items-center gap-2 justify-center bg-accent text-white rounded-lg py-2 font-medium">
          <Plus size={16} /> {t(lang, "newChat")}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        <div className="px-2 pt-1 pb-2 text-xs uppercase tracking-wider text-muted">{t(lang, "history")}</div>
        {sessions.length === 0 && <div className="px-2 py-3 text-sm text-muted">—</div>}
        <ul className="space-y-1">
          {sessions.map((c) => (
            <li key={c.id}
              className={clsx(
                "group flex items-center gap-2 rounded-lg px-2 py-2 cursor-pointer",
                activeId === c.id ? "bg-accent/15" : "hover:bg-white/5",
              )}
              onClick={() => onSelect(c.id)}>
              <MessageSquare size={14} className="shrink-0 text-muted" />
              <span className="flex-1 text-sm truncate">{c.title}</span>
              <button title={t(lang, "delete")}
                className="opacity-0 group-hover:opacity-100 text-muted hover:text-red-400"
                onClick={(e) => { e.stopPropagation(); if (confirm(t(lang, "confirmDelete"))) onDelete(c.id); }}>
                <Trash2 size={14} />
              </button>
            </li>
          ))}
        </ul>

        <div className="px-2 pt-5 pb-2 text-xs uppercase tracking-wider text-muted">{t(lang, "documents")}</div>
        {documents.length === 0 ? (
          <div className="px-2 py-1 text-sm text-muted">{t(lang, "noDocs")}</div>
        ) : (
          <ul className="space-y-1">
            {documents.map((d) => (
              <li key={d.id} className="group flex items-center gap-2 rounded-lg px-2 py-1.5 hover:bg-white/5"
                title={d.filename}>
                <FileText size={14} className="shrink-0 text-muted" />
                <span className="flex-1 text-xs truncate">{d.filename}</span>
                <button className="opacity-0 group-hover:opacity-100 text-muted hover:text-red-400"
                  onClick={() => { if (confirm(t(lang, "confirmDeleteDoc"))) onDeleteDoc(d.id); }}>
                  <Trash2 size={12} />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="border-t border-border p-3 flex items-center gap-2 text-xs">
        <Cpu size={14} className="text-muted" />
        <div className="flex-1 truncate text-muted" title={modelInfo?.checkpoint || ""}>
          {modelInfo
            ? `${(modelInfo.n_params / 1e6).toFixed(1)}M · ${modelInfo.device}`
            : "…"}
        </div>
        <button onClick={onOpenSettings} title={t(lang, "settings")} className="text-muted hover:text-text">
          <SettingsIcon size={16} />
        </button>
      </div>
    </aside>
  );
}
