"use client";
import { Activity, ChevronDown, ChevronUp, Loader2 } from "lucide-react";
import { useState } from "react";
import clsx from "clsx";
import { Lang, t } from "@/lib/i18n";

export type TraceItem = {
  id: string;
  name: string;
  status: "running" | "done" | "error";
  summary?: string;
};

export function ToolActivity({ lang, items }: { lang: Lang; items: TraceItem[] }) {
  const [open, setOpen] = useState(true);
  if (items.length === 0) return null;
  return (
    <div className="my-2 rounded-lg border border-border bg-panel/60 text-xs">
      <button
        className="w-full flex items-center gap-2 px-3 py-2 text-muted"
        onClick={() => setOpen((v) => !v)}
      >
        <Activity size={12} />
        <span className="flex-1 text-left">{t(lang, "toolTrace")} · {items.length}</span>
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>
      {open && (
        <ul className="px-3 pb-2 space-y-1">
          {items.map((it) => (
            <li key={it.id} className="flex items-center gap-2">
              <span className={clsx("w-1.5 h-1.5 rounded-full shrink-0",
                it.status === "running" && "bg-amber-400 animate-pulse",
                it.status === "done" && "bg-emerald-400",
                it.status === "error" && "bg-red-400")} />
              <span className="font-mono shrink-0 text-muted">{it.name}</span>
              <span className="truncate">{it.summary || ""}</span>
              {it.status === "running" && <Loader2 size={10} className="animate-spin" />}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
