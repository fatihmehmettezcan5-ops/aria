"use client";
import { FileText, Globe } from "lucide-react";
import { Lang, t } from "@/lib/i18n";

export type Source = {
  type: "kb" | "web";
  title: string;
  url?: string;
};

export function SourceCitation({ lang, sources }: { lang: Lang; sources: Source[] }) {
  if (!sources.length) return null;
  return (
    <div className="mt-3 border-t border-border pt-2">
      <div className="text-xs uppercase tracking-wider text-muted mb-1">{t(lang, "sources")}</div>
      <ol className="space-y-1 text-xs">
        {sources.map((s, i) => (
          <li key={i} className="flex items-start gap-2">
            <span className="text-muted">[{i + 1}]</span>
            {s.type === "web" ? <Globe size={12} className="mt-0.5" /> : <FileText size={12} className="mt-0.5" />}
            {s.url ? (
              <a href={s.url} target="_blank" rel="noreferrer" className="text-accent hover:underline truncate">
                {s.title || s.url}
              </a>
            ) : (
              <span className="truncate">{s.title}</span>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}
