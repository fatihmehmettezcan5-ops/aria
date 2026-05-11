"use client";
import { Paperclip, Send, Square } from "lucide-react";
import { useRef, useState } from "react";
import { Lang, t } from "@/lib/i18n";

export function ChatInput({
  lang, busy, onSend, onUpload, onStop,
}: {
  lang: Lang;
  busy: boolean;
  onSend: (text: string) => void;
  onUpload: (file: File) => void;
  onStop: () => void;
}) {
  const [text, setText] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  function submit(e?: React.FormEvent) {
    e?.preventDefault();
    const v = text.trim();
    if (!v || busy) return;
    onSend(v);
    setText("");
  }

  return (
    <form onSubmit={submit} className="border-t border-border bg-panel/60 backdrop-blur p-3 sticky bottom-0">
      <div className="max-w-3xl mx-auto">
        <div className="flex items-end gap-2 bg-bg border border-border rounded-2xl px-2 py-2 focus-within:border-accent">
          <button type="button" title={t(lang, "upload")}
            onClick={() => fileRef.current?.click()}
            className="p-2 text-muted hover:text-text">
            <Paperclip size={18} />
          </button>
          <input ref={fileRef} type="file"
            accept=".pdf,.txt,.md,.markdown,.docx,.csv,.log,.py,.js,.ts,.json"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) onUpload(f);
              if (fileRef.current) fileRef.current.value = "";
            }} />
          <textarea rows={1} value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
            }}
            placeholder={t(lang, "placeholder")}
            className="flex-1 bg-transparent outline-none resize-none py-2 max-h-40 text-sm" />
          {busy ? (
            <button type="button" onClick={onStop}
              className="p-2 rounded-lg bg-red-500/20 text-red-300 hover:bg-red-500/30"
              title={t(lang, "stop")}>
              <Square size={16} />
            </button>
          ) : (
            <button type="submit" disabled={!text.trim()}
              className="p-2 rounded-lg bg-accent text-white disabled:opacity-40"
              title={t(lang, "send")}>
              <Send size={16} />
            </button>
          )}
        </div>
      </div>
    </form>
  );
}
