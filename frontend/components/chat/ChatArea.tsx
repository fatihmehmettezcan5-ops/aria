"use client";
import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { ChatSession, Message } from "@/lib/types";
import { Lang, t } from "@/lib/i18n";
import { ChatMessage } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { TraceItem } from "./ToolActivity";
import { Source } from "./SourceCitation";
import { useUI } from "@/lib/store";

type StreamingState = { text: string; trace: TraceItem[] };

export function ChatArea({
  lang, session, onSessionUpdated, onUploaded,
}: {
  lang: Lang;
  session: ChatSession | null;
  onSessionUpdated: (s: ChatSession) => void;
  onUploaded: () => void;
}) {
  const { temperature } = useUI();
  const [messages, setMessages] = useState<Message[]>([]);
  const [streaming, setStreaming] = useState<StreamingState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setMessages([]); setStreaming(null); setError(null);
    if (!session) return;
    api.getSession(session.id).then((d) => setMessages(d.messages)).catch((e) => setError(String(e)));
  }, [session?.id]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, streaming?.text, streaming?.trace.length]);

  async function send(text: string) {
    if (!session) return;
    setError(null); setBusy(true);
    setMessages((m) => [...m, {
      id: `tmp-${Date.now()}`, role: "user", content: text, meta: {},
      created_at: new Date().toISOString(),
    }]);
    setStreaming({ text: "", trace: [] });

    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      await api.streamMessage(session.id, text, {
        signal: ctrl.signal,
        temperature,
        onEvent: ({ event, data }) => {
          if (event === "token") {
            setStreaming((s) => s ? { ...s, text: s.text + data.content } : s);
          } else if (event === "tool_start") {
            setStreaming((s) => s ? {
              ...s,
              trace: [...s.trace, { id: `${Date.now()}-${data.name}`, name: data.name, status: "running" }],
            } : s);
          } else if (event === "tool_end") {
            setStreaming((s) => s ? {
              ...s,
              trace: s.trace.map((t, i) =>
                i === s.trace.length - 1 ? { ...t, status: "done", summary: data.summary } : t),
            } : s);
          } else if (event === "done") {
            api.getSession(session.id).then((d) => {
              setMessages(d.messages);
              setStreaming(null);
              if (d.session.title !== session.title) onSessionUpdated(d.session);
            });
          } else if (event === "error") {
            setError(data.message || "stream error");
          }
        },
      });
    } catch (e: any) {
      if (e?.name !== "AbortError") setError(String(e?.message || e));
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  function stop() {
    abortRef.current?.abort();
    setStreaming(null); setBusy(false);
  }

  async function upload(file: File) {
    setError(null);
    try { await api.uploadDocument(file); onUploaded(); }
    catch (e: any) { setError(e?.message || "upload failed"); }
  }

  function deriveSources(m: Message): Source[] {
    const out: Source[] = [];
    const seen = new Set<string>();
    const calls = (m.meta?.tool_calls || []) as Array<{ name: string; arguments: any; result: any }>;
    for (const c of calls) {
      if (c.name === "knowledge_base") {
        for (const h of c.result?.hits || []) {
          const k = `kb:${h.document_id}:${h.chunk_index}`;
          if (seen.has(k)) continue; seen.add(k);
          out.push({ type: "kb", title: `${h.filename} #${h.chunk_index}` });
        }
      } else if (c.name === "web_search") {
        for (const r of c.result?.results || []) {
          if (!r.url || seen.has(`w:${r.url}`)) continue; seen.add(`w:${r.url}`);
          out.push({ type: "web", title: r.title || r.url, url: r.url });
        }
      } else if (c.name === "url_fetch" && c.result?.url) {
        if (!seen.has(`w:${c.result.url}`)) { seen.add(`w:${c.result.url}`);
          out.push({ type: "web", title: c.result.title || c.result.url, url: c.result.url });
        }
      }
    }
    return out;
  }

  if (!session) {
    return (
      <div className="flex-1 flex items-center justify-center text-center p-8">
        <div className="max-w-sm">
          <h2 className="text-xl font-semibold mb-2">{t(lang, "emptyTitle")}</h2>
          <p className="text-muted text-sm">{t(lang, "emptyBody")}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col min-w-0">
      <div className="border-b border-border px-4 py-2 text-sm font-medium truncate">{session.title}</div>
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4">
        <div className="max-w-3xl mx-auto py-2">
          {messages.length === 0 && !streaming && (
            <div className="text-center text-muted text-sm py-12">{t(lang, "emptyBody")}</div>
          )}
          {messages.map((m) =>
            m.role === "user" || m.role === "assistant" ? (
              <ChatMessage key={m.id} lang={lang} role={m.role} content={m.content}
                sources={m.role === "assistant" ? deriveSources(m) : undefined} />
            ) : null,
          )}
          {streaming && (
            <ChatMessage lang={lang} role="assistant" content={streaming.text} trace={streaming.trace} streaming />
          )}
          {error && (
            <div className="my-3 text-sm text-red-300 bg-red-500/10 border border-red-500/30 rounded-lg p-3">
              {t(lang, "errorPrefix")}: {error}
            </div>
          )}
        </div>
      </div>
      <ChatInput lang={lang} busy={busy} onSend={send} onUpload={upload} onStop={stop} />
    </div>
  );
}
