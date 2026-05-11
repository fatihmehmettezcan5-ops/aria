"use client";
import { Bot, User } from "lucide-react";
import clsx from "clsx";
import { Markdown } from "../common/Markdown";
import { ToolActivity, TraceItem } from "./ToolActivity";
import { SourceCitation, Source } from "./SourceCitation";
import { Lang } from "@/lib/i18n";

export function ChatMessage({
  lang, role, content, trace, sources, streaming,
}: {
  lang: Lang;
  role: "user" | "assistant";
  content: string;
  trace?: TraceItem[];
  sources?: Source[];
  streaming?: boolean;
}) {
  const isUser = role === "user";
  return (
    <div className={clsx("flex gap-3 py-4", isUser ? "justify-end" : "justify-start")}>
      {!isUser && (
        <div className="shrink-0 w-8 h-8 rounded-full bg-accent/20 flex items-center justify-center text-accent">
          <Bot size={16} />
        </div>
      )}
      <div
        className={clsx(
          "max-w-[80%] rounded-2xl px-4 py-3 text-sm",
          isUser
            ? "bg-accent text-white rounded-br-md"
            : "bg-panel border border-border rounded-bl-md",
        )}
      >
        {isUser ? (
          <div className="whitespace-pre-wrap">{content}</div>
        ) : (
          <>
            {trace && <ToolActivity lang={lang} items={trace} />}
            {content ? (
              <Markdown>{content + (streaming ? " ▍" : "")}</Markdown>
            ) : streaming ? (
              <div className="text-muted text-xs">▍</div>
            ) : null}
            {sources && <SourceCitation lang={lang} sources={sources} />}
          </>
        )}
      </div>
      {isUser && (
        <div className="shrink-0 w-8 h-8 rounded-full bg-white/10 flex items-center justify-center text-muted">
          <User size={16} />
        </div>
      )}
    </div>
  );
}
