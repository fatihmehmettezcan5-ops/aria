import type { ChatSession, DocumentInfo, Message, ModelInfo } from "./types";

const BASE = "/api/proxy";

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json())?.detail || detail; } catch {}
    throw new Error(`${r.status}: ${detail}`);
  }
  if (r.status === 204) return undefined as unknown as T;
  return r.json();
}

export const api = {
  // sessions
  listSessions: (): Promise<ChatSession[]> => fetch(`${BASE}/chat/sessions`).then(j),
  createSession: (language: "en" | "tr"): Promise<ChatSession> =>
    fetch(`${BASE}/chat/sessions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ language }),
    }).then(j),
  getSession: (id: string): Promise<{ session: ChatSession; messages: Message[] }> =>
    fetch(`${BASE}/chat/sessions/${id}`).then(j),
  updateSession: (id: string, patch: Partial<Pick<ChatSession, "title" | "language">>): Promise<ChatSession> =>
    fetch(`${BASE}/chat/sessions/${id}`, {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(patch),
    }).then(j),
  deleteSession: (id: string): Promise<void> =>
    fetch(`${BASE}/chat/sessions/${id}`, { method: "DELETE" }).then(j),

  // documents
  listDocuments: (): Promise<DocumentInfo[]> => fetch(`${BASE}/documents`).then(j),
  uploadDocument: async (file: File): Promise<DocumentInfo> => {
    const fd = new FormData();
    fd.append("file", file);
    return fetch(`${BASE}/documents/upload`, { method: "POST", body: fd }).then(j);
  },
  deleteDocument: (id: string): Promise<void> =>
    fetch(`${BASE}/documents/${id}`, { method: "DELETE" }).then(j),

  // model
  modelInfo: (): Promise<ModelInfo> => fetch(`${BASE}/model/info`).then(j),

  // tools
  listTools: (): Promise<{ name: string; description: string }[]> =>
    fetch(`${BASE}/tools`).then(j),

  // streaming chat
  async streamMessage(
    sessionId: string,
    content: string,
    opts: {
      signal?: AbortSignal;
      onEvent: (ev: { event: string; data: any }) => void;
      temperature?: number; top_p?: number; top_k?: number;
    },
  ) {
    const res = await fetch(`${BASE}/chat/sessions/${sessionId}/messages`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        content,
        temperature: opts.temperature, top_p: opts.top_p, top_k: opts.top_k,
      }),
      signal: opts.signal,
    });
    if (!res.ok || !res.body) {
      const text = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status}: ${text || res.statusText}`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const raw = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const lines = raw.split("\n");
        let event = "message", data = "";
        for (const ln of lines) {
          if (ln.startsWith("event:")) event = ln.slice(6).trim();
          else if (ln.startsWith("data:")) data += ln.slice(5).trim();
        }
        if (!data) continue;
        try { opts.onEvent({ event, data: JSON.parse(data) }); } catch {}
      }
    }
  },
};
