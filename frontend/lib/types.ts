export type Lang = "en" | "tr";

export type ChatSession = {
  id: string;
  title: string;
  language: Lang;
  created_at: string;
  updated_at: string;
};

export type Message = {
  id: string;
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  meta: Record<string, any>;
  created_at: string;
};

export type DocumentInfo = {
  id: string;
  filename: string;
  file_type: string;
  file_size: number;
  created_at: string;
};

export type ModelInfo = {
  checkpoint: string;
  tokenizer: string;
  device: string;
  fallback_untrained: boolean;
  vocab_size: number;
  n_params: number;
  n_layers: number;
  d_model: number;
  n_heads: number;
  n_kv_heads: number | null;
  max_seq_len: number;
};

export type SSEEvent =
  | { event: "start"; data: { session_id: string } }
  | { event: "token"; data: { content: string } }
  | { event: "tool_start"; data: { name: string; arguments: any } }
  | { event: "tool_end"; data: { name: string; summary: string; result: any } }
  | { event: "done"; data: { final_text: string; tool_calls: any[] } }
  | { event: "error"; data: { message: string } };
