"use client";
import { X } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { ModelInfo } from "@/lib/types";
import { Lang, t } from "@/lib/i18n";
import { useUI } from "@/lib/store";

export function SettingsModal({
  open, onClose, lang, onLang,
}: {
  open: boolean;
  onClose: () => void;
  lang: Lang;
  onLang: (l: Lang) => void;
}) {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [tools, setTools] = useState<{ name: string; description: string }[]>([]);
  const { temperature, setTemperature } = useUI();

  useEffect(() => {
    if (!open) return;
    api.modelInfo().then(setInfo).catch(() => {});
    api.listTools().then(setTools).catch(() => {});
  }, [open]);

  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
      <div className="w-full max-w-md bg-panel border border-border rounded-2xl p-5 shadow-2xl">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-semibold">{t(lang, "settings")}</h2>
          <button onClick={onClose} className="text-muted hover:text-text"><X size={18} /></button>
        </div>

        <div className="space-y-4 text-sm">
          <div>
            <label className="text-xs text-muted">{t(lang, "language")}</label>
            <div className="mt-1 flex gap-2">
              {(["en", "tr"] as const).map((l) => (
                <button key={l} onClick={() => onLang(l)}
                  className={"px-3 py-1.5 rounded-lg text-sm border " +
                    (lang === l ? "border-accent bg-accent/15" : "border-border hover:bg-white/5")}>
                  {l === "en" ? "English" : "Türkçe"}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="text-xs text-muted">Temperature: {temperature.toFixed(2)}</label>
            <input type="range" min={0} max={1.5} step={0.05}
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full" />
          </div>

          {info && (
            <div className="text-xs text-muted space-y-1">
              <div><b>{t(lang, "model")}:</b> {(info.n_params / 1e6).toFixed(2)}M params · {info.n_layers}L · d={info.d_model}</div>
              <div>checkpoint: <code className="text-text">{info.checkpoint}</code></div>
              <div>device: <code className="text-text">{info.device}</code></div>
              {info.fallback_untrained && (
                <div className="text-amber-300">{t(lang, "untrainedWarning")}</div>
              )}
            </div>
          )}

          {tools.length > 0 && (
            <div className="text-xs text-muted">
              <b>{t(lang, "tools")}:</b> {tools.map((t) => t.name).join(", ")}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
