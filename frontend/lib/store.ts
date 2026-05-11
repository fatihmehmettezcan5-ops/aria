import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { Lang } from "./types";

type UI = {
  lang: Lang;
  setLang: (l: Lang) => void;
  temperature: number;
  setTemperature: (n: number) => void;
};

export const useUI = create<UI>()(
  persist(
    (set) => ({
      lang: "en",
      setLang: (l) => set({ lang: l }),
      temperature: 0.8,
      setTemperature: (n) => set({ temperature: n }),
    }),
    { name: "aria-ui" },
  ),
);
