import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Aria — From-scratch AI Assistant",
  description: "Self-hostable AI assistant with our own model, RAG and tools.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="h-full">{children}</body>
    </html>
  );
}
