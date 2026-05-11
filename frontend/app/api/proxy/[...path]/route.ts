// Same-origin proxy → backend, preserves streaming.
import { NextRequest } from "next/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const BACKEND =
  process.env.BACKEND_INTERNAL_URL ||
  process.env.NEXT_PUBLIC_API_BASE ||
  "http://backend:8000";

async function proxy(req: NextRequest, path: string[]) {
  const url = new URL(req.url);
  const target = `${BACKEND}/api/${path.join("/")}${url.search}`;
  const headers = new Headers(req.headers);
  headers.delete("host");
  headers.delete("connection");
  headers.delete("content-length");

  const init: RequestInit = {
    method: req.method,
    headers,
    redirect: "manual",
    // @ts-expect-error duplex required for streaming bodies
    duplex: "half",
  };
  if (!["GET", "HEAD"].includes(req.method)) init.body = req.body as any;

  const upstream = await fetch(target, init);
  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: upstream.headers,
  });
}

export async function GET(r: NextRequest, c: { params: { path: string[] } })   { return proxy(r, c.params.path); }
export async function POST(r: NextRequest, c: { params: { path: string[] } })  { return proxy(r, c.params.path); }
export async function PATCH(r: NextRequest, c: { params: { path: string[] } }) { return proxy(r, c.params.path); }
export async function PUT(r: NextRequest, c: { params: { path: string[] } })   { return proxy(r, c.params.path); }
export async function DELETE(r: NextRequest, c: { params: { path: string[] } }){ return proxy(r, c.params.path); }
