# API Reference

Base URL: `http://localhost:8000` (or your deployed domain).
OpenAPI/Swagger: `GET /docs`.

If `ARIA_API_KEY` is set, every `/api/*` call must include
`X-API-Key: <key>` (the frontend proxy forwards browser cookies but you
can also set it as a static header for service-to-service calls).

## Health

| Method | Path     | Returns |
|--|--|--|
| GET    | /health  | `{"status":"ok"}` |
| GET    | /ready   | `{"status":"ready"}` |

## Sessions

| Method | Path                                | Body / Notes |
|--|--|--|
| GET    | /api/chat/sessions                  | List sessions |
| POST   | /api/chat/sessions                  | `{title?, language: "en"\|"tr"}` |
| GET    | /api/chat/sessions/{id}             | Session + messages |
| PATCH  | /api/chat/sessions/{id}             | `{title?, language?}` |
| DELETE | /api/chat/sessions/{id}             | 204 |
| POST   | /api/chat/sessions/{id}/messages    | `{content, temperature?, top_p?, top_k?}` → SSE stream |

### SSE event types

| event       | data |
|--|--|
| `start`     | `{"session_id": "<uuid>"}` |
| `token`     | `{"content": "string"}` (one or more chars at a time) |
| `tool_start`| `{"name": "...", "arguments": {...}}` |
| `tool_end`  | `{"name": "...", "summary": "...", "result": {...}}` |
| `done`      | `{"final_text": "...", "tool_calls": [...]}` |
| `error`     | `{"message": "..."}` |

## Documents

| Method | Path                  | Body / Notes |
|--|--|--|
| GET    | /api/documents        | List uploaded docs |
| POST   | /api/documents/upload | `multipart/form-data` with `file` |
| GET    | /api/documents/{id}   | Doc + chunks |
| DELETE | /api/documents/{id}   | 204 |
| POST   | /api/documents/query  | `{query, top_k?}` → list of hits |

## Tools

| Method | Path                  | Body |
|--|--|--|
| GET    | /api/tools            | Tool schemas |
| POST   | /api/tools/execute    | `{name, arguments}` |

## Model

| Method | Path                  | Body |
|--|--|--|
| GET    | /api/model/info       | Model + tokenizer + device info |
| POST   | /api/model/generate   | `{prompt, max_new_tokens?, temperature?, top_p?, top_k?}` |

## Examples

```bash
# Create a session
sid=$(curl -s -X POST http://localhost:8000/api/chat/sessions \
   -H "content-type: application/json" -d '{"language":"en"}' | jq -r .id)

# Send a streaming message
curl -N -X POST "http://localhost:8000/api/chat/sessions/$sid/messages" \
  -H "content-type: application/json" \
  -d '{"content":"What is 23 * 47?"}'

# Upload a document
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@./report.pdf"
```
