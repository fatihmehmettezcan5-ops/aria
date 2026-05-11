#!/usr/bin/env bash
# Postgres dump + checkpoint snapshot. Run from project root.
set -euo pipefail
ts=$(date -u +%Y%m%dT%H%M%SZ)
out="backups/$ts"
mkdir -p "$out"
docker compose -f infrastructure/docker-compose.prod.yml exec -T db \
    pg_dump -U "${POSTGRES_USER:-aria}" "${POSTGRES_DB:-aria}" | gzip > "$out/db.sql.gz"
[ -d runs ] && tar -czf "$out/runs.tgz" runs/
echo "backup → $out"
