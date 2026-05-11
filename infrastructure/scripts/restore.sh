#!/usr/bin/env bash
# Restore a backup folder produced by backup.sh
set -euo pipefail
src="${1:?usage: restore.sh BACKUPDIR}"
gunzip -c "$src/db.sql.gz" | \
    docker compose -f infrastructure/docker-compose.prod.yml exec -T db \
        psql -U "${POSTGRES_USER:-aria}" "${POSTGRES_DB:-aria}"
[ -f "$src/runs.tgz" ] && tar -xzf "$src/runs.tgz"
echo "restored from $src"
