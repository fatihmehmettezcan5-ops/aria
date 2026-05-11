# Deployment Guide

## Local development

```bash
cp .env.example .env

# 1. Train a tiny model (~3 min on CPU) so the API has something to serve
make smoke-train

# 2. Bring up the stack
make up
# open http://localhost:3000
```

The dev compose mounts the source directories so Python and Next hot-reload
as you edit. The DB persists in a named volume (`db_data`).

### Alembic

The backend container runs `alembic upgrade head` on start, so new
migrations apply automatically. To create one:

```bash
docker compose -f infrastructure/docker-compose.yml exec backend \
  bash -c "cd backend && alembic revision --autogenerate -m 'description'"
```

## Production on Ubuntu VPS

### Server requirements

| Workload | RAM | Disk | CPU/GPU |
|--|--|--|--|
| Inference, `tiny` model | 2GB | 5GB | 2 vCPU is enough |
| Inference, `small` model | 4GB | 10GB | 2 vCPU OK; GPU >2× faster |
| Training | 16GB+ | 50GB+ | NVIDIA GPU strongly recommended |

### One-time server setup

```bash
# As a sudoer:
sudo apt update && sudo apt install -y curl ca-certificates
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker

sudo ufw allow OpenSSH
sudo ufw allow 80
sudo ufw allow 443
sudo ufw --force enable
```

### Deploy

```bash
git clone https://github.com/<you>/aria.git
cd aria
cp .env.example .env
# Edit .env — set strong APP_SECRET, POSTGRES_PASSWORD, ARIA_API_KEY,
# ALLOWED_ORIGINS=https://yourdomain.com, PUBLIC_BASE_URL=...

# Bring up backend + DB + frontend (still HTTP at this point)
docker compose -f infrastructure/docker-compose.prod.yml up -d
```

If you want the bundled smoke model on the server, copy `runs/smoke/`
across with `scp` or `rsync`. For real deployments, copy your trained
`runs/finetune/best.pt` + `runs/tokenizer/tokenizer.json`.

### HTTPS with Let's Encrypt

```bash
# 1) Edit infrastructure/nginx/conf.d/aria.conf — replace example.com
# 2) Bring up nginx alone first (HTTP only)
docker compose -f infrastructure/docker-compose.prod.yml up -d nginx

# 3) Run the bootstrap script
DOMAIN=yourdomain.com EMAIL=you@yourdomain.com \
  bash infrastructure/scripts/init-letsencrypt.sh

# 4) Uncomment the HTTPS server block in nginx/conf.d/aria.conf
#    and reload nginx
docker compose -f infrastructure/docker-compose.prod.yml exec nginx nginx -s reload
```

The certbot container runs in the background and renews every 12h.

### Backups

```bash
bash infrastructure/scripts/backup.sh        # → backups/<timestamp>/
bash infrastructure/scripts/restore.sh backups/20250101T000000Z
```

Schedule via cron:
```
30 3 * * * cd /home/aria/aria && bash infrastructure/scripts/backup.sh
```

### Updates

```bash
git pull
docker compose -f infrastructure/docker-compose.prod.yml build --pull
docker compose -f infrastructure/docker-compose.prod.yml up -d
```

If you trained a new model, replace `runs/finetune/best.pt` and restart
`backend` only (`docker compose ... restart backend`).

### Monitoring

- Backend logs are JSON: `docker compose ... logs -f backend | jq`.
- TensorBoard logs (training only) live in `runs/<run>/tb/`.
- Health: `curl https://yourdomain.com/health`.
