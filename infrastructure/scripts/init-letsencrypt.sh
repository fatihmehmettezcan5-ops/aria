#!/usr/bin/env bash
# Bootstrap Let's Encrypt certs. Run on the VPS after `docker compose up -d nginx`.
set -euo pipefail

DOMAIN="${DOMAIN:-example.com}"
EMAIL="${EMAIL:-admin@${DOMAIN}}"
STAGING="${STAGING:-0}"

CONF=./nginx/certbot/conf
mkdir -p ./nginx/certbot/www "$CONF"

if [ ! -e "$CONF/options-ssl-nginx.conf" ]; then
  curl -fsSL https://raw.githubusercontent.com/certbot/certbot/main/certbot-nginx/certbot_nginx/_internal/tls_configs/options-ssl-nginx.conf > "$CONF/options-ssl-nginx.conf"
fi
if [ ! -e "$CONF/ssl-dhparams.pem" ]; then
  openssl dhparam -out "$CONF/ssl-dhparams.pem" 2048
fi

STAGING_FLAG=""
[ "$STAGING" = "1" ] && STAGING_FLAG="--staging"

docker compose -f docker-compose.prod.yml run --rm --entrypoint "\
  certbot certonly --webroot -w /var/www/certbot \
    --email ${EMAIL} --agree-tos --no-eff-email \
    ${STAGING_FLAG} -d ${DOMAIN}" certbot

docker compose -f docker-compose.prod.yml exec nginx nginx -s reload || true
echo "Done. Edit nginx/conf.d/aria.conf to enable the HTTPS block, then reload nginx."
