#!/bin/bash
# Insert paperreader nginx config

CONF_FILE="/www/server/panel/vhost/nginx/game.agaii.org.conf"
NEW_CONFIG="/tmp/paperreader_nginx.txt"

# Create backup
cp "$CONF_FILE" "$CONF_FILE.bak"

# Remove last line (closing brace), append new config, add closing brace back
head -n -1 "$CONF_FILE" > /tmp/nginx_temp.conf
cat "$NEW_CONFIG" >> /tmp/nginx_temp.conf
echo "}" >> /tmp/nginx_temp.conf

# Replace original
cp /tmp/nginx_temp.conf "$CONF_FILE"

# Test nginx config
nginx -t

echo "Config updated. Run 'nginx -s reload' to apply."
