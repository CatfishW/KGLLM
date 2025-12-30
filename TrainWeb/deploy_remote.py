import paramiko
import os
import sys

HOSTNAME = "vpn.agaii.org"
USERNAME = "lobin"
PASSWORD = "Clb1997521"
REMOTE_DIR = "/mnt/data/Yanlai/TrainWebFrontend"
LOCAL_DIR = "/data/Yanlai/KGLLM/TrainWeb/frontend"
NGINX_CONF = "/www/server/panel/vhost/nginx/game.agaii.org.conf"

def run_sudo_command(client, command):
    print(f"Running: {command}")
    stdin, stdout, stderr = client.exec_command(f"sudo -S -p '' {command}")
    stdin.write(PASSWORD + "\n")
    stdin.flush()
    out = stdout.read().decode()
    err = stderr.read().decode()
    if out: print(out)
    if err: print(err)  # Could include sudo prompt or errors
    return out, err

def main():
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(HOSTNAME, username=USERNAME, password=PASSWORD)
        
        sftp = client.open_sftp()
        
        # 1. Create Remote Directory
        print(f"Creating remote directory: {REMOTE_DIR}")
        run_sudo_command(client, f"mkdir -p {REMOTE_DIR}")
        run_sudo_command(client, f"chown {USERNAME}:{USERNAME} {REMOTE_DIR}") # Ensure we can write to it via SFTP
        
        # 2. Upload Files
        print("Uploading frontend files...")
        for file in ['index.html', 'style.css', 'app.js']:
            local_path = os.path.join(LOCAL_DIR, file)
            remote_path = f"{REMOTE_DIR}/{file}"
            print(f"  {file} -> {remote_path}")
            sftp.put(local_path, remote_path)
            
        sftp.close()
        
        # 3. Update Nginx Config
        print("Updating Nginx Configuration...")
        # Read current config
        conf_content, _ = run_sudo_command(client, f"cat {NGINX_CONF}")
        
        if "/trainweb/" not in conf_content:
            # Construct the new block
            new_block = """
    # ============================================
    # TrainWeb - Training Monitor (port 72026 tunneled)
    # ============================================
    location /trainweb/ {
        alias /mnt/data/Yanlai/TrainWebFrontend/;
        index index.html;
        try_files $uri $uri/ /trainweb/index.html;
    }
    
    location /trainweb/api/ {
        rewrite ^/trainweb/api/(.*) /$1 break;
        
        proxy_pass http://127.0.0.1:72026;
        proxy_http_version 1.1;
        
        # WebSocket support
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
"""
            # Insert before the last closing brace (end of server block)
            # Find the last '}'
            last_brace_idx = conf_content.rfind('}')
            if last_brace_idx != -1:
                new_conf = conf_content[:last_brace_idx] + new_block + conf_content[last_brace_idx:]
                
                # Write to a temporary file locally
                with open("nginx_temp.conf", "w") as f:
                    f.write(new_conf)
                
                # Upload temp file
                sftp = client.open_sftp()
                sftp.put("nginx_temp.conf", "/tmp/nginx_temp.conf")
                sftp.close()
                
                # Move temp file to correct location with sudo
                run_sudo_command(client, f"mv /tmp/nginx_temp.conf {NGINX_CONF}")
                print("Nginx configuration updated.")
                
                # Reload Nginx
                print("Reloading Nginx...")
                run_sudo_command(client, "systemctl reload nginx") # Try systemctl first
                # Backup reload command just in case
                # run_sudo_command(client, "/etc/init.d/nginx reload") 
            else:
                print("Error: Could not find closing brace in nginx config.")
        else:
            print("Nginx configuration already contains /trainweb/ block. Skipping update.")
            
        client.close()
        print("Deployment Complete.")
        
    except Exception as e:
        print(f"Deployment Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
