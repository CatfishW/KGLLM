import paramiko
import sys
import re

HOSTNAME = "vpn.agaii.org"
USERNAME = "lobin"
PASSWORD = "Clb1997521"
NGINX_CONF = "/www/server/panel/vhost/nginx/game.agaii.org.conf"
OLD_PORT = "72026"
NEW_PORT = "32026"

def run_sudo_command(client, command):
    print(f"Running: {command}")
    stdin, stdout, stderr = client.exec_command(f"sudo -S -p '' {command}")
    stdin.write(PASSWORD + "\n")
    stdin.flush()
    out = stdout.read().decode()
    err = stderr.read().decode()
    if out: print(out)
    if err: print(err)
    return out, err

def main():
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(HOSTNAME, username=USERNAME, password=PASSWORD)
        
        # Read current config
        print("Reading Nginx config...")
        conf_content, _ = run_sudo_command(client, f"cat {NGINX_CONF}")
        
        # Replace port
        if OLD_PORT in conf_content:
            print(f"Replacing port {OLD_PORT} with {NEW_PORT}...")
            new_conf = conf_content.replace(f"127.0.0.1:{OLD_PORT}", f"127.0.0.1:{NEW_PORT}")
            
            # Write temp file and upload
            with open("nginx_port_update.conf", "w") as f:
                f.write(new_conf)
            
            sftp = client.open_sftp()
            sftp.put("nginx_port_update.conf", "/tmp/nginx_port_update.conf")
            sftp.close()
            
            # Move and reload
            run_sudo_command(client, f"mv /tmp/nginx_port_update.conf {NGINX_CONF}")
            print("Config updated.")
            print("Reloading Nginx...")
            run_sudo_command(client, "systemctl reload nginx")
            print("Done.")
        else:
            print(f"Port {OLD_PORT} not found in config. Maybe already updated?")
            
        client.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
