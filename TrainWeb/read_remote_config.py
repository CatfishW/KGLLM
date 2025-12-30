import paramiko
import sys

hostname = "vpn.agaii.org"
username = "lobin"
password = "Clb1997521"
nginx_config_path = "/www/server/panel/vhost/nginx/game.agaii.org.conf"

try:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, username=username, password=password)
    
    # Use sudo to cat the file
    command = f"sudo -S cat {nginx_config_path}"
    stdin, stdout, stderr = client.exec_command(command)
    stdin.write(password + "\n")
    stdin.flush()
    
    content = stdout.read().decode()
    error = stderr.read().decode()
    
    if content:
        print("=== Nginx Config Content ===")
        print(content)
    if error:
        # Sudo prompt might appear in stderr but if it works we ignore it
        if "Password:" not in error or len(error.splitlines()) > 1:
             print("=== Error ===")
             print(error)
        
    client.close()
except Exception as e:
    print(f"Connection failed: {e}")
