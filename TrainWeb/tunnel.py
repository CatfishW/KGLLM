
import paramiko
import threading
import sys
import socket
import select

# Configuration
HOSTNAME = "vpn.agaii.org"
USERNAME = "lobin"
PASSWORD = "Clb1997521"
REMOTE_PORT = 32026
LOCAL_PORT = 32026
LOCAL_HOST = "127.0.0.1"

def handler(chan, host, port):
    sock = socket.socket()
    try:
        sock.connect((host, port))
    except Exception as e:
        print(f"Forwarding request to {host}:{port} failed: {e}")
        return

    print(f"Connected! Tunnel: remote:{REMOTE_PORT} -> local:{host}:{port}")
    
    while True:
        r, w, x = select.select([sock, chan], [], [])
        if sock in r:
            data = sock.recv(1024)
            if len(data) == 0:
                break
            chan.send(data)
        if chan in r:
            data = chan.recv(1024)
            if len(data) == 0:
                break
            sock.send(data)
    chan.close()
    sock.close()
    print(f"Tunnel connection closed")

def reverse_forward_tunnel(server_port, remote_host, remote_port, transport):
    transport.request_port_forward('', server_port)
    print(f"Now forwarding remote port {server_port} to {remote_host}:{remote_port}...")
    
    while True:
        chan = transport.accept(1000)
        if chan is None:
            continue
        thr = threading.Thread(target=handler, args=(chan, remote_host, remote_port))
        thr.daemon = True
        thr.start()

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    print(f"Connecting to {HOSTNAME}...")
    try:
        client.connect(HOSTNAME, username=USERNAME, password=PASSWORD)
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)

    print("Authentication successful.")
    
    try:
        reverse_forward_tunnel(REMOTE_PORT, LOCAL_HOST, LOCAL_PORT, client.get_transport())
    except KeyboardInterrupt:
        print("C-C! Terminating tunnel.")
        sys.exit(0)
    except Exception as e:
        print(f"Tunnel error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
