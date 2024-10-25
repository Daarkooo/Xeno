from flask import Flask, request, jsonify
import subprocess
import sqlite3
import time
import threading
from datetime import datetime

app = Flask(__name__)

# Set up SQLite database for logging
conn = sqlite3.connect('network_metrics.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS metrics (client_id TEXT, speed REAL, timestamp TEXT)''')
conn.commit()

def log_metrics():
    while True:
        for client_id in ["xeno-client1", "xeno-client2"]:  # Example client IDs
            metrics = get_current_speed(client_id)  # Get metrics
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            speed = metrics['bandwidth']
            c.execute("INSERT INTO metrics (client_id, speed, timestamp) VALUES (?, ?, ?)", (client_id, speed, timestamp))
            conn.commit()
            print(f"{timestamp} - Bandwidth for {client_id}: {speed} Mbps")

        time.sleep(5)

def get_current_speed(client_id):
    server_ip = "xeno-server"  # Use the service name
    command = f"docker exec {client_id} iperf -c {server_ip} -f m -t 5 -i 1"
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if "Mbits/sec" in line:
                speed = float(line.split()[6])  # Adjust based on actual output format
                return {
                    "client_id": client_id,
                    "bandwidth": speed,
                    "unit": "Mbps",
                    "timestamp": datetime.now().isoformat()
                }
    except Exception as e:
        print(f"Error measuring speed for {client_id}: {e}")

    return {
        "client_id": client_id,
        "bandwidth": 0.0,
        "unit": "Mbps",
        "timestamp": datetime.now().isoformat()
    }

# Start logging in a background thread
logging_thread = threading.Thread(target=log_metrics)
logging_thread.daemon = True
logging_thread.start()

@app.route('/limit_bandwidth', methods=['POST'])
def limit_bandwidth():
    # Validate incoming data
    if not request.json or 'client_id' not in request.json or 'limit' not in request.json:
        return jsonify({'error': 'Invalid input. Please provide client_id and limit.'}), 400

    client_id = request.json['client_id']
    limit = request.json['limit']  # in Mbps

    # Check if the limit is a positive number
    if not isinstance(limit, (int, float)) or limit <= 0:
        return jsonify({'error': 'Limit must be a positive number.'}), 400

    limit_kbps = limit * 1000  # Convert Mbps to kbps

    # Router's container ID
    router_id = 'ae244a46c3ac'  # Replace with your router's container ID

    # Cleanup: remove existing bandwidth limit if present
    cleanup_command = "tc qdisc del dev eth0 root"  # Consider modifying 'eth0' if your interface name is different
    cleanup_result = subprocess.run(
        ["docker", "exec", router_id, "bash", "-c", cleanup_command],
        capture_output=True,
        text=True
    )

    if cleanup_result.returncode != 0:
        print(f"Cleanup command failed: {cleanup_result.stderr}")
        return jsonify({'error': 'Failed to cleanup existing bandwidth limit. Check logs.'}), 500

    # Apply the new bandwidth limit
    command = f"tc qdisc add dev eth0 root tbf rate {limit_kbps}kbit burst 32kbit latency 400ms"
    result = subprocess.run(
        ["docker", "exec", router_id, "bash", "-c", command],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Bandwidth limit command failed: {result.stderr}")
        return jsonify({'error': 'Failed to apply bandwidth limit. Check router logs.'}), 500

    return jsonify({'message': f'Bandwidth limited for {client_id} to {limit} Mbps'})

@app.route('/metrics', methods=['GET'])
def get_metrics():
    # Retrieve the latest metrics from the database
    c.execute("SELECT * FROM metrics ORDER BY timestamp DESC LIMIT 10")  # Get the last 10 entries
    rows = c.fetchall()
    return jsonify({'metrics': rows})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
