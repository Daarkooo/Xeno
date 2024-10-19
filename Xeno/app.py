from flask import Flask, request, jsonify
import subprocess
import sqlite3
import time
import threading

app = Flask(__name__)

# Set up SQLite database for logging
conn = sqlite3.connect('network_metrics.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS metrics (client_id TEXT, speed REAL, timestamp TEXT)''')
conn.commit()

def log_metrics():
    while True:
        for client_id in ["xeno-client1-1", "xeno-client2-1"]:  # Example client IDs
            speed = get_current_speed(client_id)  # Measure actual speed
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

            # Log metrics to the database
            c.execute("INSERT INTO metrics (client_id, speed, timestamp) VALUES (?, ?, ?)", (client_id, speed, timestamp))
            conn.commit()

        time.sleep(5)  # Wait for 5 seconds before logging again

def get_current_speed(client_id):
    # Use iperf to measure speed
    server_ip = "172.18.0.4"  #! Replace with actual server IP (service name in Docker Compose)
    command = f"docker exec {client_id} iperf -c {server_ip} -f m -t 5 -i 1"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    #! Print the entire output for debugging
    print(f"iperf output for {client_id}:\n{result.stdout}")
    print(f"iperf error output for {client_id}:\n{result.stderr}")

    # Extract speed from the result if available
    for line in result.stdout.splitlines():
        if "Mbits/sec" in line:
            try:
                speed = float(line.split()[6])  # Assuming the speed is at the 7th position in the line
                return speed
            except (IndexError, ValueError):
                print(f"Error extracting speed from line: {line}")
                return 0.0
 
    return 0.0  # Default value if speed measurement fails


# Start the logging thread
logging_thread = threading.Thread(target=log_metrics)
logging_thread.daemon = True
logging_thread.start()

@app.route('/limit_bandwidth', methods=['POST'])
def limit_bandwidth():
    client_id = request.json['client_id']
    limit = request.json['limit']  # in Mbps
    limit_kbps = limit * 1000  # Convert Mbps to kbps

    # Apply the bandwidth limit using Docker exec
    router_id = 'ae244a46c3ac'  #! Replace with your router's container ID
    command = f"tc qdisc add dev eth0 root tbf rate {limit_kbps}kbit burst 32kbit latency 400ms"
    subprocess.run(["docker", "exec", router_id, "bash", "-c", command])

    return jsonify({'message': f'Bandwidth limited for {client_id} to {limit} Mbps'})

@app.route('/metrics', methods=['GET'])
def get_metrics():
    # Retrieve the latest metrics from the database
    c.execute("SELECT * FROM metrics ORDER BY timestamp DESC LIMIT 10")  # Get the last 10 entries
    rows = c.fetchall()
    return jsonify({'metrics': rows})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
