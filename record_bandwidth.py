import psutil
import time
import csv

# Define the duration (in seconds) for which you want to monitor bandwidth
monitoring_duration = 60

# Start the monitoring process
start_time = time.time()

# Initialize variables to store bandwidth usage
total_bytes_sent = 0
total_bytes_received = 0

# Create a list to store bandwidth data
bandwidth_data = []

# Main monitoring loop
while time.time() - start_time < monitoring_duration:
    # Get the current network statistics
    net_io = psutil.net_io_counters()

    # Get the bytes sent and received since the last measurement
    bytes_sent = net_io.bytes_sent - total_bytes_sent
    bytes_received = net_io.bytes_recv - total_bytes_received

    # Update the total bytes sent and received
    total_bytes_sent = net_io.bytes_sent
    total_bytes_received = net_io.bytes_recv

    # Add the bandwidth data to the list
    bandwidth_data.append([bytes_sent, bytes_received])

    # Delay for a short interval (e.g., 1 second) before the next measurement
    time.sleep(1)

# Calculate average bandwidth per second
average_bandwidth_sent = total_bytes_sent / monitoring_duration
average_bandwidth_received = total_bytes_received / monitoring_duration

# Print the average bandwidth usage
print(f"Average bandwidth sent per second: {average_bandwidth_sent} bytes")
print(f"Average bandwidth received per second: {average_bandwidth_received} bytes")

# Save the bandwidth data to a CSV file
csv_filename = "bandwidth_data.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Bytes Sent", "Bytes Received"])  # Write header
    writer.writerows(bandwidth_data)  # Write data rows

print(f"Bandwidth data saved to {csv_filename}.")
