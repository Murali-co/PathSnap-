# Two-laptop demo

1. Connect both devices to the same network that permits device-to-device traffic.
2. Start Laptop 1 backend using the backend guide; verify `/health` is ready.
3. Find its LAN IPv4 address. `192.168.1.10` is an example placeholder, not a detected address.
4. Set Laptop 2's `BACKEND_URL=http://<LAPTOP_1_LAN_IP>:8000`.
5. Run its `scripts/check_connection.py`, then its frontend start script.
6. Open http://localhost:8501; run each of the ten configured development trips, inspect actual decoded route/metrics and repeat a previously completed trip.
7. Show corresponding live request/job/stage/duration lines on Laptop 1. Preserve screenshots/logs as physical demonstration evidence.
8. Repeat in evaluation mode only after organiser data is authorised and the model/config are frozen. Development IDs must never be relabelled as reserved IDs.

Troubleshoot IP/port, same subnet, VPN/proxy, firewall and Wi-Fi client isolation. If an inbound firewall rule is needed, limit it to the backend port and trusted local network; never disable the entire firewall.

Automated localhost testing verifies separate application processes and actual HTTP requests. It does not verify a physical LAN connection, friend-laptop dependencies or Windows execution.
