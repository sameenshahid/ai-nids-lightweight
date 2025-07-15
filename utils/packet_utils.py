# utils/packet_utils.py

# packet_utils.py
import pyshark
import asyncio

def get_capture(mode, source):
    """
    Returns a pyshark LiveCapture object with proper asyncio setup for threads.
    """
    # Ensure the thread has an event loop (Python 3.10+ compatibility)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    interface = source or "eth0"  # Adjust if you're on Windows
    return pyshark.LiveCapture(interface=interface)


def packet_to_text(packet):
    try:
        return f"{packet.highest_layer} {packet.transport_layer or ''} {packet.length} {packet.ip.src} -> {packet.ip.dst}"
    except AttributeError:
        return "UNKNOWN PACKET"

label_map = {
    "0": "BENIGN",
    "1": "DoS Hulk",
    "2": "DoS GoldenEye",
    "3": "DoS slowloris",
    "4": "DoS Slowhttptest",
    "5": "DDoS",
    "6": "PortScan",
    "7": "Brute Force -XSS",
    "8": "Brute Force -Web",
    "9": "Brute Force -Sql Injection",
    "10": "Bot",
    "11": "Infiltration",
    "12": "Heartbleed",
    "13": "FTP-Patator",
    "14": "SSH-Patator"
}
