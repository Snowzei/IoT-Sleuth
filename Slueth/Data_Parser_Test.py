from os.path import dirname, join
from scapy.all import *

# Scapy Zigbee Parameters
conf.dot15d4_protocol = "zigbee"

# Logged Device Information
device_types = {
    '192.168.100.118': 0,
    '1' : 0, 
    "192.168.1.132" : 0, # Phillips Hue Bridge
    '25117' : 1,  
    "3" : 1, # Phillips Hue Switch
    "192.168.2.3" : 2 # Amazon Echo Dot
}


def parse_pcap(pcap_file:str) -> tuple:
    """Reads a pcap file and returns a tuple of the labels and one for the data.

    Args:
        pcap_file: The name of the pcap file.

    Returns:
        tuple: A tuple of two lists, one for the label and one for the data of each packet.
    """
    # Initialize an empty list for data and labels
    data = []
    data_labels = []
    # Load the pcap file using scapy
    packets = rdpcap(join(dirname(__file__), pcap_file))
    # Loop through the packets in the pcap file
    for packet in packets:
        # Check for IP layers
        if scapy.layers.inet.IP in packet:
            try: 
                # Set the label based on known device addresses
                data_labels.append(device_types[str(packet[scapy.layers.inet.IP].src)])
                # Set the data
                data.append(bytes_hex(packet))
            # Ignore any packet from unknown addresses
            except: pass
        # Check for Zigbee layers
        elif scapy.layers.zigbee.ZigbeeNWK in packet:
            try: 
                # Set the label based on known device addresses
                data_labels.append(device_types[str(packet[scapy.layers.zigbee.ZigbeeNWK].source)])
                # Set the data
                data.append(bytes_hex(packet))
            # Ignore any packet from unknown addresses
            except: pass
    return data_labels, data

def main():
    """The main function that runs the script."""
    # data_labels = []
    # data = []
    data_labels, data = parse_pcap('Datasets\Philips-Hue-Bridge.pcap')
    print(data_labels)
    print(data)
    print("-------------------------------------------")
    

if __name__ == '__main__':
    # Run the main function
    main()