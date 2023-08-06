from os.path import dirname, join
from os import listdir
from pickle import dump, load
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

def parse_pcap(pcap_file_name:str) -> tuple:
    """Reads a pcap file and returns a tuple of the labels and one for the data.

    Args:
        pcap_file_name: The name of the pcap file.

    Returns:
        tuple: A tuple of two lists, one for the label and one for the data of each packet.
    """
    # Initialize an empty list for data and labels
    data = []
    data_labels = []
    # Load the pcap file using scapy
    packets = rdpcap(join(dirname(__file__), pcap_file_name))
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

def save_parsed_data_pickle(data: any, file_name: str) -> None:
  """Saves a list object to a file using the pickle library.

  Args:
    data (any): The variable to be saved.
    file_name (str): The name of the file to save the variable to.
  """
  # open the file in write binary mode
  with open(join(dirname(__file__), file_name), "wb") as f:
    # use dump to save the variable to the file
    dump(data, f)

def load_saved_data_pickle(file_name: str) -> any:
  """Loads a variable from a file using the pickle library.

    Args:
        file_name (str): The name of the pickle file.

    Returns:
        any: The variable in the pickle file.
  """
  # open the file in binary mode
  with open(join(dirname(__file__), file_name), 'rb') as f:
    # Call load method to deserialze
    return load(f)

def parse_and_combine_datasets():
    """Reads all the .pcap files within the "Datasets" directory and parses them individually 
    to get the data and it's labels into seperate lists. Then, it saves the lists to the "Parsed Datasets" directory
    using the pickle library"""
    data_labels = []
    data = []
    for pcap_file in listdir(join(dirname(__file__), "Datasets")):
        parsed_data_labels, parsed_data = parse_pcap('Datasets/' + pcap_file )
        data_labels += parsed_data_labels
        data += parsed_data
    save_parsed_data_pickle(data_labels ,'Parsed Datasets/dataset_data_labels') 
    save_parsed_data_pickle(data ,'Parsed Datasets/dataset_data')

def load_parsed_datasets() -> tuple:
    """Loads the data and it's accosiated lists from the "Parsed Datasets" directory using the pickle library.

    Returns:
        tuple: The lists, the labels and the data as a tuple.
    """
    return load_saved_data_pickle('Parsed Datasets/dataset_data_labels'), load_saved_data_pickle('Parsed Datasets/dataset_data')
    
if __name__ == '__main__':
    # Run the main function
    pass