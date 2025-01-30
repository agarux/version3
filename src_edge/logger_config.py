# logger_config.py
import logging
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Create the full path for the log file
log_file_path = os.path.join(script_dir, 'edge.log')

# Create a logger
logger = logging.getLogger("edge_logger")
logger.setLevel(logging.DEBUG)

# Create a console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create a file handler
fh = logging.FileHandler(log_file_path)
fh.setLevel(logging.DEBUG)

# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(ch)
logger.addHandler(fh)