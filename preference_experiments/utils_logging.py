"""
Logging utility for experiment scripts. Provides a logger that writes to both a log file and stdout.
"""
import logging
import os

def get_logger(output_folder, log_level='INFO'):
    """
    Returns a logger that writes to both run.log in the output_folder and to stdout.
    Args:
        output_folder (str): Path to the experiment output folder.
        log_level (str): Logging level (e.g., 'INFO', 'DEBUG').
    Returns:
        logger (logging.Logger): Configured logger.
    """
    logger = logging.getLogger('experiment_logger')
    if getattr(logger, '_has_handlers', False):
        return logger
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # File handler
    os.makedirs(output_folder, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_folder, 'run.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # Stream (stdout) handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger._has_handlers = True
    return logger 