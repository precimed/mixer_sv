import logging, sys

def get_logger():
    """
    Returns a logger configured to output to stdout with DEBUG level.
    """
    logger = logging.getLogger()
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def setup_logger(filename, verbose):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(filename)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(console_handler)

def is_verbose():
    """
    Returns True if the logging level is set to DEBUG, otherwise False.
    """
    return logging.INFO >= logging.getLogger().getEffectiveLevel()
