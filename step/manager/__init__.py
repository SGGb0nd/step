import logging

# Create a logger
logger = logging.getLogger(__name__)

# If the logger has handlers, remove them
if logger.hasHandlers():
    logger.handlers.clear()

# Set the logging level to INFO
logger.setLevel(logging.INFO)

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(ch)


def set_verbosity(level=3):
    """
    Set the verbosity level of the logger.

    Args:
        level (int): The verbosity level. 0 is warnings and errors, 1 is info, 2 is debug.
    """
    global verbosity
    verbosity = level
    if verbosity == 0:
        logger.setLevel(logging.WARNING)
    elif verbosity == 1:
        logger.setLevel(logging.INFO)
    elif verbosity >= 2:
        logger.setLevel(logging.DEBUG)
