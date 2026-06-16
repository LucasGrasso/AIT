import logging


def get_logger(name):
    logger = logging.getLogger(name)
    if logger.handlers:  # ya configurado -> no dupliques
        return logger
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(ch)
    return logger
