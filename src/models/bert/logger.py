import logging

def get_custom_logger(name: str, level=logging.INFO):
    # create formatter
    formatter = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d :: %(funcName)20s()} -  %(levelname)s - %(message)s')

    # create console handler and set level to debug
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    #logger.handlers = []
    logger.addHandler(handler)
    
    return logger
