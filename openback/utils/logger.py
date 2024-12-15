import logging


def set_logger(log_file='../sample.log', level='info'):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()

    for handler in logger.handlers:
        print(f'remove handler: {handler}')
        logger.removeHandler(handler)
    
    if level == 'debug':
        log_lebel = logging.DEBUG
    elif level == 'info':
        log_lebel = logging.INFO
    logger.setLevel(log_lebel)

    if not logger.hasHandlers():
        logger.addHandler(stream_handler)
    logger.addHandler(file_handler)


class AverageManager:
    def __init__(self):
        self.value = 0.0
        self.n = 1
    
    def __iadd__(self, other: float):
        self.value = other
        self.n += 1
        return self

    @property
    def mean(self) -> float:
        return self.value / self.n
