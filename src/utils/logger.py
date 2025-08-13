# logger.py
import logging
import sys
import json
from pythonjsonlogger import jsonlogger  # pip install python-json-logger

def setup_logger(name=__name__, level=logging.INFO, logfile=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s %(job_id)s %(run_id)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # add context defaults
    extra = {"job_id": "-", "run_id": "-"}
    logger = logging.LoggerAdapter(logger, extra)
    return logger

