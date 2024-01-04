import logging
import sys

logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)

format_str = "%(asctime)s [%(module)s] %(funcName)s %(lineno)s: %(levelname)-8s [%(process)d] %(message)s"
formatter = logging.Formatter(format_str)
stream_hanlder = logging.StreamHandler(sys.stdout)
stream_hanlder.setFormatter(formatter)
logger.addHandler(stream_hanlder)
