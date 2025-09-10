
import logging
import os


def set_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        "%m/%d/%Y %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    return logger


def get_unused_gpu_ids() -> list:
    """Get the indices of the unused GPU devices according to nvidia-smi.
    """
    cmd = "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits"
    res = os.popen(cmd).read().strip().split('\n')
    data = [x.split(',') for x in res]
    device_usage = [(int(x[0]), int(x[1]) / int(x[2]) * 100) for x in data]
    to_print = [f"{idx}: {int(usage)}%" for idx, usage in device_usage]
    # print("GPU usage:", ', '.join(to_print))
    unused_devices = [x[0] for x in device_usage if int(x[1]) == 0]
    return unused_devices
