import rldev

import logging
import os
import time
import pathlib
from pathlib import Path
from omegaconf import DictConfig
import hydra
import tqdm
from typing import List
import re

import torch

from apep.trainer.builder import build_trainer

logging.getLogger('numba').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'apep/config')
CONFIG_NAME = 'default_training'


LOGGING_LEVEL_MAP = {
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG,
}


class PathKeywordMatch(logging.Filter):
    """
    This implements simple logging.Filter, by running a regexp match on the path of the log record path name.
    """

    def __init__(self, regexp: str = ''):
        """
        :param regexp: Regexp used for filtering.
        """
        self.regexp = regexp
        super().__init__()

    def filter(self, log_record: logging.LogRecord) -> bool:
        """
        Determine if the specified record is to be logged.
        :param log_record: Logging.LogRecord, the record to emit.
        :return: Is the specified record to be logged? False for no, True for yes.
        """
        return re.match(self.regexp, log_record.pathname) is not None


class TqdmLoggingHandler(logging.Handler):
    """
    Log consistently when using the tqdm progress bar.
    From https://stackoverflow.com/questions/38543506/
    change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit
    """

    def __init__(self, level: int = logging.NOTSET) -> None:
        """
        Constructor.
        :param level: A log level.
        """
        super().__init__(level)

    def emit(self, record: logging.LogRecord) -> None:
        """
        Consistently emit the specified logging record.
        :param record: Logging.LogRecord, the record to emit.
        """
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)



class LogHandlerConfig:
    """This is a simple config struct for log handles. Used by configure_logger method."""

    def __init__(self, level: str, path= None, filter_regexp: str = '') -> None:
        """
        :param level: logging level represented as string, E.g. 'info'.
        :param path: Path to where to store the log. Leave as None for logging to console.
        :param filter_regexp: Regexp defining the filter. This will be used in a PathKeywordMatch object.
        """
        self.level = level
        self.path = path
        self.filter_regexp = filter_regexp

        if self.path is not None:
            # Create the directory if not present already.
            _dir = os.path.dirname(self.path)
            if not os.path.exists(_dir):
                os.makedirs(_dir, exist_ok=True)



def configure_logger(
    handler_configs: List[LogHandlerConfig],
    format_str: str = '%(asctime)s %(levelname)-2s {%(pathname)s:%(lineno)d}  %(message)s',
) -> logging.Logger:
    """
    Configures the python default logger.
    :param handler_configs: List of LogHandlerConfig objects specifying the logger handlers.
    :param format_str: Formats the log events.
    :return: A logger.
    """
    # Purge old handlers.
    logger = logging.getLogger()
    for old_handler in logger.handlers:
        logger.removeHandler(old_handler)

    # Setup streaming handler by default
    for config in handler_configs:
        if not config.path:
            handler = TqdmLoggingHandler()
        else:
            handler = logging.FileHandler(config.path)  # type: ignore
        handler.setLevel(LOGGING_LEVEL_MAP[config.level])
        handler.setFormatter(logging.Formatter(format_str))
        handler.addFilter(PathKeywordMatch(config.filter_regexp))
        logger.addHandler(handler)

    return logger

def build_logger(cfg: DictConfig) -> logging.Logger:
    """
    Setup the standard logger, always log to sys.stdout and optionally log to disk.
    :param cfg: Input dict config.
    :return: Logger with associated handlers.
    """
    handler_configs = [LogHandlerConfig(level=cfg.logger_level)]

    if cfg.output_dir is not None:
        path = str(Path(cfg.output_dir) / 'log.txt')
        handler_configs.append(LogHandlerConfig(level=cfg.logger_level, path=path))

    format_string = (
        '%(asctime)s %(levelname)-2s {%(pathname)s:%(lineno)d}  %(message)s'
        if not cfg.logger_format_string
        else cfg.logger_format_string
    )
    logger = configure_logger(handler_configs, format_str=format_string)

    # Disable logger if it's not main process. This is useful when the trainer uses multiple processes in the DDP mode.
    # if cfg.gpu:
    logger.disabled = int(os.environ.get('LOCAL_RANK', 0)) != 0

    logger.setLevel(level=LOGGING_LEVEL_MAP[cfg.logger_level])

    return logger






@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    """
    Main entrypoint for training/validation experiments.
    :param cfg: omegaconf dictionary
    """
    # Fix random seed
    # pl.seed_everything(cfg.seed, workers=True)
    rldev.setup_seed(cfg.seed)

    # Configure logger
    build_logger(cfg)

    logger.info('Building experiment folders...')
    main_exp_folder = pathlib.Path(cfg.output_dir).expanduser()
    logger.info(f'Experimental folder: {main_exp_folder}')
    main_exp_folder.mkdir(parents=True, exist_ok=True)

    trainer = build_trainer(cfg)
    if cfg.py_func == 'train':
        trainer.train()
    elif cfg.py_func == 'val':
        trainer.validate()
    else:
        raise NameError(f'Function {cfg.py_func} does not exist')




if __name__ == '__main__':
    from apep.config.parse import parse_command_line_arguments, args_to_hydra_list
    args_dict = parse_command_line_arguments()


    CONFIG_PATH = 'apep/config'
    CONFIG_NAME = 'default_training'

    num_gpus = torch.cuda.device_count()
    job_name = args_dict['model']

    if job_name == 'apep':
        args_dict['description'] = '--reward-${model.model_params.reward_func}--dyna-${model.model_params.vehicle_dynamics}' + args_dict['description']


    if 'py_func' in args_dict and args_dict['py_func'] == 'val':
        base_dir = '${ckpt_dir}/../validate'
    else:
        base_dir = Path('results/${py_func}') / job_name

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=CONFIG_PATH)
    cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
        f'base_dir={base_dir}',
        f'num_gpus={num_gpus}',
        f'job_name={job_name}',
    ] + args_to_hydra_list(args_dict))

    main(cfg)

