# src/utils/logger.py

import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from datetime import datetime

class RepoLogger:
    def __init__(self, log_dir="logs", max_size=5*1024*1024, backup_count=3):
        """
        Initializes the logger.

        Args:
            log_dir (str): The directory to store log files. Defaults to 'logs'.
            max_size (int): Maximum size of each log file in bytes. Defaults to 5MB.
            backup_count (int): Number of backup files to keep. Defaults to 3.
        """
        self.log_dir = log_dir
        self.max_size = max_size
        self.backup_count = backup_count
        self.loggers = {}

    def get_logger(self, name=None, level=logging.INFO):
        """
        Retrieves or creates a logger with the given name.

        Args:
            name (str, optional): The name of the logger. If None, uses the calling module's name.
            level (int, optional): The logging level to be set. Defaults to logging.INFO.

        Returns:
            logging.Logger: The logger instance.
        """
        if name is None:
            name = self._get_caller_name()

        if name not in self.loggers:
            logger = logging.getLogger(name)
            logger.setLevel(level)
            
            if not logger.handlers:
                self._setup_file_handler(logger, name)
                self._setup_console_handler(logger)

            self.loggers[name] = logger

        return self.loggers[name]

    def _get_caller_name(self):
        """Get the name of the calling module."""
        frame = sys._getframe(2)
        return os.path.splitext(os.path.basename(frame.f_code.co_filename))[0]

    def _setup_file_handler(self, logger, name):
        """Sets up a rotating file handler for the logger."""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        log_file = os.path.join(self.log_dir, f"{name}.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=self.max_size,
            backupCount=self.backup_count
        )
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def _setup_console_handler(self, logger):
        """Sets up a console handler for the logger."""
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

# Global instance of RepoLogger
repo_logger = RepoLogger()

# Convenience function to get a logger
def get_logger(name=None, level=logging.INFO):
    return repo_logger.get_logger(name, level)