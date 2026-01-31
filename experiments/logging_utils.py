"""
Logging utilities for experiments.

This module provides convenient logging helpers specifically designed
for the experiments in the FlowGSP framework.

Note: This module provides simpler, experiment-focused utilities that wrap
the core logging functions from flowgsp.utils.logging_config. Use these
when you want experiment-specific defaults without needing to manage
log formats and other details. For more control, use the core module directly.
"""

import logging
from pathlib import Path
from typing import Optional, Union

from flowgsp.utils.logging_config import (
    setup_logger,
    set_library_log_levels,
    EXPERIMENT_FORMAT,
)


def setup_experiment_logger(
    experiment_name: str,
    verbose: bool = True,
    log_to_file: bool = False,
    results_dir: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """
    Set up a logger for an experiment with sensible defaults.

    This is a convenience function that wraps the core logging setup
    with experiment-specific defaults.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment. This will be used as the logger name
        and (optionally) as part of the log filename.
    verbose : bool, optional
        If True, log level is INFO. If False, log level is WARNING.
        This matches the existing verbose parameter pattern in experiments.
        Default is True.
    log_to_file : bool, optional
        If True, also log to a file. Default is False.
    results_dir : str or Path, optional
        Directory where log files should be saved. Only used if
        log_to_file is True. If None, logs to current directory.

    Returns
    -------
    logging.Logger
        Configured logger for the experiment.

    Examples
    --------
    >>> from experiments.logging_utils import setup_experiment_logger
    >>> logger = setup_experiment_logger("my_experiment", verbose=True)
    >>> logger.info("Starting experiment")

    >>> logger = setup_experiment_logger(
    ...     "my_experiment",
    ...     verbose=True,
    ...     log_to_file=True,
    ...     results_dir="results"
    ... )
    >>> logger.info("This will appear in console and file")
    """
    # Determine log level
    level = logging.INFO if verbose else logging.WARNING

    # Determine log file path
    log_file = None
    if log_to_file:
        if results_dir is not None:
            log_path = Path(results_dir)
        else:
            log_path = Path(".")
        log_file = log_path / f"{experiment_name}.log"

    # Create logger
    logger = setup_logger(
        name=experiment_name,
        level=level,
        log_file=log_file,
        format_string=EXPERIMENT_FORMAT,
        console=True,
    )

    # Suppress noisy libraries unless in DEBUG mode
    if level > logging.DEBUG:
        set_library_log_levels(logging.ERROR)

    return logger


class ExperimentLogger:
    """
    Context manager for experiment logging.

    This class provides a convenient way to set up logging for an
    experiment and automatically handle setup and teardown.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    verbose : bool, optional
        If True, log level is INFO. If False, log level is WARNING.
        Default is True.
    log_to_file : bool, optional
        If True, also log to a file. Default is False.
    results_dir : str or Path, optional
        Directory where log files should be saved.

    Examples
    --------
    >>> from experiments.logging_utils import ExperimentLogger
    >>> with ExperimentLogger("my_experiment", verbose=True) as logger:
    ...     logger.info("Starting experiment")
    ...     # Your experiment code here
    ...     logger.info("Experiment completed")
    """

    def __init__(
        self,
        experiment_name: str,
        verbose: bool = True,
        log_to_file: bool = False,
        results_dir: Optional[Union[str, Path]] = None,
    ):
        self.experiment_name = experiment_name
        self.verbose = verbose
        self.log_to_file = log_to_file
        self.results_dir = results_dir
        self.logger = None

    def __enter__(self) -> logging.Logger:
        """Set up the logger when entering the context."""
        self.logger = setup_experiment_logger(
            experiment_name=self.experiment_name,
            verbose=self.verbose,
            log_to_file=self.log_to_file,
            results_dir=self.results_dir,
        )
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up when exiting the context."""
        if self.logger is not None:
            # Log exception if one occurred
            if exc_type is not None:
                self.logger.error(
                    f"Experiment failed with {exc_type.__name__}: {exc_val}",
                    exc_info=True,
                )
            # Close handlers
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
        return False  # Don't suppress exceptions


def print_section(logger: logging.Logger, title: str, width: int = 60) -> None:
    """
    Print a formatted section header to logs.

    This maintains the visual style of the existing print statements
    while using the logging system.

    Parameters
    ----------
    logger : logging.Logger
        Logger to use.
    title : str
        Title of the section.
    width : int, optional
        Width of the separator line. Default is 60.

    Examples
    --------
    >>> logger = setup_experiment_logger("my_experiment")
    >>> print_section(logger, "Data Loading")
    """
    logger.info("=" * width)
    logger.info(title)
    logger.info("=" * width)
