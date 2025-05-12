"""
Logging configuration module for SPT Pro.

This module provides functions for configuring logging throughout the package.
"""

import os
import logging
from typing import Optional, Dict, Any, Union
import json
import yaml

# Default logging configuration
DEFAULT_LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': os.path.expanduser('~/.spt_analyzer/logs/spt_analyzer.log'),
            'mode': 'a'
        }
    },
    'loggers': {
        'spt_analyzer': {
            'level': 'INFO',
            'handlers': ['console', 'file'],
            'propagate': False
        }
    },
    'root': {
        'level': 'WARNING',
        'handlers': ['console']
    }
}

# Create logger for this module
logger = logging.getLogger(__name__)


def configure_logging(config: Optional[Dict[str, Any]] = None, 
                     log_file: Optional[str] = None,
                     log_level: Optional[str] = None) -> None:
    """
    Configure logging for the package.
    
    Parameters
    ----------
    config : dict, optional
        Logging configuration dictionary, by default None
    log_file : str, optional
        Path to log file, by default None
    log_level : str, optional
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL), by default None
        
    Notes
    -----
    If config is provided, it will be used as the logging configuration.
    If log_file is provided, it will override the file handler's filename.
    If log_level is provided, it will override the logger's level.
    """
    try:
        import logging.config
        
        # Start with default configuration
        logging_config = DEFAULT_LOGGING_CONFIG.copy()
        
        # Update with provided configuration
        if config is not None:
            _update_nested_dict(logging_config, config)
        
        # Override log file if provided
        if log_file is not None:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(os.path.abspath(log_file))
            os.makedirs(log_dir, exist_ok=True)
            
            # Update file handler configuration
            if 'handlers' in logging_config and 'file' in logging_config['handlers']:
                logging_config['handlers']['file']['filename'] = log_file
        else:
            # Create default log directory
            log_dir = os.path.dirname(logging_config['handlers']['file']['filename'])
            os.makedirs(log_dir, exist_ok=True)
        
        # Override log level if provided
        if log_level is not None:
            # Convert string level to logging level
            numeric_level = getattr(logging, log_level.upper(), None)
            if not isinstance(numeric_level, int):
                logger.warning(f"Invalid log level: {log_level}, using INFO")
                numeric_level = logging.INFO
            
            # Update logger configuration
            if 'loggers' in logging_config and 'spt_analyzer' in logging_config['loggers']:
                logging_config['loggers']['spt_analyzer']['level'] = log_level.upper()
        
        # Apply configuration
        logging.config.dictConfig(logging_config)
        
        logger.info(f"Logging configured with level {logging_config['loggers']['spt_analyzer']['level']}")
        
    except Exception as e:
        # Fallback to basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger.error(f"Error configuring logging: {str(e)}")
        logger.info("Falling back to basic logging configuration")


def load_logging_config(file_path: str) -> Dict[str, Any]:
    """
    Load logging configuration from file.
    
    Parameters
    ----------
    file_path : str
        Path to logging configuration file
        
    Returns
    -------
    dict
        Logging configuration dictionary
        
    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the file format is not supported
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Logging configuration file not found: {file_path}")
            
        # Determine file format from extension
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.json':
            with open(file_path, 'r') as f:
                config = json.load(f)
        elif ext in ['.yaml', '.yml']:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}")
        
        logger.info(f"Loaded logging configuration from {file_path}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading logging configuration: {str(e)}")
        raise


def save_logging_config(config: Dict[str, Any], file_path: str) -> None:
    """
    Save logging configuration to file.
    
    Parameters
    ----------
    config : dict
        Logging configuration dictionary
    file_path : str
        Path to save logging configuration file
        
    Raises
    ------
    ValueError
        If the file format is not supported
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Determine file format from extension
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.json':
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
        elif ext in ['.yaml', '.yml']:
            with open(file_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}")
        
        logger.info(f"Saved logging configuration to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving logging configuration: {str(e)}")
        raise


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Parameters
    ----------
    name : str
        Logger name
        
    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(name)


def set_log_level(level: Union[str, int], logger_name: Optional[str] = None) -> None:
    """
    Set the log level for a logger.
    
    Parameters
    ----------
    level : str or int
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) or numeric level
    logger_name : str, optional
        Logger name, by default None (root logger)
    """
    try:
        # Convert string level to numeric level if needed
        if isinstance(level, str):
            numeric_level = getattr(logging, level.upper(), None)
            if not isinstance(numeric_level, int):
                logger.warning(f"Invalid log level: {level}, using INFO")
                numeric_level = logging.INFO
        else:
            numeric_level = level
        
        # Get logger
        target_logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
        
        # Set level
        target_logger.setLevel(numeric_level)
        
        logger.info(f"Set log level to {level} for logger {logger_name or 'root'}")
        
    except Exception as e:
        logger.error(f"Error setting log level: {str(e)}")


def _update_nested_dict(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update nested dictionary recursively.
    
    Parameters
    ----------
    d : dict
        Dictionary to update
    u : dict
        Dictionary with updates
        
    Returns
    -------
    dict
        Updated dictionary
    """
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _update_nested_dict(d[k], v)
        else:
            d[k] = v
    return d
