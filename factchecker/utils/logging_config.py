import logging
import sys

# Add custom log level for API interactions
API_CALLS = 15  # Between DEBUG (10) and INFO (20)
logging.addLevelName(API_CALLS, 'API')

def api(self, message, *args, **kwargs):
    if self.isEnabledFor(API_CALLS):
        self._log(API_CALLS, message, args, **kwargs)

# Add our custom method to the Logger class
logging.Logger.api = api

def setup_logging(level=logging.INFO):
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set specific levels for different loggers
    logging.getLogger('fsspec').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(API_CALLS)  # Set OpenAI to use our custom level
    logging.getLogger('llama_index').setLevel(logging.INFO)
    
    # Our application loggers
    logging.getLogger('factchecker').setLevel(logging.INFO)
    logging.getLogger('factchecker.api').setLevel(API_CALLS)

    # Create loggers for different parts of the app
    loggers = {
        'factchecker': logging.DEBUG,
        'factchecker.indexing': logging.DEBUG,
        'factchecker.strategies': logging.DEBUG,
        'factchecker.retrieval': logging.DEBUG
    }

    for name, level in loggers.items():
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = True

    return logging.getLogger('factchecker') 