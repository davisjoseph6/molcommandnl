import sys
import io
import contextlib
import logging


class StreamCapture(io.StringIO):
    def __init__(self, buffer_list):
        super().__init__()
        self.buffer_list = buffer_list

    def write(self, s):
        self.buffer_list.append(s)
        return super().write(s)

    def flush(self):
        pass


class LogCaptureHandler(logging.Handler):
    def __init__(self, buffer_list):
        super().__init__()
        self.buffer_list = buffer_list

    def emit(self, record):
        log_entry = self.format(record)
        self.buffer_list.append(log_entry)


@contextlib.contextmanager
def capture_python_output(debug_buffer):
    # Redirection stdout/stderr
    new_stdout = StreamCapture(debug_buffer)
    new_stderr = StreamCapture(debug_buffer)

    log_handler = LogCaptureHandler(debug_buffer)
#    log_handler.setLevel(logging.DEBUG)
#    log_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    # Sauvegarder les streams système
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old___stdout__ = sys.__stdout__
    old___stderr__ = sys.__stderr__

    sys.stdout = new_stdout
    sys.stderr = new_stderr
    sys.__stdout__ = new_stdout
    sys.__stderr__ = new_stderr

    # Collecte de tous les loggers existants (incluant les personnalisés)
    manager = logging.Logger.manager
    all_loggers = [logging.getLogger(name) for name in manager.loggerDict.keys()]
    all_loggers.append(logging.getLogger())  # inclut root

    # Sauvegarde des handlers existants
    saved_handlers = {}
    for logger in all_loggers:
        saved_handlers[logger] = logger.handlers[:]
        logger.handlers = [log_handler]
 #       logger.setLevel(logging.DEBUG)
        logger.propagate = False

    try:
        yield
    finally:
        # Restauration des streams système
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        sys.__stdout__ = old___stdout__
        sys.__stderr__ = old___stderr__

        # Restauration des loggers
        for logger in all_loggers:
            logger.handlers = saved_handlers.get(logger, [])

