import logging
import os
import sys
from pathlib import Path


def setup_logging(
    log_dir="logs",
    log_filenames=("all.log", "debug.log", "errors.log"),
    email_config=None,
):

    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Alte Handler entfernen (optional, aber nützlich bei Re-Import)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 🧾 Log-Dateien
    for name in log_filenames:
        path = Path(log_dir) / name
        file_handler = logging.FileHandler(path, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 📺 Konsole
    console_handler = logging.StreamHandler(sys.__stdout__)  # Verwende echtes stdout
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 📧 Optionaler E-Mail-Handler
    if email_config:
        from logging.handlers import SMTPHandler

        mailhost = (email_config["server"], email_config.get("port", 587))
        credentials = (email_config["email"], email_config["password"])
        secure = () if email_config.get("use_tls", True) else None

        mail_handler = SMTPHandler(
            mailhost=mailhost,
            fromaddr=email_config["email"],
            toaddrs=email_config["to"],
            subject=email_config.get("subject", "Fehler im Python-Programm"),
            credentials=credentials,
            secure=secure,
        )
        mail_handler.setLevel(logging.ERROR)
        mail_handler.setFormatter(formatter)
        logger.addHandler(mail_handler)

    # 🔁 print() umleiten, aber zur Konsole und ins Log
    class LoggerWriter:
        def __init__(self, level_func, stream):
            self.level_func = level_func
            self.stream = stream

        def write(self, message):
            message = message.strip()
            if message:
                self.level_func(message)
                self.stream.write(message + "\n")
                self.stream.flush()

        def flush(self):
            self.stream.flush()

    sys.stdout = LoggerWriter(logger.info, sys.__stdout__)
    sys.stderr = LoggerWriter(logger.error, sys.__stderr__)

    return logger
