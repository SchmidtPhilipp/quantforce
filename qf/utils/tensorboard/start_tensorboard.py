import shutil
import socket
import subprocess
import time
import webbrowser

from qf.utils.logging_config import get_logger
from qf.utils.tensorboard.safari import bring_safari_tab_to_front

logger = get_logger(__name__)


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def install_vscode_extension(extension_id):
    subprocess.run(
        ["code", "--install-extension", extension_id], stdout=subprocess.DEVNULL
    )


def is_vscode_extension_installed(extension_id):
    try:
        output = subprocess.check_output(["code", "--list-extensions"])
        return extension_id in output.decode()
    except Exception:
        return False


def start_tensorboard(logdir="runs", port=6004, mode="safari", reload_interval=30):
    tb_url = f"http://localhost:{port}"

    if not is_port_in_use(port):
        logger.info("🚀 Starting TensorBoard...")
        subprocess.Popen(
            [
                "tensorboard",
                f"--logdir={logdir}",
                f"--port={port}",
                f"--reload_interval={reload_interval}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(3)
    else:
        logger.info("ℹ️ TensorBoard already running.")

    if mode == "safari":
        logger.info("🌐 Bringing TensorBoard tab to front (Safari)...")
        bring_safari_tab_to_front(title_match="TensorBoard", url_match=tb_url)
    elif mode == "chrome":
        logger.info("🌐 Opening TensorBoard in Google Chrome...")
        webbrowser.get("chrome").open(tb_url)
    elif mode == "firefox":
        logger.info("🌐 Opening TensorBoard in Firefox...")
        webbrowser.get("firefox").open(tb_url)
    elif mode == "vscode":
        if shutil.which("code") is None:
            logger.error(
                "❌ VS Code CLI not found. Install it via 'Shell Command: Install code in PATH'."
            )
            return

        extension_id = "ms-toolsai.jupyter"
        if not is_vscode_extension_installed(extension_id):
            logger.info("🔧 Installing VS Code TensorBoard extension...")
            install_vscode_extension(extension_id)
            time.sleep(2)

        subprocess.run(["code", logdir])
        logger.info(
            f"🧠 Open TensorBoard tab inside VS Code manually if needed ({logdir})"
        )
    else:
        logger.error(
            f"❌ Unknown mode: {mode} — use 'safari', 'chrome', 'firefox', or 'vscode'"
        )
