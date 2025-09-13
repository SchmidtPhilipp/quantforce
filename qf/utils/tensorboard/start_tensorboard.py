import platform
import shutil
import socket
import subprocess
import time
import webbrowser
from pathlib import Path

from qf.utils.logging_config import get_logger
from qf.utils.tensorboard.safari import bring_safari_tab_to_front

logger = get_logger(__name__)


def detect_os():
    """Detect the operating system and return a string identifier."""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    elif system == "linux":
        return "linux"
    elif system == "windows":
        return "windows"
    else:
        return "unknown"


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


def get_default_browser():
    """Get the default browser based on the operating system."""
    system = detect_os()

    if system == "macos":
        return "safari"
    elif system == "linux":
        # Try to detect common Linux browsers
        if shutil.which("google-chrome"):
            return "chrome"
        elif shutil.which("firefox"):
            return "firefox"
        else:
            return "default"  # Use system default
    elif system == "windows":
        return "chrome"  # Default to Chrome on Windows
    else:
        return "default"


def start_tensorboard(
    logdir="runs", port=6004, mode=None, reload_interval=30, host=None
):
    """
    Start TensorBoard with appropriate browser based on OS.

    Args:
        logdir (str): Directory containing the logs
        port (int): Port to run TensorBoard on
        mode (str): Browser mode ('safari', 'chrome', 'firefox', 'vscode', 'default')
        reload_interval (int): How often to reload the data
        host (str): Host to bind TensorBoard to. If None, will use '0.0.0.0' on Linux and 'localhost' on other systems
    """
    # Convert logdir to absolute path
    logdir = str(Path(logdir).resolve())

    # Set default host based on OS
    if host is None:
        host = "0.0.0.0" if detect_os() == "linux" else "localhost"
        logger.info(f"Using default host for {detect_os()}: {host}")

    tb_url = f"http://{host}:{port}"

    # If no mode specified, use OS-specific default
    if mode is None:
        mode = get_default_browser()
        logger.info(f"Using default browser mode for {detect_os()}: {mode}")

    if not is_port_in_use(port):
        logger.info("🚀 Starting TensorBoard...")
        subprocess.Popen(
            [
                "tensorboard",
                f"--logdir={logdir}",
                f"--port={port}",
                f"--reload_interval={reload_interval}",
                f"--host={host}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(3)

        if host == "0.0.0.0":
            # Get local IP address
            try:
                import socket

                hostname = socket.gethostname()
                local_ip = socket.gethostbyname(hostname)
                logger.info(f"🌐 TensorBoard is accessible at:")
                logger.info(f"   - Local: http://localhost:{port}")
                logger.info(f"   - Network: http://{local_ip}:{port}")

                # Additional Linux-specific information
                if detect_os() == "linux":
                    logger.info(
                        "ℹ️ On Linux, TensorBoard is automatically accessible from other computers in the network."
                    )
                    logger.info(
                        "ℹ️ Make sure your firewall allows connections on port {port}."
                    )
            except Exception as e:
                logger.warning(f"Could not determine local IP: {e}")
    else:
        logger.info("ℹ️ TensorBoard already running.")

    # Handle different browser modes
    if mode == "safari" and detect_os() == "macos":
        logger.info("🌐 Bringing TensorBoard tab to front (Safari)...")
        bring_safari_tab_to_front(title_match="TensorBoard", url_match=tb_url)
    elif mode == "chrome":
        logger.info("🌐 Opening TensorBoard in Google Chrome...")
        try:
            if detect_os() == "linux":
                webbrowser.get("google-chrome").open(tb_url)
            else:
                webbrowser.get("chrome").open(tb_url)
        except Exception as e:
            logger.warning(
                f"Could not open Chrome: {e}. Falling back to default browser."
            )
            webbrowser.open(tb_url)
    elif mode == "firefox":
        logger.info("🌐 Opening TensorBoard in Firefox...")
        try:
            webbrowser.get("firefox").open(tb_url)
        except Exception as e:
            logger.warning(
                f"Could not open Firefox: {e}. Falling back to default browser."
            )
            webbrowser.open(tb_url)
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
    elif mode == "default":
        logger.info("🌐 Opening TensorBoard in default browser...")
        webbrowser.open(tb_url)
    else:
        logger.error(
            f"❌ Unknown mode: {mode} — use 'safari', 'chrome', 'firefox', 'vscode', or 'default'"
        )
        logger.info("Falling back to default browser...")
        webbrowser.open(tb_url)
