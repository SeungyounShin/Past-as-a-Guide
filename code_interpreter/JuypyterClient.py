import os
import signal
import re
import threading
from jupyter_client import KernelManager


class JupyterNotebook:
    def __init__(self):
        self.lock = threading.Lock()
        self.km = None
        self.kc = None
        self._initialize_kernel()

    def _initialize_kernel(self):
        """Starts the Jupyter kernel."""
        self.km = KernelManager()
        self.km.start_kernel()
        self.kc = self.km.client()

    def clean_output(self, outputs):
        outputs_only_str = []
        for i in outputs:
            if isinstance(i, dict) and "text/plain" in i:
                outputs_only_str.append(i["text/plain"])
            elif isinstance(i, str):
                outputs_only_str.append(i)
            elif isinstance(i, list):
                error_msg = "\n".join(i)
                error_msg = re.sub(r"\x1b\[.*?m", "", error_msg)
                outputs_only_str.append(error_msg)

        full_output_str = "\n".join(outputs_only_str).strip()
        if len(full_output_str) > 1000:
            full_output_str = (
                f"{full_output_str[:333]}\n... skip ...\n{full_output_str[-333:]}"
            )
        return full_output_str

    def add_and_run(self, code_string):
        def run_code_in_thread():
            nonlocal outputs, error_flag
            if not self.kc:
                return

            msg_id = self.kc.execute(code_string)
            while True:
                try:
                    msg = self.kc.get_iopub_msg(timeout=10)
                    msg_type = msg["header"]["msg_type"]
                    content = msg["content"]

                    if msg_type == "execute_result":
                        outputs.append(content["data"])
                    elif msg_type == "stream":
                        outputs.append(content["text"])
                    elif msg_type == "error":
                        error_flag = True
                        outputs.append(content["traceback"])
                    if msg_type == "status" and content["execution_state"] == "idle":
                        break
                except:
                    break

        outputs = []
        error_flag = False

        thread = threading.Thread(target=run_code_in_thread)
        thread.start()
        thread.join(timeout=30)

        if thread.is_alive():
            outputs = ["Timeout after 30 seconds"]
            error_flag = True

        return self.clean_output(outputs), error_flag

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Shutdown the kernel."""
        with self.lock:
            if self.km:
                try:
                    self.km.shutdown_kernel(now=True)
                except Exception as e:
                    print(f"Exception while jupyter notebook close -> {e}")
                    if self.km.has_kernel:
                        os.kill(self.km.kernel.pid, signal.SIGKILL)
            self.km = None
            self.kc = None
            print("Jupyter Client Killed Successfully")
