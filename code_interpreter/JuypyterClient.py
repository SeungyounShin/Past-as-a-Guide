from jupyter_client import KernelManager
import threading
import re


class JupyterNotebook:
    def __init__(self):
        self.km = KernelManager()
        self.km.start_kernel()
        self.kc = self.km.client()

    def clean_output(self, outputs):
        outputs_only_str = list()
        for i in outputs:
            if type(i) == dict:
                if "text/plain" in list(i.keys()):
                    outputs_only_str.append(i["text/plain"])
            elif type(i) == str:
                outputs_only_str.append(i)
            elif type(i) == list:
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
        # This inner function will be executed in a separate thread
        def run_code_in_thread():
            nonlocal outputs, error_flag

            # Execute the code and get the execution count
            msg_id = self.kc.execute(code_string)

            while True:
                try:
                    msg = self.kc.get_iopub_msg(timeout=20)

                    msg_type = msg["header"]["msg_type"]
                    content = msg["content"]

                    if msg_type == "execute_result":
                        outputs.append(content["data"])
                    elif msg_type == "stream":
                        outputs.append(content["text"])
                    elif msg_type == "error":
                        error_flag = True
                        outputs.append(content["traceback"])

                    # If the execution state of the kernel is idle, it means the cell finished executing
                    if msg_type == "status" and content["execution_state"] == "idle":
                        break
                except:
                    break

        outputs = []
        error_flag = False

        # Start the thread to run the code
        thread = threading.Thread(target=run_code_in_thread)
        thread.start()

        # Wait for 10 seconds for the thread to finish
        thread.join(timeout=60)

        # If the thread is still alive after 10 seconds, it's a timeout
        if thread.is_alive():
            outputs = ["Timeout after 60 seconds"]
            error_flag = True

        return self.clean_output(outputs), error_flag

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Shutdown the kernel."""
        if self.km:
            try:
                # Try to shutdown the kernel gracefully
                self.km.shutdown_kernel(now=True)
            except Exception as e:
                print(f"exception in jupyter close -> {e}")
                if self.km.has_kernel:
                    os.kill(
                        self.km.kernel.pid, signal.SIGKILL
                    )  # Forcefully kill the kernel process

        self.km = None
        self.kc = None
        print(f"Jupyter Client Killed Successfully")
