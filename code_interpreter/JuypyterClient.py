import os
import signal
import re
import asyncio
from jupyter_client import KernelManager
from weakref import finalize

class JupyterNotebook:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.km = None
        self.kc = None
        # Use a single global loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._initialize_kernel()

    def _initialize_kernel(self):
        """Starts the Jupyter kernel."""
        try:
            self.km = KernelManager()
            self.km.start_kernel()
            self.kc = self.km.client()

            # Use finalize to ensure cleanup when the object is garbage collected
            finalize(self, self.close)
        except Exception as e:
            print(f"Exception during kernel initialization -> {e}")
            self.close()

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

    async def run_code_async(self, code_string, outputs, error_flag):
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
                    error_flag[0] = True
                    outputs.append(content["traceback"])
                if msg_type == "status" and content["execution_state"] == "idle":
                    break
            except:
                break
            await asyncio.sleep(0)

    def add_and_run(self, code_string):
        outputs = []
        error_flag = [False]  # Use a list so it's mutable

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.run_code_async(code_string, outputs, error_flag))
        finally:
            loop.close()

        return self.clean_output(outputs), error_flag[0]

    def __enter__(self):
        return self

    async def async_shutdown(self):
        """Asynchronous shutdown of the kernel."""
        if self.kc:
            # Stop the channels explicitly
            self.kc.stop_channels()
            # Explicitly delete the reference
            del self.kc
            self.kc = None

        if self.km:
            try:
                # Use the internal kernel attribute to get its pid
                kernel_pid = self.km._kernel.pid if hasattr(self.km, "_kernel") else None
                if kernel_pid:  # Only if the kernel's pid is found
                    os.kill(kernel_pid, signal.SIGKILL)
                else:
                    await self.km._async_shutdown_kernel(now=True)
            except Exception as e:
                print(f"Exception while shutting down jupyter notebook -> {e}")
            # Explicitly delete the reference
            del self.km
            self.km = None
            print("Jupyter Client Killed Successfully")

    def close(self):
        """Shutdown the kernel."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.async_shutdown())
        finally:
            loop.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        # Close the event loop
        self.loop.close()


if __name__ == "__main__":
    import gc

    notebook = JupyterNotebook()
    out = notebook.add_and_run("a=1\nb=2")
    print(out) # ("", False)
    out = notebook.add_and_run("a+b")
    print(out) # ("3", False)
    gc.collect() 
    
    # solution for problem in jupyter notebook
    """
    # upgrade all related packages

    pip install --upgrade ipykernel jupyter_core jupyter_client pyzmq

    """
