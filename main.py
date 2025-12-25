import sys
import os
from datetime import datetime
from modules.ControllerAgent import DialogueController

class DualLogger:

    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.logfile = open(filepath, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush() 

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

if __name__ == "__main__":
    log_dir = "output/logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_{timestamp}.log")
    original_stdout = sys.stdout
    sys.stdout = DualLogger(log_path)

    profile_src = "output/run.json"
    output_dst = "output/sample_dialogue.json"
    controller = DialogueController(
        profile_path=profile_src,
        output_path=output_dst
    )
    
    controller.run()
        