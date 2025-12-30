import sys
import os
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from modules.ControllerAgent import DialogueController
import config

class DualLogger:

    def __init__(self, filepath, mirror_to_terminal: bool = True):
        self.terminal = sys.stdout
        self.logfile = open(filepath, "a", encoding='utf-8')
        self.mirror_to_terminal = mirror_to_terminal

    def write(self, message):
        if self.mirror_to_terminal:
            self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush() 

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        try:
            self.logfile.close()
        except Exception:
            pass

def load_profiles(profile_path: str) -> list[dict]:
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Profile not found: {profile_path}")
    with open(profile_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return [data]

def run_profile_job(profile: dict, idx: int, timestamp: str, log_dir: str, enable_file_log: bool, verbose: bool) -> dict:
    original_stdout = sys.stdout
    user_id = profile.get("user_id", f"user_{idx}")
    log_path = os.path.join(log_dir, f"run_{timestamp}_{user_id}.log")
    logger = None
    devnull = None
    try:
        if enable_file_log:
            logger = DualLogger(log_path, mirror_to_terminal=verbose)
            sys.stdout = logger
        elif not verbose:
            devnull = open(os.devnull, "w")
            sys.stdout = devnull

        controller = DialogueController(
            profile_data=profile,
            output_path="",
            enable_result_file=False
        )
        return controller.run()
    finally:
        if enable_file_log and logger:
            logger.close()
        if devnull:
            devnull.close()
        sys.stdout = original_stdout

class ProgressBar:
    def __init__(self, total: int, width: int = 30):
        self.total = max(total, 0)
        self.width = width
        self.current = 0
        if self.total == 0:
            print("No profiles to process.")
        else:
            self._render()

    def update(self, step: int = 1):
        if self.total == 0:
            return
        self.current += step
        self._render()

    def _render(self):
        filled = 0 if self.total == 0 else int(self.width * self.current / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        print(f"\rProgress [{bar}] {self.current}/{self.total}", end="", flush=True)

    def close(self):
        if self.total > 0:
            print()

if __name__ == "__main__":
    PROFILE_SRC = "output/sample_profile_100.json"
    OUTPUT_DST = "output/dialogue_10.json"
    PROFILE_LIMIT = 10  # 0 表示全量；>0 则仅生成前 N 个 profile
    WORKERS = config.DIALOGUE_MAX_WORKERS
    VERBOSE_LOG = False         # True 时在控制台打印详细对话
    LOG_TO_FILE = False         # True 时将详细日志写入 output/logs

    log_dir = "output/logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    profiles = load_profiles(PROFILE_SRC)
    total_available = len(profiles)
    if PROFILE_LIMIT and PROFILE_LIMIT > 0:
        profiles = profiles[:PROFILE_LIMIT]
    selected_count = len(profiles)

    indexed_results = []
    errors = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(
                run_profile_job,
                profile,
                idx,
                timestamp,
                log_dir,
                LOG_TO_FILE,
                VERBOSE_LOG,
            ): idx
            for idx, profile in enumerate(profiles)
        }
        progress = ProgressBar(len(futures))
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                res = fut.result()
                indexed_results.append((idx, res))
            except Exception as exc:
                errors.append((idx, str(exc)))
            finally:
                progress.update()
        progress.close()

    indexed_results.sort(key=lambda x: x[0])
    all_results = [res for _, res in indexed_results if res is not None]

    aggregate_payload = all_results
    os.makedirs(os.path.dirname(OUTPUT_DST) or ".", exist_ok=True)
    with open(OUTPUT_DST, "w", encoding="utf-8") as f:
        json.dump(aggregate_payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(aggregate_payload)} dialogues (requested: {selected_count}, available: {total_available}) to {OUTPUT_DST}")
    if errors:
        print(f"Completed with {len(errors)} error(s):")
        for idx, msg in errors:
            print(f"  - Profile #{idx}: {msg}")
