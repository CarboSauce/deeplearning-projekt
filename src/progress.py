"""Simple progress tracking: one line per finished training with elapsed time and ETA."""
import time
from pathlib import Path


def fmt_time(seconds):
    seconds = int(seconds)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class Progress:
    def __init__(self, total, log_path=None):
        self.total = total
        self.done = 0
        self.start = time.time()
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("", encoding="utf-8")   # fresh log for every run

    def log(self, message):
        """Print (flushed immediately, so the IDE console updates) and append to progress.log."""
        print(message, flush=True)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(message + "\n")

    def step(self, label, run_seconds, details=""):
        """Call once after every finished training run."""
        self.done += 1
        elapsed = time.time() - self.start
        eta = elapsed / self.done * (self.total - self.done)   # rough: assumes equal run times
        self.log(
            f"[{self.done:>3}/{self.total}] {100 * self.done / self.total:5.1f}% | "
            f"elapsed {fmt_time(elapsed)} | ETA ~{fmt_time(eta)} | "
            f"{label} | {run_seconds:.0f}s | {details}"
        )

    def finish(self):
        self.log(f"\nAll {self.total} runs finished in {fmt_time(time.time() - self.start)}.")