from ImageIO import load_image, save_image
from config import cfg

SNAPSHOT_INTERVAL = 5

class Pipeline:
    def __init__(self):
        self.original = None
        self.snapshots = {}  # position -> image (numpy array)
        self.history = []  # list of (func, args, kwargs, name)
        self.position = -1

    def load(self, filename):
        input_folder = cfg.get("input_dir")

        try:
            self.original = load_image(filename, folder_path=input_folder)

            self.snapshots.clear()
            self.history.clear()
            self.position = -1
            self.snapshots[0] = self.original.copy()
            print(f"Image loaded from {input_folder}/{filename}")
        except Exception as e:
            print(f"Failed to load image: {e}")

    def apply(self, func, func_name, *args, **kwargs):
        """Applies a filter and handles auto-save modes."""
        self.history = self.history[:self.position + 1]
        self.history.append((func, args, kwargs, func_name))
        self.position += 1

        if self.position % SNAPSHOT_INTERVAL == 0:
            self.snapshots[self.position] = self.current().copy()

        # Handle Auto-save modes
        mode = cfg.get("mode")
        if mode == 1:
            # Mode 1: New file for every step
            out_name = f"{self.position:02d}_{func_name}.png"
            self.save(out_name, silent=True)
            print(f"[Auto-Save] Saved: {out_name}")

        elif mode == 2:
            # Mode 2: Overwrite fixed file
            out_name = cfg.get("fixed_filename")
            self.save(out_name, silent=True)
            print(f"[Auto-Save] Updated: {out_name}")

    def undo(self):
        if self.position >= 0:
            self.position -= 1
            print("Undo successful.")
            # Trigger auto-save for Mode 2 if needed to reflect undo
            if cfg.get("mode") == 2:
                self.save(cfg.get("fixed_filename"), silent=True)
        else:
            print("Nothing to undo.")

    def redo(self):
        if self.position < len(self.history) - 1:
            self.position += 1
            print("Redo successful.")
            # Trigger auto-save for Mode 2
            if cfg.get("mode") == 2:
                self.save(cfg.get("fixed_filename"), silent=True)
        else:
            print("Nothing to redo.")

    def current(self):
        if self.original is None:
            raise RuntimeError("No image loaded.")

        if self.position == -1:
            return self.snapshots[0].copy()

        snap_pos = self.position
        while snap_pos not in self.snapshots:
            snap_pos -= 1

        img = self.snapshots[snap_pos].copy()

        for i in range(snap_pos + 1, self.position + 1):
            func, args, kwargs, _ = self.history[i]
            img = func(img, *args, **kwargs)

        return img

    def save(self, filename, silent=False):
        output_folder = cfg.get("output_dir")

        try:
            # Передаем папку отдельно
            save_image(filename, self.current(), folder_path=output_folder)
            if not silent:
                print(f"Saved to {output_folder}/{filename}")
        except Exception as e:
            print(f"Failed to save image: {e}")

    def print_history(self):
        print("\n--- Command History ---")
        if not self.history:
            print("(Empty)")
            return

        for i, item in enumerate(self.history):
            func_name = item[3]
            marker = " <--- Current" if i == self.position else ""
            print(f"{i}: {func_name} {marker}")
        print("-----------------------\n")
