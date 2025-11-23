from ImageIO import load_image, save_image

SNAPSHOT_INTERVAL = 5

class Pipeline:
    def __init__(self):
        self.original = None
        self.snapshots = {}  # position → image
        self.history = []  # list of (func, args, kwargs)
        self.position = -1

    def load(self, filename):
        self.original = load_image(filename)
        self.snapshots.clear()
        self.history.clear()
        self.position = -1
        self.snapshots[0] = self.original.copy()
        print("Image loaded.")

    def apply(self, func, *args, **kwargs):
        self.history = self.history[:self.position + 1]
        self.history.append((func, args, kwargs))
        self.position += 1
        if self.position % SNAPSHOT_INTERVAL == 0:
            self.snapshots[self.position] = self.current().copy()

    def undo(self):
        if self.position >= 0:
            self.position -= 1
        else:
            print("Nothing to undo.")

    def redo(self):
        if self.position < len(self.history) - 1:
            self.position += 1
        else:
            print("Nothing to redo.")

    def current(self):
        if self.original is None:
            raise RuntimeError("No image loaded.")

        snap_pos = self.position
        while snap_pos not in self.snapshots:
            snap_pos -= 1

        img = self.snapshots[snap_pos].copy()

        for i in range(snap_pos + 1, self.position + 1):
            func, args, kwargs = self.history[i]
            img = func(img, *args, **kwargs)

        return img

    def save(self, filename):
        save_image(filename, self.current())
        print(f"Saved to {filename}")