import json
import os

CONFIG_FILE = "config.json"

# Default configuration
DEFAULT_CONFIG = {
    "input_dir": ".",
    "output_dir": ".",
    # Modes:
    # 0 = Manual (standard)
    # 1 = Auto-save to new file (e.g., 1_blur.png, 2_sharpen.png)
    # 2 = Auto-save to fixed file (overwrite specific file)
    "mode": 0,
    "fixed_filename": "output.jpg"
}


class ConfigManager:
    def __init__(self):
        self.config = DEFAULT_CONFIG.copy()
        self.load()

    def load(self):
        """Loads configuration from the JSON file."""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    data = json.load(f)
                    # Update only existing keys to avoid breaking changes
                    for key in self.config:
                        if key in data:
                            self.config[key] = data[key]
            except Exception as e:
                print(f"Warning: Could not load config: {e}")

    def save(self):
        """Saves current configuration to the JSON file."""
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get(self, key):
        return self.config.get(key)

    def set(self, key, value):
        self.config[key] = value
        self.save()


# Global instance to be used across the app
cfg = ConfigManager()
