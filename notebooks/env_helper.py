from pathlib import Path
import os

def load_env():
    print("loading env")
    """
    Load environment variables from a .env file located in the same directory as the script.
    Variables should be in the format KEY=VALUE, one per line.
    """
    env_path = Path(__file__).parent / '.env'

    if not env_path.exists():
        print(f"Warning: No .env file found at {env_path}")
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Split on first = only
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()

            # Remove quotes if present
            if value and value[0] == value[-1] and value[0] in ['"', "'"]:
                value = value[1:-1]

            os.environ[key] = value

# Load environment variables immediately
load_env()