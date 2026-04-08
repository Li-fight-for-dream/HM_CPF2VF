from pathlib import Path


def create_save_path(save_root: str, task_name: str) -> str:
    path = Path(save_root) / task_name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
