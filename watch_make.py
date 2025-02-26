import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class MakefileEventHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        """Exécute make dès qu'un fichier du projet est modifié."""
        print(f"Modification détectée : {event.src_path}")
        subprocess.run(["make", "all"], check=True)


if __name__ == "__main__":
    path = "."  # Surveiller tout le dossier du projet
    event_handler = MakefileEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    print("Surveillance du projet activée. Toute modification déclenchera make...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
