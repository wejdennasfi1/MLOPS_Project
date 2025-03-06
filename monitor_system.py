import psutil
import time


def monitor_system():
    # Surveillance de la CPU
    cpu_usage = psutil.cpu_percent(interval=1)
    print(f"Usage CPU : {cpu_usage}%")

    # Surveillance de la RAM
    memory = psutil.virtual_memory()
    print(f"Usage RAM : {memory.percent}% (Total: {memory.total / (1024 ** 3):.2f} GB)")

    # Surveillance de l'espace disque
    disk = psutil.disk_usage("/")
    print(f"Usage disque : {disk.percent}% (Total: {disk.total / (1024 ** 3):.2f} GB)")


if __name__ == "__main__":
    while True:
        monitor_system()
        time.sleep(5)  # Rafraîchir toutes les 5 secondes
