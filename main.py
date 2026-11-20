"""WLE Filter — точка входу.

    python main.py serve                         # веб-інтерфейс на http://127.0.0.1:8000
    python main.py serve --lan                   # доступ з телефона/планшета в тій самій мережі
    python main.py run --category "Wiki Loves Earth Ukraine 2026 - Quality"
    python main.py run --folder ~/photos --nomination quantity --limit 100

Команда run працює без браузера: аналізує фото, зберігає їх у тій самій базі
(прогін потім видно в інтерфейсі) і записує CSV у теку даних.
"""
from __future__ import annotations

import argparse
import sys
import time

from wle_filter import config
from wle_filter.store import Store
from wle_filter.worker import Worker


def cmd_run(args) -> int:
    presets = config.load_presets()
    if args.nomination not in presets:
        print(f"Невідома номінація: {args.nomination}. Доступні: {', '.join(presets)}")
        return 2
    if bool(args.category) == bool(args.folder):
        print("Вкажіть рівно одне: --category або --folder")
        return 2
    store = Store()
    src_type, src = ("commons", args.category) if args.category else ("folder", args.folder)
    name = args.name or f"{presets[args.nomination]['label']}: {src}"
    run_id = store.create_run(name, args.nomination, src_type, src, presets[args.nomination]["thresholds"])
    print(f"Прогін #{run_id}: {name}")

    worker = Worker(store)
    worker.start(run_id, limit=args.limit)
    try:
        while worker.is_running(run_id):
            r = store.get_run(run_id)
            print(f"\r  {r['processed']}/{r['total']}  {r['message'] or r['status']}" + " " * 20,
                  end="", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nЗупиняю… (продовжити можна з інтерфейсу)")
        worker.pause(run_id)
        worker.join(run_id)
    print()
    from recount import print_summary, export_csv
    print_summary(store, run_id)
    print(f"CSV: {export_csv(store, run_id)}")
    return 0


def lan_addresses() -> list[str]:
    """IP-адреси цього комп'ютера в локальній мережі."""
    import socket
    ips = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))      # пакет не надсилається — лише вибір інтерфейсу
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    ips.discard("127.0.0.1")
    return sorted(ips) or ["<IP-адреса цього комп'ютера>"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="WLE Filter")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="запустити веб-інтерфейс")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--lan", action="store_true",
                   help="слухати всі мережеві інтерфейси (доступ з інших пристроїв у мережі)")
    r = sub.add_parser("run", help="проаналізувати фото без інтерфейсу")
    r.add_argument("--category", help="категорія Commons")
    r.add_argument("--folder", help="локальна папка з фото")
    r.add_argument("--nomination", default="quality", help="quality | quantity")
    r.add_argument("--limit", type=int)
    r.add_argument("--name")
    args = ap.parse_args(argv)

    if args.cmd == "run":
        return cmd_run(args)
    # За замовчуванням — веб-інтерфейс
    import uvicorn
    from wle_filter.server import create_app
    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8000)
    if getattr(args, "lan", False):
        host = "0.0.0.0"
    if host in ("0.0.0.0", "::"):
        print("WLE Filter доступний у локальній мережі:")
        for ip in lan_addresses():
            print(f"  http://{ip}:{port}")
        print("  ⚠ Входу з паролем немає: будь-хто в цій мережі може переглядати фото,\n"
              "    змінювати рішення та бачити назви папок на цьому комп'ютері.")
    else:
        print(f"WLE Filter: відкрийте http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
