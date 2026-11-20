"""Оновлення вбудованих файлів Wikimedia Codex у wle_filter/static/codex/.

    cd /tmp && npm install @wikimedia/codex @wikimedia/codex-design-tokens
    python tools/update_codex.py /tmp/node_modules

Codex (https://doc.wikimedia.org/codex/latest/) — дизайн-система Wikimedia,
ліцензія GPL-2.0-or-later. Використовуються лише CSS-файли (CSS-only компоненти),
без Vue, тож інтерфейс працює без збірки й без інтернету.
"""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "wle_filter" / "static" / "codex"


def main(node_modules: str) -> None:
    nm = Path(node_modules)
    codex = nm / "@wikimedia" / "codex"
    tokens = nm / "@wikimedia" / "codex-design-tokens" / "dist"
    version = json.loads((codex / "package.json").read_text())["version"]
    DEST.mkdir(parents=True, exist_ok=True)

    shutil.copy(codex / "dist" / "codex.style.css", DEST / "codex.style.css")

    light = (tokens / "theme-wikimedia-ui-root.css").read_text(encoding="utf-8")
    dark = (tokens / "theme-wikimedia-ui-mode-dark.css").read_text(encoding="utf-8")
    dark_body = re.search(r":root\s*\{(.*)\}", dark, re.S).group(1)

    theme = (
        f"/* Wikimedia Codex design tokens v{version} (GPL-2.0-or-later).\n"
        f" * Згенеровано tools/update_codex.py — не редагуйте вручну.\n"
        f" * Тема: html[data-theme=light|dark|auto]; auto слідує налаштуванню системи. */\n\n"
        + light.split("*/", 1)[1].strip() + "\n\n"
        + "html[data-theme=\"dark\"] {" + dark_body + "}\n\n"
        + "@media (prefers-color-scheme: dark) {\n  html[data-theme=\"auto\"] {"
        + dark_body + "}\n}\n"
        + "html[data-theme=\"dark\"] { color-scheme: dark; }\n"
        + "@media (prefers-color-scheme: dark) { html[data-theme=\"auto\"] { color-scheme: dark; } }\n"
    )
    (DEST / "codex-theme.css").write_text(theme, encoding="utf-8")
    (DEST / "VERSION").write_text(version + "\n")
    (DEST / "README.md").write_text(
        f"Wikimedia Codex v{version} — https://doc.wikimedia.org/codex/latest/\n\n"
        "Ліцензія: GPL-2.0-or-later. Оновлення: `python tools/update_codex.py <node_modules>`.\n",
        encoding="utf-8")
    print(f"Codex v{version} → {DEST}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "node_modules")
