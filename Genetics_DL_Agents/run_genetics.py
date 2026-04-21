"""
run_genetics.py — запуск crypto_genetics.py с сохранением логов в Agents/

Что делает:
  - Создаёт папку Agents/<YYYY-MM-DD_HH-MM-SS>/
  - Дублирует весь вывод → Agents/<ts>/genetics_<ts>.log
  - Геномы (best_genome.npy, best_island_*.npy) сохраняются там же
    где их кладёт сам crypto_genetics.py (рядом со скриптом) — НЕ ТРОГАЕМ

Запуск:
    python run_genetics.py
    python run_genetics.py --generations 100 --population 50
"""

import os, sys, runpy
from datetime import datetime

_BASE      = os.path.dirname(os.path.abspath(__file__))
_TS        = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
_AGENTS_DIR = os.path.join(_BASE, "Agents", _TS)
os.makedirs(_AGENTS_DIR, exist_ok=True)

_LOG_PATH = os.path.join(_AGENTS_DIR, f"genetics_{_TS}.log")


class _Tee:
    """Дублирует stdout в лог-файл."""
    def __init__(self, stream, path):
        self._s = stream
        self._f = open(path, "w", encoding="utf-8", buffering=1)

    def write(self, data):
        self._s.write(data)
        try: self._f.write(data)
        except: pass

    def flush(self):
        self._s.flush()
        try: self._f.flush()
        except: pass

    def __getattr__(self, a): return getattr(self._s, a)


sys.stdout = _Tee(sys.stdout, _LOG_PATH)

print("=" * 70)
print("  run_genetics.py")
print(f"  Лог: {_LOG_PATH}")
print("=" * 70)
print()

try:
    runpy.run_path(os.path.join(_BASE, "crypto_genetics.py"), run_name="__main__")
except KeyboardInterrupt:
    print("\n[run_genetics] ⏹  Остановка.")
finally:
    if hasattr(sys.stdout, "_s"):
        sys.stdout = sys.stdout._s
    print(f"\n[run_genetics] Лог сохранён: {_LOG_PATH}")
