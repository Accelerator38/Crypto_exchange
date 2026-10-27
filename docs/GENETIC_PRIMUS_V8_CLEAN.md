# Genetic Primus V8 Clean

## Статус

V8 Clean начат заново от замороженной V7-гипотезы
`link_isolated_liquidation_reversal_prospective`. Старый контур
`exia.genetic_primus.prospective_v8` считается незавершённым прототипом и не
должен использоваться для новых запусков. Его файлы сохранены только для
аудита.

Текущий контур собирает только публичные Binance COIN-M liquidation messages.
Он не импортирует торговый runtime, не читает credentials и не умеет создавать
ордера.

## Зафиксированные ограничения

- принимаются только сообщения `forceOrder` с `st=2`;
- используется неизменяемый full8-набор COIN-M perpetual symbols;
- исходная LINK-гипотеза и event thresholds закреплены SHA и значениями V7;
- raw source message сохраняется до нормализованного liquidation event;
- tape является append-only SHA-256 chain;
- существующий root нельзя продолжать или склеивать с другим root;
- disconnect/reconnect фиксируется как source gap;
- каждый clean-room запуск ограничен 300 секундами;
- paper/live/promotion/orders остаются запрещены.

## Проверка

Локальный безопасный набор:

```powershell
E:\Crypto_exchange\.venv\Scripts\python.exe -m pytest tests\test_genetic_primus_v8_clean.py -q
```

Проблемный старый тест
`tests/test_genetic_primus_v8_tape_checker.py` не относится к clean-room
контуру и не должен запускаться в текущей задаче: его запуск совпадал с
повторными сбоями приложения.

Ограниченный read-only smoke-test:

```powershell
E:\Crypto_exchange\.venv\Scripts\python.exe tools\run_genetic_primus_v8_clean.py --duration-seconds 8
```

Проверка одного root:

```powershell
E:\Crypto_exchange\.venv\Scripts\python.exe tools\check_genetic_primus_v8_clean.py Retrodate\genetic_primus_v8_clean\<root_id>
```

Валидный smoke-root окончательного контракта:
`Retrodate/genetic_primus_v8_clean/root_20260824T115847Z_19a13362`.
Contract SHA-256:
`7e0e9eb3cfa7b8da4f4c571722ee6bca533e5f6d40991d0a4f04657e2c916fdb`.

Root `root_20260824T115710Z_5a82ca93` создан до окончательной фиксации event
thresholds и является только технической пробой, не prospective evidence.

## Следующий этап

Clean-room status checker реализован. Он проверяет PID активного процесса,
freshness status, contract SHA, целостность tape, terminal record и fail-closed
flags. Окончательный smoke-root получает `SUCCESS`; ранний pre-freeze root
получает `FAILURE` из-за contract mismatch.

Следующий этап — отдельный ограниченный supervisor, который создаёт новый root
после каждого завершения/перезапуска и проверяет предыдущий root перед новым
подключением. До его реализации непрерывный фоновый сбор не запускается.

## Retrospective shadow research

V8 Clean не может обучаться до накопления независимого prospective corpus:
текущие clean roots содержат ноль событий. Для проверки возможных улучшений
добавлен отдельный fail-closed shadow-контур. Он обучает детерминированный
линейный genetic gate на V7-событиях всех символов, кроме LINK, а LINK оставляет
как symbol-holdout. Этот запуск является только engineering diagnostic: он не
изменяет frozen V8 thresholds и всегда сохраняет `credentials_allowed=false`,
`orders_enabled=false`, `paper_trading_enabled=false`,
`live_trading_enabled=false`, `promotion_authority=false`.

```powershell
E:\Crypto_exchange\.venv\Scripts\python.exe tools\train_genetic_primus_v8_clean_shadow.py `
  --output-dir Reports\Exia\Genetic_Primus\v8_clean_shadow_<id> `
  --acknowledge-retrospective-only
```

Первый воспроизводимый shadow-run отклонён: отбор повысил средний LINK net bps,
но снизил диагностический LCB относительно frozen rule. На историческом LINK
event set frozen V8 также совпадает с обычным 4H mean-reversion control, поэтому
никакого доказанного incremental edge у shadow-модели нет.
