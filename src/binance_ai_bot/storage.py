"""SQLite kalıcılık katmanı: Bakiye, pozisyonlar, emirler, kararlar ve AI maliyeti."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

from .domain import Position, RiskSettings, SimulatedOrderResult


class SQLiteStorage:
    """
    Program kapanıp açıldığında bakiye, açık pozisyonlar, ayarlar ve analiz geçmişini
    veri kaybı olmadan geri yükleyen yerel SQLite deposu.
    """

    def __init__(
        self,
        db_path: Path,
        initial_balance_usdc: Decimal = Decimal("86.00"),
        timezone_name: str = "Europe/Istanbul",
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initial_balance_usdc = initial_balance_usdc
        self.tz = ZoneInfo(timezone_name)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def istanbul_date_str(self, dt_utc: datetime | None = None) -> str:
        ref = dt_utc if dt_utc is not None else datetime.now(timezone.utc)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        return ref.astimezone(self.tz).strftime("%Y-%m-%d")

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS app_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at_iso TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS positions (
                    position_id TEXT PRIMARY KEY,
                    portfolio_track TEXT NOT NULL DEFAULT 'MAIN',
                    symbol TEXT NOT NULL,
                    quantity TEXT NOT NULL,
                    entry_price TEXT NOT NULL,
                    cost_basis_usdc TEXT NOT NULL,
                    entry_fee_usdc TEXT NOT NULL,
                    opened_at_iso TEXT NOT NULL,
                    signal_key TEXT NOT NULL,
                    strategy_source TEXT NOT NULL,
                    stop_loss_price TEXT,
                    take_profit_price TEXT,
                    status TEXT NOT NULL,
                    closed_at_iso TEXT,
                    exit_price TEXT,
                    exit_fee_usdc TEXT NOT NULL DEFAULT '0',
                    realized_pnl_usdc TEXT NOT NULL DEFAULT '0',
                    close_reason TEXT,
                    closed_istanbul_date TEXT
                );

                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    portfolio_track TEXT NOT NULL DEFAULT 'MAIN',
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    requested_qty TEXT NOT NULL,
                    filled_qty TEXT NOT NULL,
                    avg_fill_price TEXT NOT NULL,
                    gross_quote_usdc TEXT NOT NULL,
                    fee_usdc TEXT NOT NULL,
                    slippage_bps TEXT NOT NULL,
                    signal_key TEXT NOT NULL,
                    strategy_source TEXT NOT NULL,
                    explanation TEXT NOT NULL,
                    realized_pnl_usdc TEXT NOT NULL DEFAULT '0',
                    istanbul_date TEXT NOT NULL,
                    created_at_iso TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS processed_signals (
                    portfolio_track TEXT NOT NULL,
                    signal_key TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    processed_at_iso TEXT NOT NULL,
                    PRIMARY KEY (portfolio_track, signal_key)
                );

                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_iso TEXT NOT NULL,
                    istanbul_date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    candle_open_time_ms INTEGER NOT NULL,
                    rule_action TEXT NOT NULL,
                    rule_reason TEXT NOT NULL,
                    ai_action TEXT NOT NULL,
                    ai_reason TEXT NOT NULL,
                    final_action TEXT NOT NULL,
                    mode_label TEXT NOT NULL,
                    risk_notes_json TEXT NOT NULL,
                    self_reported_score INTEGER,
                    executed_order_id TEXT,
                    block_reason TEXT
                );

                CREATE TABLE IF NOT EXISTS ai_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_iso TEXT NOT NULL,
                    istanbul_date TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    from_cache INTEGER NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    estimated_cost_usd TEXT NOT NULL,
                    response_json TEXT,
                    status TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_ai_cache_key ON ai_usage(cache_key, status);

                CREATE TABLE IF NOT EXISTS equity_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_iso TEXT NOT NULL,
                    istanbul_date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    main_equity_usdc TEXT NOT NULL,
                    cash_balance_usdc TEXT NOT NULL,
                    unrealized_pnl_usdc TEXT NOT NULL,
                    rule_shadow_equity_usdc TEXT NOT NULL,
                    benchmark_equity_usdc TEXT NOT NULL
                );
                """
            )
            now_iso = datetime.now(timezone.utc).isoformat()
            defaults = {
                "cash_balance_usdc": f"{self.initial_balance_usdc:.8f}",
                "initial_balance_usdc": f"{self.initial_balance_usdc:.8f}",
                "rule_shadow_cash_usdc": f"{self.initial_balance_usdc:.8f}",
                "new_entries_paused": "0",
                "pause_reason": "",
            }
            for k, v in defaults.items():
                conn.execute(
                    "INSERT OR IGNORE INTO app_state (key, value, updated_at_iso) VALUES (?, ?, ?)",
                    (k, v, now_iso),
                )

    def get_state(self, key: str, default: str = "") -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
            return str(row["value"]) if row else default

    def set_state(self, key: str, value: str) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO app_state (key, value, updated_at_iso)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at_iso = excluded.updated_at_iso
                """,
                (key, value, now_iso),
            )

    def _cash_key_for_track(self, track: str) -> str:
        if track == "MAIN":
            return "cash_balance_usdc"
        if track == "LIVE_SPOT":
            return "live_spot_cash_usdc"
        return "rule_shadow_cash_usdc"

    def get_cash_balance(self, track: str = "MAIN") -> Decimal:
        state_key = self._cash_key_for_track(track)
        default_val = "0.00000000" if track == "LIVE_SPOT" else str(self.initial_balance_usdc)
        raw = self.get_state(state_key, default_val)
        return Decimal(raw)

    def set_cash_balance(self, amount: Decimal, track: str = "MAIN") -> None:
        state_key = self._cash_key_for_track(track)
        self.set_state(state_key, f"{amount:.8f}")

    def is_new_entries_paused(self) -> tuple[bool, str]:
        paused = self.get_state("new_entries_paused", "0") == "1"
        reason = self.get_state("pause_reason", "")
        return paused, reason

    def set_new_entries_paused(self, paused: bool, reason: str = "") -> None:
        self.set_state("new_entries_paused", "1" if paused else "0")
        self.set_state("pause_reason", reason if paused else "")

    def save_risk_settings(self, settings: RiskSettings) -> None:
        payload = {
            "max_open_positions": settings.max_open_positions,
            "max_position_size_usdc": str(settings.max_position_size_usdc),
            "daily_loss_limit_usdc": str(settings.daily_loss_limit_usdc),
            "max_daily_new_positions": settings.max_daily_new_positions,
            "daily_limit_exit_policy": settings.daily_limit_exit_policy,
            "estimated_fee_rate": str(settings.estimated_fee_rate),
            "extra_slippage_bps": str(settings.extra_slippage_bps),
            "stop_loss_pct": str(settings.stop_loss_pct),
            "take_profit_pct": str(settings.take_profit_pct),
            "allow_stablecoin_conversion": settings.allow_stablecoin_conversion,
        }
        self.set_state("risk_settings_json", json.dumps(payload, ensure_ascii=False))

    def load_risk_settings(self, fallback: RiskSettings) -> RiskSettings:
        raw = self.get_state("risk_settings_json", "")
        if not raw:
            return fallback
        try:
            data = json.loads(raw)
            return RiskSettings(
                max_open_positions=int(data.get("max_open_positions", fallback.max_open_positions)),
                max_position_size_usdc=Decimal(
                    str(data.get("max_position_size_usdc", fallback.max_position_size_usdc))
                ),
                daily_loss_limit_usdc=Decimal(
                    str(data.get("daily_loss_limit_usdc", fallback.daily_loss_limit_usdc))
                ),
                max_daily_new_positions=int(
                    data.get("max_daily_new_positions", fallback.max_daily_new_positions)
                ),
                daily_limit_exit_policy=str(
                    data.get("daily_limit_exit_policy", fallback.daily_limit_exit_policy)
                ),
                estimated_fee_rate=Decimal(
                    str(data.get("estimated_fee_rate", fallback.estimated_fee_rate))
                ),
                extra_slippage_bps=Decimal(
                    str(data.get("extra_slippage_bps", fallback.extra_slippage_bps))
                ),
                stop_loss_pct=Decimal(str(data.get("stop_loss_pct", fallback.stop_loss_pct))),
                take_profit_pct=Decimal(
                    str(data.get("take_profit_pct", fallback.take_profit_pct))
                ),
                allow_stablecoin_conversion=bool(
                    data.get("allow_stablecoin_conversion", fallback.allow_stablecoin_conversion)
                ),
            )
        except Exception:
            return fallback

    def has_processed_signal(self, signal_key: str, track: str = "MAIN") -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM processed_signals WHERE portfolio_track = ? AND signal_key = ?",
                (track, signal_key),
            ).fetchone()
            return row is not None

    def mark_signal_processed(
        self, signal_key: str, order_id: str, processed_at_iso: str, track: str = "MAIN"
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO processed_signals (portfolio_track, signal_key, order_id, processed_at_iso)
                VALUES (?, ?, ?, ?)
                """,
                (track, signal_key, order_id, processed_at_iso),
            )

    def _row_to_position(self, row: sqlite3.Row) -> Position:
        return Position(
            position_id=str(row["position_id"]),
            symbol=str(row["symbol"]),
            quantity=Decimal(str(row["quantity"])),
            entry_price=Decimal(str(row["entry_price"])),
            cost_basis_usdc=Decimal(str(row["cost_basis_usdc"])),
            entry_fee_usdc=Decimal(str(row["entry_fee_usdc"])),
            opened_at_iso=str(row["opened_at_iso"]),
            signal_key=str(row["signal_key"]),
            strategy_source=str(row["strategy_source"]),
            stop_loss_price=Decimal(str(row["stop_loss_price"])) if row["stop_loss_price"] else None,
            take_profit_price=(
                Decimal(str(row["take_profit_price"])) if row["take_profit_price"] else None
            ),
            status=str(row["status"]),
            closed_at_iso=str(row["closed_at_iso"]) if row["closed_at_iso"] else None,
            exit_price=Decimal(str(row["exit_price"])) if row["exit_price"] else None,
            exit_fee_usdc=Decimal(str(row["exit_fee_usdc"])),
            realized_pnl_usdc=Decimal(str(row["realized_pnl_usdc"])),
            close_reason=str(row["close_reason"]) if row["close_reason"] else None,
        )

    def get_open_positions(self, track: str = "MAIN") -> list[Position]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM positions WHERE portfolio_track = ? AND status = 'OPEN' ORDER BY opened_at_iso ASC",
                (track,),
            ).fetchall()
            return [self._row_to_position(r) for r in rows]

    def get_all_positions(self, track: str = "MAIN", limit: int = 200) -> list[Position]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM positions WHERE portfolio_track = ? ORDER BY opened_at_iso DESC LIMIT ?",
                (track, limit),
            ).fetchall()
            return [self._row_to_position(r) for r in rows]

    def insert_position(self, pos: Position, track: str = "MAIN") -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO positions (
                    position_id, portfolio_track, symbol, quantity, entry_price,
                    cost_basis_usdc, entry_fee_usdc, opened_at_iso, signal_key,
                    strategy_source, stop_loss_price, take_profit_price, status,
                    closed_at_iso, exit_price, exit_fee_usdc, realized_pnl_usdc, close_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pos.position_id,
                    track,
                    pos.symbol,
                    f"{pos.quantity:.8f}",
                    f"{pos.entry_price:.8f}",
                    f"{pos.cost_basis_usdc:.8f}",
                    f"{pos.entry_fee_usdc:.8f}",
                    pos.opened_at_iso,
                    pos.signal_key,
                    pos.strategy_source,
                    f"{pos.stop_loss_price:.8f}" if pos.stop_loss_price is not None else None,
                    f"{pos.take_profit_price:.8f}" if pos.take_profit_price is not None else None,
                    pos.status,
                    pos.closed_at_iso,
                    f"{pos.exit_price:.8f}" if pos.exit_price is not None else None,
                    f"{pos.exit_fee_usdc:.8f}",
                    f"{pos.realized_pnl_usdc:.8f}",
                    pos.close_reason,
                ),
            )

    def update_position_closed(
        self,
        position_id: str,
        *,
        closed_at_iso: str,
        closed_istanbul_date: str,
        exit_price: Decimal,
        exit_fee_usdc: Decimal,
        realized_pnl_usdc: Decimal,
        close_reason: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE positions
                SET status = 'CLOSED',
                    closed_at_iso = ?,
                    closed_istanbul_date = ?,
                    exit_price = ?,
                    exit_fee_usdc = ?,
                    realized_pnl_usdc = ?,
                    close_reason = ?
                WHERE position_id = ?
                """,
                (
                    closed_at_iso,
                    closed_istanbul_date,
                    f"{exit_price:.8f}",
                    f"{exit_fee_usdc:.8f}",
                    f"{realized_pnl_usdc:.8f}",
                    close_reason,
                    position_id,
                ),
            )

    def record_order(
        self, order: SimulatedOrderResult, istanbul_date: str, track: str = "MAIN"
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO orders (
                    order_id, portfolio_track, symbol, side, order_type, status,
                    requested_qty, filled_qty, avg_fill_price, gross_quote_usdc,
                    fee_usdc, slippage_bps, signal_key, strategy_source,
                    explanation, realized_pnl_usdc, istanbul_date, created_at_iso
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order.order_id,
                    track,
                    order.symbol,
                    order.side,
                    order.order_type,
                    order.status,
                    f"{order.requested_qty:.8f}",
                    f"{order.filled_qty:.8f}",
                    f"{order.avg_fill_price:.8f}",
                    f"{order.gross_quote_usdc:.8f}",
                    f"{order.fee_usdc:.8f}",
                    f"{order.slippage_bps:.4f}",
                    order.signal_key,
                    order.strategy_source,
                    order.explanation,
                    f"{order.realized_pnl_usdc:.8f}",
                    istanbul_date,
                    order.created_at_iso,
                ),
            )

    def get_orders(self, track: str = "MAIN", limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM orders WHERE portfolio_track = ? ORDER BY created_at_iso DESC LIMIT ?",
                (track, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def count_daily_new_positions(self, istanbul_date: str, track: str = "MAIN") -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM orders
                WHERE portfolio_track = ?
                  AND istanbul_date = ?
                  AND side = 'BUY'
                  AND status IN ('FILLED', 'PARTIALLY_FILLED')
                """,
                (track, istanbul_date),
            ).fetchone()
            return int(row["cnt"]) if row else 0

    def get_daily_realized_pnl_and_entry_fees(
        self, istanbul_date: str, track: str = "MAIN"
    ) -> tuple[Decimal, Decimal]:
        """
        Belirtilen Europe/Istanbul gününde:
        1) Kapanmış pozisyonların net gerçekleşmiş kâr/zararı (giriş+çıkış komisyonu dahil),
        2) Bugün açılıp hâlâ açık olan pozisyonların ödenmiş giriş komisyonları.
        """
        with self._connect() as conn:
            closed_rows = conn.execute(
                """
                SELECT realized_pnl_usdc
                FROM positions
                WHERE portfolio_track = ?
                  AND status = 'CLOSED'
                  AND closed_istanbul_date = ?
                """,
                (track, istanbul_date),
            ).fetchall()
            realized_today = sum(
                (Decimal(str(r["realized_pnl_usdc"])) for r in closed_rows), Decimal("0")
            )

            open_rows = conn.execute(
                """
                SELECT opened_at_iso, entry_fee_usdc
                FROM positions
                WHERE portfolio_track = ? AND status = 'OPEN'
                """,
                (track,),
            ).fetchall()
            open_entry_fees_today = Decimal("0")
            for r in open_rows:
                opened_dt = datetime.fromisoformat(str(r["opened_at_iso"]))
                if self.istanbul_date_str(opened_dt) == istanbul_date:
                    open_entry_fees_today += Decimal(str(r["entry_fee_usdc"]))

            return realized_today, open_entry_fees_today

    def get_total_realized_pnl_and_fees(self, track: str = "MAIN") -> tuple[Decimal, Decimal]:
        with self._connect() as conn:
            pos_rows = conn.execute(
                "SELECT realized_pnl_usdc FROM positions WHERE portfolio_track = ? AND status = 'CLOSED'",
                (track,),
            ).fetchall()
            total_realized = sum(
                (Decimal(str(r["realized_pnl_usdc"])) for r in pos_rows), Decimal("0")
            )

            fee_rows = conn.execute(
                """
                SELECT fee_usdc FROM orders
                WHERE portfolio_track = ? AND status IN ('FILLED', 'PARTIALLY_FILLED')
                """,
                (track,),
            ).fetchall()
            total_fees = sum((Decimal(str(r["fee_usdc"])) for r in fee_rows), Decimal("0"))
            return total_realized, total_fees

    def record_decision(
        self,
        *,
        created_at_iso: str,
        istanbul_date: str,
        symbol: str,
        interval: str,
        candle_open_time_ms: int,
        rule_action: str,
        rule_reason: str,
        ai_action: str,
        ai_reason: str,
        final_action: str,
        mode_label: str,
        risk_notes: list[str],
        self_reported_score: int | None,
        executed_order_id: str | None = None,
        block_reason: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO decisions (
                    created_at_iso, istanbul_date, symbol, interval, candle_open_time_ms,
                    rule_action, rule_reason, ai_action, ai_reason, final_action,
                    mode_label, risk_notes_json, self_reported_score, executed_order_id, block_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at_iso,
                    istanbul_date,
                    symbol,
                    interval,
                    candle_open_time_ms,
                    rule_action,
                    rule_reason,
                    ai_action,
                    ai_reason,
                    final_action,
                    mode_label,
                    json.dumps(risk_notes, ensure_ascii=False),
                    self_reported_score,
                    executed_order_id,
                    block_reason,
                ),
            )

    def get_recent_decisions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_cached_ai_response(self, cache_key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT response_json, model_name
                FROM ai_usage
                WHERE cache_key = ? AND status = 'OK' AND response_json IS NOT NULL
                ORDER BY id DESC LIMIT 1
                """,
                (cache_key,),
            ).fetchone()
            if not row or not row["response_json"]:
                return None
            try:
                return json.loads(str(row["response_json"]))
            except Exception:
                return None

    def count_daily_ai_network_calls(self, istanbul_date: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM ai_usage
                WHERE istanbul_date = ? AND from_cache = 0
                """,
                (istanbul_date,),
            ).fetchone()
            return int(row["cnt"]) if row else 0

    def get_last_ai_network_call_time(self) -> datetime | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT created_at_iso
                FROM ai_usage
                WHERE from_cache = 0
                ORDER BY id DESC LIMIT 1
                """
            ).fetchone()
            if not row:
                return None
            return datetime.fromisoformat(str(row["created_at_iso"]))

    def record_ai_usage(
        self,
        *,
        created_at_iso: str,
        istanbul_date: str,
        cache_key: str,
        model_name: str,
        from_cache: bool,
        prompt_tokens: int,
        output_tokens: int,
        estimated_cost_usd: Decimal,
        response_payload: dict[str, Any] | None,
        status: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ai_usage (
                    created_at_iso, istanbul_date, cache_key, model_name,
                    from_cache, prompt_tokens, output_tokens, estimated_cost_usd,
                    response_json, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at_iso,
                    istanbul_date,
                    cache_key,
                    model_name,
                    1 if from_cache else 0,
                    prompt_tokens,
                    output_tokens,
                    f"{estimated_cost_usd:.8f}",
                    json.dumps(response_payload, ensure_ascii=False) if response_payload else None,
                    status,
                ),
            )

    def get_ai_usage_summary(self, istanbul_date: str) -> dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM ai_usage").fetchall()
            total_calls = sum(1 for r in rows if int(r["from_cache"]) == 0)
            cache_hits = sum(1 for r in rows if int(r["from_cache"]) == 1)
            today_calls = sum(
                1
                for r in rows
                if int(r["from_cache"]) == 0 and str(r["istanbul_date"]) == istanbul_date
            )
            prompt_tokens = sum(int(r["prompt_tokens"]) for r in rows)
            output_tokens = sum(int(r["output_tokens"]) for r in rows)
            est_cost = sum((Decimal(str(r["estimated_cost_usd"])) for r in rows), Decimal("0"))
            return {
                "total_network_calls": total_calls,
                "cache_hits": cache_hits,
                "today_network_calls": today_calls,
                "total_prompt_tokens": prompt_tokens,
                "total_output_tokens": output_tokens,
                "estimated_cost_usd": est_cost,
            }

    def record_equity_snapshot(
        self,
        *,
        created_at_iso: str,
        istanbul_date: str,
        symbol: str,
        main_equity_usdc: Decimal,
        cash_balance_usdc: Decimal,
        unrealized_pnl_usdc: Decimal,
        rule_shadow_equity_usdc: Decimal,
        benchmark_equity_usdc: Decimal,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO equity_snapshots (
                    created_at_iso, istanbul_date, symbol, main_equity_usdc,
                    cash_balance_usdc, unrealized_pnl_usdc,
                    rule_shadow_equity_usdc, benchmark_equity_usdc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at_iso,
                    istanbul_date,
                    symbol,
                    f"{main_equity_usdc:.8f}",
                    f"{cash_balance_usdc:.8f}",
                    f"{unrealized_pnl_usdc:.8f}",
                    f"{rule_shadow_equity_usdc:.8f}",
                    f"{benchmark_equity_usdc:.8f}",
                ),
            )

    def get_equity_snapshots(self, limit: int = 500) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM (SELECT * FROM equity_snapshots ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def export_table_to_csv_bytes(self, table_name: str) -> bytes:
        """Tabloyu Türkçe Excel uyumlu UTF-8 BOM (utf-8-sig) CSV bayt dizisine aktarır."""
        allowed = {"orders", "positions", "decisions", "equity_snapshots", "ai_usage"}
        if table_name not in allowed:
            raise ValueError(f"Dışa aktarma için geçersiz tablo: {table_name}")

        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM {table_name} ORDER BY 1 DESC").fetchall()

        buf = io.StringIO()
        if rows:
            headers = list(rows[0].keys())
            writer = csv.DictWriter(buf, fieldnames=headers)
            writer.writeheader()
            for r in rows:
                writer.writerow(dict(r))
        else:
            buf.write("bilgi\nKayit bulunamadi\n")

        return buf.getvalue().encode("utf-8-sig")
