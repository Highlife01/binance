"""Reproduce a public-data Spot baseline without account keys or order APIs.

Run with --fetch once, then reuse the exact cached inputs without that flag.
Each interval has 1000 bars: different calendar durations must not be ranked
as directly comparable performance windows.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import sys
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from binance_ai_bot.domain import RiskSettings
from binance_ai_bot.market_data import extract_closed_candles, parse_symbol_filters
from binance_ai_bot.profiles import TRADING_PROFILES
from binance_ai_bot.validation import run_chronological_validation


def public_json(path, params):
    url = "https://data-api.binance.vision/api/v3/" + path + "?" + urlencode(params)
    with urlopen(url, timeout=20) as response:
        return json.load(response)


def compact(metrics):
    return {key: value for key, value in asdict(metrics).items()
            if key not in {"equity_curve", "trade_pnls_usdc"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Download public OHLCV and current filters")
    parser.add_argument("--as-of", default="2026-11-15T00:00:00Z")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "profitability.json")
    args = parser.parse_args()
    asof = datetime.fromisoformat(args.as_of.replace("Z", "+00:00"))
    if asof.tzinfo is None:
        parser.error("--as-of must include a timezone")
    asof_ms = int(asof.timestamp() * 1000)
    cache = ROOT / "reports" / "market_data" / asof.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cache.mkdir(parents=True, exist_ok=True)
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    intervals = ("1m", "5m", "15m")

    def cached(filename, path, params):
        target = cache / filename
        if args.fetch:
            data = public_json(path, params)
            target.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        elif target.exists():
            data = json.loads(target.read_text(encoding="utf-8"))
        else:
            raise SystemExit(f"Missing {target}; run once with --fetch")
        return data

    info = cached("exchangeInfo.json", "exchangeInfo", {"symbols": json.dumps(symbols, separators=(",", ":"))})
    filters = {item["symbol"]: parse_symbol_filters(item) for item in info["symbols"]}
    base = RiskSettings(
        max_open_positions=1, max_position_size_usdc=Decimal("15"),
        min_buy_notional_usdc=Decimal("10"), daily_loss_limit_usdc=Decimal("2"),
        estimated_fee_rate=Decimal("0.001"), extra_slippage_bps=Decimal("2"), leverage=1,
    )
    results = []
    for symbol in symbols:
        for interval in intervals:
            raw = cached(f"{symbol}_{interval}.json", "klines", {
                "symbol": symbol, "interval": interval, "limit": 1000, "endTime": asof_ms - 1,
            })
            candles = extract_closed_candles(raw, asof_ms)
            for profile in TRADING_PROFILES.values():
                if profile.interval != interval:
                    continue
                settings = profile.apply_to_risk_settings(base, leverage=1)
                # Explicitly scope this to the simulator's implemented behavior.
                settings = replace(settings, max_open_positions=1, expert_mode=False,
                                   min_net_profit_exit_pct=Decimal("0"),
                                   trailing_stop_pct=Decimal("0"), risk_per_trade_pct=Decimal("0"))
                report = run_chronological_validation(
                    candles, symbol=symbol, interval=interval, settings=settings,
                    initial_balance_usdc=Decimal("86"), strategy_params=profile.strategy,
                    symbol_filters=filters[symbol],
                )
                results.append({
                    "symbol": symbol, "profile": profile.key, "interval": interval,
                    "total_candles": len(candles), "settings": asdict(settings),
                    "fixed_strategy": asdict(profile.strategy),
                    "train": compact(report.train_metrics),
                    "validation": compact(report.validation_metrics),
                    "test": compact(report.test_metrics), "verdict": report.overall_verdict,
                })
                test = report.test_metrics
                print(f"{symbol} {profile.key}: test {test.net_return_usdc:+.4f} USDT, "
                      f"trades={test.trade_count}, insufficient={test.insufficient_data}")

    output = {
        "as_of_utc": asof.astimezone(timezone.utc).isoformat(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "public_source": "https://data-api.binance.vision/api/v3/klines",
        "scope": "Single-symbol, unleveraged OHLC Spot baseline; fixed profile parameters",
        "assumptions": {"initial_balance_usdt": 86, "max_position_usdt": 15,
                        "min_buy_notional_usdt": 10, "fee_per_side_pct": 0.10,
                        "half_spread_bps": 1.5, "extra_slippage_bps_per_side": 2,
                        "daily_loss_limit_usdt": 2, "bars_per_interval": 1000},
        "limitations": [
            "1000 bars cover different durations for each interval; no cross-interval winner is selected.",
            "A 200-bar test slice and even 15 trades cannot establish durable profitability.",
            "Expert/HTF/quant filters, ATR sizing, profit waiting, trailing exits and concurrent portfolios are not simulated.",
            "Current exchange filters may differ from the historical filters; modeled costs are not account fee quotes.",
            "No account balance, API key, live order or production database is accessed.",
        ],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, default=str, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
