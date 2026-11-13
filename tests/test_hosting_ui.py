"""Contract and syntax tests for the TradeCebrail hosting dashboard UI."""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import tempfile

import pytest

INDEX_HTML = pathlib.Path(__file__).resolve().parents[1] / "hosting" / "tradecebrail" / "index.html"


def _load_html() -> tuple[str, str]:
    text = INDEX_HTML.read_text(encoding="utf-8")
    m = re.search(r'<script type="module">(.*?)</script>', text, re.S)
    assert m is not None, "Expected <script type='module'> block in index.html"
    return text, m.group(1)


def test_dashboard_dom_ids_and_required_controls() -> None:
    text, js = _load_html()
    html_part = text[: text.index('<script type="module">')]
    ids_all = set(re.findall(r'id=["\']([A-Za-z0-9_-]+)["\']', text))
    ids_static = set(re.findall(r'id=["\']([A-Za-z0-9_-]+)["\']', html_part))
    used_ids = set(re.findall(r'\$\(\s*["\']([A-Za-z0-9_-]+)["\']\s*\)', js))

    missing = sorted(used_ids - ids_all)
    assert not missing, f"JS references missing DOM IDs: {missing}"

    required_static = {
        "subtitleText",
        "aliveBadge",
        "pollToolbar",
        "pollIntervalSel",
        "refreshNowBtn",
        "userEmail",
        "logoutBtn",
        "authLoading",
        "authSection",
        "loginBtn",
        "authError",
        "appSection",
        "modeBanner",
        "lastErrorAlert",
        "toastAlert",
        "mEquity",
        "mCash",
        "mUnreal",
        "mRealized",
        "mDaily",
        "mLevInfo",
        "equityChartBox",
        "perfPanel",
        "perfSummary",
        "perfReasons",
        "perfFooter",
        "perfUniverse",
        "scanNowBtn",
        "toggleAutoBtn",
        "togglePauseBtn",
        "closeAllBtn",
        "selMode",
        "selProfile",
        "selLeverage",
        "inpPosSize",
        "inpMaxOpen",
        "inpDailyLoss",
        "selAi",
        "saveSettingsBtn",
        "resetPaperBtn",
        "heartbeatInfo",
        "checkWalletBtn",
        "walletContent",
        "openCount",
        "positionsBody",
        "ordersBody",
        "aiUsageText",
        "decisionsBody",
    }
    assert required_static <= ids_static

    for mode_val in ("PAPER", "LIVE_SPOT", "LIVE_MARGIN"):
        assert f'value="{mode_val}"' in html_part
    for prof_val in ("AL_YUKSELINCE_SAT", "HIZLI_30SN", "AKTIF", "TEMKINLI"):
        assert f'value="{prof_val}"' in html_part


def test_dashboard_optimizations_and_safety_features_present() -> None:
    text, js = _load_html()
    assert 'rel="icon" href="data:image/svg+xml' in text
    assert 'rel="preconnect" href="https://www.gstatic.com"' in text
    assert "AbortController" in js
    assert "visibilitychange" in js
    assert "inFlightDash" in js
    assert "SETTINGS_CONTROL_IDS" in js
    assert "/api/margin-liquidate" in js
    assert "btn-margin-liq" in js
    assert "renderWalletData" in js


def test_dashboard_inline_module_script_valid_js() -> None:
    if shutil.which("node") is None:
        pytest.skip("node executable not found on PATH")
    _, js = _load_html()
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False, encoding="utf-8") as tf:
        tf.write(js)
        tmp_path = pathlib.Path(tf.name)
    try:
        res = subprocess.run(["node", "--check", str(tmp_path)], capture_output=True, text=True)
        assert res.returncode == 0, f"node --check failed:\n{res.stderr}"
    finally:
        tmp_path.unlink(missing_ok=True)
