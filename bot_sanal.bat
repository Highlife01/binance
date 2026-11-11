@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

if not exist ".venv\Scripts\python.exe" (
    echo [UYARI] Sanal ortam bulunamadi. Once "kurulum.bat" calistiriliyor...
    call kurulum.bat
    if %ERRORLEVEL% NEQ 0 exit /b 1
)

echo ============================================================================
echo   SANAL DENEME - KESINTISIZ ARKA PLAN BOTU (TEMKINLI PROFIL, 15dk)
echo   10 coin taranir: BTC ETH SOL BNB XRP ADA AVAX DOGE LINK SUI (USDT)
echo   Ayni anda tek pozisyon, islem basina en fazla 10 USDT (SANAL - gercek bakiyeye DOKUNULMAZ)
echo   Durdurmak icin: Ctrl+C veya bu pencereyi kapatin.
echo   Kayitlar: logs\bot_YYYY-MM-DD.log  ^|  Izleme: http://127.0.0.1:8501
echo ============================================================================
echo.

.\.venv\Scripts\python.exe bot_calistir.py --mod sanal --profil temkinli --pariteler varsayilan
pause
