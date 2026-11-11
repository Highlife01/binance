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
echo   SANAL DENEME - HIZLI AKTIF BOT (30 SN DONGU, 1DK MUM, 10 COIN TARAMA)
echo   10 coin taranir: BTC ETH SOL BNB XRP ADA AVAX DOGE LINK SUI (USDT)
echo   Her 30 sn kontrol + Anlik Kar/Zarar takibi (SANAL - gercek bakiyeye DOKUNULMAZ)
echo   Durdurmak icin: Ctrl+C veya bu pencereyi kapatin.
echo   Kayitlar: logs\bot_YYYY-MM-DD.log  ^|  Izleme: http://127.0.0.1:8501
echo ============================================================================
echo.

.\.venv\Scripts\python.exe bot_calistir.py --mod sanal --profil hizli_30sn --pariteler varsayilan
pause
