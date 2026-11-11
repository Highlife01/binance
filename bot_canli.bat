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
echo   !!! GERCEK PARA - HIZLI AKTIF COKLU COIN BOTU (30 SN DONGU, 1DK MUM) !!!
echo   10 Coin Taranir: BTC ETH SOL BNB XRP ADA AVAX DOGE LINK SUI (USDT)
echo   Ayni anda en fazla 5 farkli coin (her biri maks 10 USDT) + 2.00 USDT gunluk zarar kilidi.
echo   Bilgisayar acik kaldigi surece calisir (uyku modu bot tarafindan engellenir).
echo   Durdurmak icin: Ctrl+C, bu pencereyi kapatmak veya paneldeki "Botu Durdur".
echo   Kayitlar: logs\bot_YYYY-MM-DD.log  ^|  Izleme: http://127.0.0.1:8501
echo ============================================================================
echo.

if "%1"=="" (
    .\.venv\Scripts\python.exe bot_calistir.py --mod canli --profil hizli_30sn --pariteler varsayilan
) else (
    .\.venv\Scripts\python.exe bot_calistir.py --mod canli --profil hizli_30sn --parite %1
)
pause
