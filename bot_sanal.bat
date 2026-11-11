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

set PARITE=%1
if "%PARITE%"=="" set PARITE=BTCUSDT

echo ============================================================================
echo   SANAL DENEME - KESINTISIZ ARKA PLAN BOTU (AKTIF PROFIL, 5dk)
echo   Parite : %PARITE%   (Gercek bakiyenize DOKUNULMAZ)
echo   Durdurmak icin: Ctrl+C veya bu pencereyi kapatin.
echo   Kayitlar: logs\bot_YYYY-MM-DD.log  ^|  Izleme: http://127.0.0.1:8501
echo ============================================================================
echo.

.\.venv\Scripts\python.exe bot_calistir.py --mod sanal --profil aktif --parite %PARITE%
pause
