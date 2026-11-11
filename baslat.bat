@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [UYARI] Sanal ortam ^(.venv^) bulunamadi. Once "kurulum.bat" calistiriliyor...
    call kurulum.bat
    if %ERRORLEVEL% NEQ 0 exit /b 1
)

if not exist ".env" (
    copy ".env.example" ".env" >nul
)

REM Eger 8501 portu zaten aciksa dogrudan tarayiciyi ac
netstat -ano | findstr "127.0.0.1:8501" | findstr "LISTENING" >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Panel zaten calisiyor! Tarayicida http://127.0.0.1:8501 aciliyor...
    start "" "http://127.0.0.1:8501"
    exit /b 0
)

echo ============================================================================
echo   BINANCE USDC SPOT YEREL KONTROL PANELI BASLATILIYOR
echo   Adres : http://127.0.0.1:8501 ^(Yalnizca kendi bilgisayarinizdan erisilebilir^)
echo   ONEMLI: Bu siyah pencere acik kaldigi surece panel calisir.
echo           Bu pencereyi kapattiginizda veya bilgisayar uykuya gectiginde bot DURUR.
echo ============================================================================
echo.

REM Sunucu ayaga kalktiktan 3 saniye sonra varsayilan tarayiciyi otomatik ac
start "" /b cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:8501"

.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
pause
