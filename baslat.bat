@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [UYARI] Sanal ortam ^(.venv^) bulunamadi. Once "kurulum.bat" calistirilmalidir.
    pause
    exit /b 1
)

if not exist ".env" (
    copy ".env.example" ".env" >nul
)

echo ============================================================================
echo   BINANCE USDC SPOT YEREL KONTROL PANELI BASLATILIYOR
echo   Adres : http://127.0.0.1:8501 ^(Yalnizca kendi bilgisayarinizdan erisilebilir^)
echo   ONEMLI: Bilgisayar kapatildiginda veya uyku moduna gectiginde bot DURUR.
echo   Kapatmak icin bu pencereyi kapatabilir veya Ctrl+C tuslarina basabilirsiniz.
echo ============================================================================
echo.

.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
pause
