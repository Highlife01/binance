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
echo   GERCEK PARA - 2x CAPRAZ MARJIN (CROSS MARGIN) 10 COIN AKTIF BOT
echo   ON KOSUL: Binance uygulamasinda Cuzdanlar -^> Marjin (Cross) hesabini acip
echo             Spot cuzdaninizdan Marjin cuzdanina USDT aktarmis olmalisiniz!
echo   2x Kaldirac: Islem basina 10 USDT teminat ile 20 USDT buyuklugunde islem,
echo                ayni anda en fazla 5 farkli coin, 2.00 USDT gunluk zarar kilidi.
echo   Baslamadan once konsolda buyuk harflerle EVET yazmaniz istenecektir.
echo ============================================================================
echo.

.\.venv\Scripts\python.exe bot_calistir.py --mod marjin --profil hizli_30sn --kaldirac 2 --pariteler varsayilan %*
pause
