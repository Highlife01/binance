@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\pytest.exe" (
    echo [HATA] Test araci bulunamadi. Lutfen once "kurulum.bat" dosyasini calistirin.
    pause
    exit /b 1
)

echo Dogrulama test paketi calistiriliyor...
.\.venv\Scripts\pytest.exe
echo.
pause
