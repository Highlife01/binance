@echo off
chcp 65001 >nul
setlocal

echo 127.0.0.1:8501 uzerinde calisan yerel panel araniyor...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr "127.0.0.1:8501" ^| findstr "LISTENING"') do (
    echo Panel sureci ^(PID: %%a^) kapatiliyor...
    taskkill /PID %%a /F >nul 2>nul
)
echo Yerel panel durduruldu.
pause
