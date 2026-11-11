@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================================
echo   BINANCE GLOBAL USDC SPOT SISTEMI - WINDOWS KURULUM ARACI
echo   (Yonetici yetkisi gerektirmez; Guvenlik Duvari/Antivirus kapatilmaz)
echo ============================================================================
echo.

where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [HATA] Python bulunamadi! Lutfen python.org uzerinden Python 3.11 veya 3.12 kurun
    echo        ve kurulum sirasinda "Add Python to PATH" kutusunu isaretleyin.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Yerel Python sanal ortami (.venv) olusturuluyor...
    python -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo [HATA] Sanal ortam olusturulamadi.
        pause
        exit /b 1
    )
) else (
    echo [1/4] Mevcut sanal ortam (.venv) bulundu.
)

echo [2/4] Sabitlenmis bagimliliklar (requirements.txt) yukleniyor...
.\.venv\Scripts\python.exe -m pip install --disable-pip-version-check -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [HATA] Paket kurulumu sirasinda sorun olustu. Internet baglantinizi kontrol edin.
    pause
    exit /b 1
)

if not exist "data" mkdir data

if not exist ".env" (
    echo [3/4] .env.example dosyasindan yerel .env ayar dosyasi olusturuluyor...
    copy ".env.example" ".env" >nul
) else (
    echo [3/4] Mevcut .env dosyaniz korundu ^(uzerine yazilmadi^).
)

echo [4/4] Guvenlik, filtre ve simulasyon dogrulama testleri calistiriliyor...
.\.venv\Scripts\pytest.exe
if %ERRORLEVEL% NEQ 0 (
    echo [UYARI] Bazi testler basarisiz oldu. Lutfen SORUN_GIDERME.md dosyasini inceleyin.
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo   KURULUM BASARIYLA TAMAMLANDI!
echo   Sistemi acmak icin "baslat.bat" dosyasina cift tiklayabilirsiniz.
echo ============================================================================
pause
