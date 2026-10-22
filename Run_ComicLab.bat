@echo off
title ComicLab Studio Launcher
echo ========================================================
echo   ComicLab Studio - Windows Desktop Application
echo ========================================================
echo.
set APP_EXE="%~dp0dist\ComicLab\ComicLab.exe"

if exist %APP_EXE% (
    echo [START] Ishga tushirilmoqda...
    start "" %APP_EXE%
) else (
    echo [XATO] Dastur fayli topilmadi: %APP_EXE%
    pause
)
