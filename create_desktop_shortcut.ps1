# ComicLab Desktop Shortcut Creator
$wshShell = New-Object -ComObject WScript.Shell
$desktopDir = [System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::Desktop)
$appDir = (Resolve-Path (Join-Path $PSScriptRoot "dist\ComicLab")).Path
$targetExe = Join-Path $appDir "ComicLab.exe"
$iconPath = (Join-Path $PSScriptRoot "assets\app_icon.ico")
$shortcutPath = Join-Path $desktopDir "Comic-Lab Studio.lnk"

if (Test-Path $targetExe) {
    $shortcut = $wshShell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $targetExe
    $shortcut.WorkingDirectory = $appDir
    if (Test-Path $iconPath) {
        $shortcut.IconLocation = "$iconPath,0"
    }
    $shortcut.Description = "Comic-Lab Ultra Studio Desktop Application"
    $shortcut.Save()
    Write-Host "[OK] Desktop shortcut created at $shortcutPath"
} else {
    Write-Host "[ERROR] Target executable not found at $targetExe"
}
