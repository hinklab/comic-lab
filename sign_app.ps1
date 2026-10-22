# ComicLab Authenticode Digital Signature Script
param (
    [string]$TargetExe = "dist\ComicLab\ComicLab.exe"
)

$certSubject = "CN=ComicLab Studio, O=ComicLab, C=UZ"
Write-Host "[CERT] Qidirilmoqda: $certSubject"

$cert = Get-ChildItem -Path "Cert:\CurrentUser\My" -CodeSigningCert | Where-Object { $_.Subject -match "ComicLab" } | Select-Object -First 1

if (-not $cert) {
    Write-Host "[CERT] Yangi mualliflik sertifikati yaratilmoqda..."
    $cert = New-SelfSignedCertificate -Type CodeSigningCert -Subject $certSubject -CertStoreLocation "Cert:\CurrentUser\My" -NotAfter (Get-Date).AddYears(5)
    Write-Host "[CERT] Sertifikat 'Cert:\CurrentUser\My' do'koniga muvaffaqiyatli saqlandi."
} else {
    Write-Host "[CERT] Mavjud sertifikat topildi: $($cert.Thumbprint)"
}

$fullExePath = Resolve-Path $TargetExe -ErrorAction SilentlyContinue

if ($fullExePath -and (Test-Path $fullExePath)) {
    Write-Host "[SIGN] Imzolanmoqda: $fullExePath"
    $sig = Set-AuthenticodeSignature -FilePath $fullExePath -Certificate $cert
    Write-Host "[SIGN_STATUS] Holat: $($sig.Status)"
    $verify = Get-AuthenticodeSignature -FilePath $fullExePath
    Write-Host "[VERIFY] Dastur imzo ma'lumotlari:"
    Write-Host "  Noshir: $($verify.SignerCertificate.Subject)"
    Write-Host "  Algoritm: $($verify.SignerCertificate.SignatureAlgorithm.FriendlyName)"
    Write-Host "  Holat: $($verify.Status)"
} else {
    Write-Host "[ERROR] Imzolanuvchi fayl topilmadi: $TargetExe"
}
