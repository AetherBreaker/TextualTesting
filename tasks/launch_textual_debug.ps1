param(
  [string]$CWDPath = (Get-Location).Path
)
Write-Host "Activating Python virtual environment in $CWDPath"
& "$CWDPath\.venv\Scripts\Activate.ps1"
Start-Process -FilePath "textual" -ArgumentList "serve", "--dev", ".\src\__main__.py"
Write-Host "Textual Web Server started in $CWDPath"

Start-Process -FilePath "textual" -ArgumentList "console"
Write-Host "Textual Console started in $CWDPath"

# wait 5 seconds to ensure the server is up
Start-Sleep -Seconds 1
Start-Process "http://localhost:8000"
Start-Sleep -Seconds 4