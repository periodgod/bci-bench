# Build WYU-BCI-Bench-Explorer.exe (single file, no console).
#   powershell -ExecutionPolicy Bypass -File explorer\build_exe.ps1
# Uses an isolated venv (explorer\.build-venv) with pinned pyinstaller / pywebview.
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
$sysPy = "C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"
if (-not (Test-Path ".build-venv")) { & $sysPy -m venv .build-venv }
$py = Join-Path $here ".build-venv\Scripts\python.exe"
& $py -m pip install -q "pyinstaller==6.22.3" "pywebview==6.2.1"

# Offline snapshot, used only when the exe runs away from the bench source tree.
& $py explorer_app.py --dump catalog_snapshot.json

& $py -m PyInstaller --noconfirm --clean --onefile --windowed `
    --name "WYU-BCI-Bench-Explorer" `
    --add-data "$here\ui;ui" --add-data "$here\catalog_snapshot.json;." `
    --paths "$here" `
    --hidden-import build_catalog `
    --distpath dist --workpath build --specpath build `
    explorer_app.py

$exe = Join-Path $here "dist\WYU-BCI-Bench-Explorer.exe"
Write-Host "built: $exe"
