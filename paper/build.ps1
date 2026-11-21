# Build paper/main.pdf with the local TinyTeX (tools/ is git-ignored; see CONTEXT.md).
# Usage (from the repository root):  powershell -File paper\build.ps1
$ErrorActionPreference = "Continue"
$tex = Join-Path $PSScriptRoot "..\tools\tinytex\TinyTeX\bin\windows"
Push-Location $PSScriptRoot
& "$tex\pdflatex.exe" -interaction=nonstopmode main.tex *> build.log
& "$tex\bibtex.exe" main *>> build.log
& "$tex\pdflatex.exe" -interaction=nonstopmode main.tex *>> build.log
& "$tex\pdflatex.exe" -interaction=nonstopmode main.tex *>> build.log
Select-String -Path main.log -Pattern "^!|LaTeX Warning: (Citation|Reference).*undefined|Output written" |
    ForEach-Object { $_.Line } | Select-Object -Unique
Select-String -Path main.blg -Pattern "^Warning|error" | ForEach-Object { "bibtex: " + $_.Line }
Pop-Location
