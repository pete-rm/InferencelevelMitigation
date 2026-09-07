param(
    [string[]]$Categories = @("age", "appearance", "disability", "gender", "race", "religion")
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Project Python interpreter not found: $Python"
}

if (-not $env:OPENAI_API_KEY) {
    throw "OPENAI_API_KEY must be set before running Stage 1."
}

Remove-Item Env:MAX_CONVERSATIONS -ErrorAction SilentlyContinue
Remove-Item Env:MAX_ATTRIBUTION_CONTEXTS -ErrorAction SilentlyContinue
Remove-Item Env:MAX_STAGE3_TURNS -ErrorAction SilentlyContinue

Set-Location $ProjectRoot

foreach ($Category in $Categories) {
    $env:MITIGATION_CATEGORY = $Category
    Write-Host "`n================ $Category ================"

    foreach ($Stage in @("Stage1.py", "stage2.py", "stage3.py", "stage4.py")) {
        Write-Host "Running $Stage for $Category"
        & $Python (Join-Path $ProjectRoot "MitigationCode\$Stage")
        if ($LASTEXITCODE -ne 0) {
            throw "$Stage failed for category '$Category'. No later stage or category was run."
        }
    }
}

Remove-Item Env:MITIGATION_CATEGORY -ErrorAction SilentlyContinue
Write-Host "`nAll category pipelines completed. Results: results\mistral_categories"