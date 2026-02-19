# ═══════════════════════════════════════════════════════════════════════
#  Deploy Legal Intake Validator to Azure Container Apps
# ═══════════════════════════════════════════════════════════════════════
#
#  Prerequisites:
#    1. Azure CLI installed:  winget install Microsoft.AzureCLI
#    2. Docker Desktop running
#    3. Logged into Azure:    az login
#    4. .env file with your Azure OpenAI credentials
#
#  Usage:
#    .\deploy-azure.ps1
#
# ═══════════════════════════════════════════════════════════════════════

$ErrorActionPreference = "Stop"

# ── Configuration (edit these) ────────────────────────────────────────
$RESOURCE_GROUP   = "rg-legal-intake"
$LOCATION         = "eastus2"
$ACR_NAME         = "legaIntakeAcr"           # Azure Container Registry (must be globally unique, lowercase, no hyphens)
$APP_NAME         = "legal-intake-validator"
$ENV_NAME         = "legal-intake-env"
$IMAGE_NAME       = "legal-intake-validator"
$IMAGE_TAG        = "latest"

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Deploying Legal Intake Validator to Azure Container Apps" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Create Resource Group ─────────────────────────────────────
Write-Host "[1/6] Creating resource group: $RESOURCE_GROUP ..." -ForegroundColor Yellow
az group create --name $RESOURCE_GROUP --location $LOCATION --output none
Write-Host "  Done." -ForegroundColor Green

# ── Step 2: Create Azure Container Registry ──────────────────────────
Write-Host "[2/6] Creating Azure Container Registry: $ACR_NAME ..." -ForegroundColor Yellow
az acr create `
    --resource-group $RESOURCE_GROUP `
    --name $ACR_NAME `
    --sku Basic `
    --admin-enabled true `
    --output none
Write-Host "  Done." -ForegroundColor Green

# Get ACR login server
$ACR_SERVER = az acr show --name $ACR_NAME --query loginServer --output tsv
Write-Host "  ACR Server: $ACR_SERVER"

# ── Step 3: Build & Push Docker Image ─────────────────────────────────
Write-Host "[3/6] Building and pushing Docker image ..." -ForegroundColor Yellow
az acr build `
    --registry $ACR_NAME `
    --image "${IMAGE_NAME}:${IMAGE_TAG}" `
    --file Dockerfile `
    .
Write-Host "  Done." -ForegroundColor Green

# ── Step 4: Create Container Apps Environment ─────────────────────────
Write-Host "[4/6] Creating Container Apps environment: $ENV_NAME ..." -ForegroundColor Yellow
az containerapp env create `
    --name $ENV_NAME `
    --resource-group $RESOURCE_GROUP `
    --location $LOCATION `
    --output none
Write-Host "  Done." -ForegroundColor Green

# ── Step 5: Read secrets from .env ────────────────────────────────────
Write-Host "[5/6] Reading secrets from .env ..." -ForegroundColor Yellow

# Parse .env file into a hashtable
$envVars = @{}
Get-Content .env | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith("#")) {
        $parts = $line -split "=", 2
        if ($parts.Count -eq 2) {
            $envVars[$parts[0].Trim()] = $parts[1].Trim()
        }
    }
}

# Build the --env-vars string for Azure Container Apps
$envArgs = @()
foreach ($key in $envVars.Keys) {
    $val = $envVars[$key]
    $envArgs += "$key=$val"
}
$envString = $envArgs -join " "
Write-Host "  Loaded $($envVars.Count) environment variables." -ForegroundColor Green

# ── Step 6: Deploy Container App ──────────────────────────────────────
Write-Host "[6/6] Deploying container app: $APP_NAME ..." -ForegroundColor Yellow

# Get ACR credentials
$ACR_USER = az acr credential show --name $ACR_NAME --query username --output tsv
$ACR_PASS = az acr credential show --name $ACR_NAME --query "passwords[0].value" --output tsv

az containerapp create `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --environment $ENV_NAME `
    --image "${ACR_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}" `
    --registry-server $ACR_SERVER `
    --registry-username $ACR_USER `
    --registry-password $ACR_PASS `
    --target-port 8501 `
    --ingress external `
    --min-replicas 0 `
    --max-replicas 3 `
    --cpu 1.0 `
    --memory 2.0Gi `
    --env-vars $envArgs `
    --output none

Write-Host "  Done." -ForegroundColor Green

# ── Get the app URL ───────────────────────────────────────────────────
$APP_URL = az containerapp show `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --query "properties.configuration.ingress.fqdn" `
    --output tsv

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "  DEPLOYMENT COMPLETE" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "  App URL: https://$APP_URL" -ForegroundColor Cyan
Write-Host ""
Write-Host "  To update later:" -ForegroundColor Yellow
Write-Host "    az acr build --registry $ACR_NAME --image ${IMAGE_NAME}:${IMAGE_TAG} ."
Write-Host "    az containerapp update --name $APP_NAME --resource-group $RESOURCE_GROUP --image ${ACR_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}"
Write-Host ""
Write-Host "  To tear down:" -ForegroundColor Yellow
Write-Host "    az group delete --name $RESOURCE_GROUP --yes --no-wait"
Write-Host ""
