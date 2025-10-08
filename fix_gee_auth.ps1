# Google Earth Engine Authentication Fix Script
# This script helps fix the JWT signature error by synchronizing system time

Write-Host "=====================================================" -ForegroundColor Green
Write-Host " Google Earth Engine Authentication Fix" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Green
Write-Host ""
Write-Host "The 'Invalid JWT Signature' error is caused by your" -ForegroundColor Yellow
Write-Host "system clock being out of sync with internet time." -ForegroundColor Yellow
Write-Host ""

Write-Host "Step 1: Checking current time synchronization..." -ForegroundColor Cyan

# Check current time sync status
python -c "
import datetime
local = datetime.datetime.now()
utc = datetime.datetime.utcnow()
diff = abs((local - utc).total_seconds())
print(f'Local time: {local}')
print(f'UTC time: {utc}')
print(f'Time difference: {diff:.1f} seconds')
if diff > 300:
    print('❌ Clock is NOT synchronized (difference > 5 minutes)')
    print('This WILL cause JWT signature errors')
else:
    print('✅ Clock appears to be synchronized')
"

Write-Host ""
Write-Host "Step 2: Attempting to sync system time..." -ForegroundColor Cyan

try {
    # Try to sync time (requires admin privileges)
    $result = w32tm /resync 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Time sync completed successfully!" -ForegroundColor Green
        Start-Sleep -Seconds 3
    } else {
        throw "Access denied or sync failed"
    }
} catch {
    Write-Host "⚠️  Administrator privileges required for automatic sync." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "MANUAL SYNC INSTRUCTIONS:" -ForegroundColor Red
    Write-Host "1. Right-click on the Windows clock (bottom-right corner)" -ForegroundColor White
    Write-Host "2. Select 'Adjust date/time'" -ForegroundColor White
    Write-Host "3. Turn ON 'Set time automatically'" -ForegroundColor White
    Write-Host "4. Click 'Sync now' button" -ForegroundColor White
    Write-Host "5. Close the settings window" -ForegroundColor White
    Write-Host ""
    Read-Host "Press Enter after you've synced the clock manually"
}

Write-Host ""
Write-Host "Step 3: Verifying time synchronization..." -ForegroundColor Cyan

python -c "
import datetime
local = datetime.datetime.now()
utc = datetime.datetime.utcnow()
diff = abs((local - utc).total_seconds())
print(f'After sync - Local time: {local}')
print(f'After sync - UTC time: {utc}')
print(f'After sync - Time difference: {diff:.1f} seconds')
if diff > 300:
    print('❌ Clock is still not synchronized properly')
    print('Please try the manual sync steps again')
    exit(1)
else:
    print('✅ Clock is now synchronized!')
    print('You can restart your FastAPI application')
"

Write-Host ""
Write-Host "=====================================================" -ForegroundColor Green
Write-Host "Fix complete! Your clock should now be synchronized." -ForegroundColor Green
Write-Host "Please restart your FastAPI application:" -ForegroundColor Green
Write-Host "uvicorn app:app --reload --host 0.0.0.0 --port 8000" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Green
Write-Host ""