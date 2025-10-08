@echo off
echo =====================================================
echo  Google Earth Engine Authentication Fix
echo =====================================================
echo.
echo The "Invalid JWT Signature" error is caused by your
echo system clock being out of sync with internet time.
echo.
echo This script will attempt to fix the issue.
echo.

echo Step 1: Attempting to sync system time...
echo (This may require Administrator privileges)
w32tm /resync
if %errorlevel% neq 0 (
    echo.
    echo Administrator privileges required for automatic sync.
    echo Please manually sync your clock:
    echo.
    echo 1. Right-click on the Windows clock
    echo 2. Select "Adjust date/time"
    echo 3. Turn ON "Set time automatically"
    echo 4. Click "Sync now"
    echo 5. Close the settings window
    echo.
    echo Press any key to continue after syncing manually...
    pause >nul
) else (
    echo Time sync completed successfully!
    echo Waiting 5 seconds for sync to take effect...
    timeout /t 5 /nobreak >nul
)

echo.
echo Step 2: Verifying time synchronization...
python -c "
import datetime
local = datetime.datetime.now()
utc = datetime.datetime.utcnow()
diff = abs((local - utc).total_seconds())
print(f'Local time: {local}')
print(f'UTC time: {utc}')
print(f'Time difference: {diff:.1f} seconds')
if diff > 300:
    print('❌ Clock is still not synchronized properly')
    print('Please manually sync your clock and try again')
else:
    print('✅ Clock appears to be synchronized')
    print('You can now restart your application')
"

echo.
echo =====================================================
echo Fix complete! Please restart your FastAPI application.
echo =====================================================
echo.
pause