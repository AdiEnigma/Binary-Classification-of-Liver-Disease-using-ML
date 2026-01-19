@echo off
echo ============================================================
echo Starting Liver Disease Prediction System
echo ============================================================
echo.
echo Starting Backend Server...
start "Backend Server" cmd /k "cd backend && python -m uvicorn main:app --reload --port 8000"
timeout /t 3 /nobreak >nul
echo.
echo Starting Frontend Server...
cd "Doctor-Friendly Liver Disease Dashboard"
start "Frontend Server" cmd /k "npm run dev"
echo.
echo ============================================================
echo Servers are starting...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:5173 (or check the terminal)
echo Backend API Docs: http://localhost:8000/docs
echo ============================================================
echo.
echo Press any key to close this window (servers will keep running)...
pause >nul
