@echo off

echo Starting automated model retraining...
echo Timestamp: %date% %time%

REM Change to the Django project directory
cd /d "C:\Users\dagem\Desktop\Fullstack\Internship tasks\House price-auto train\prediction"

REM Activate virtual environment
call ..\venv\Scripts\activate.bat

REM Run the Django management command
python manage.py retrain_model --samples=50

REM Check if the command was successful
if %errorlevel% equ 0 (
    echo Model retraining completed successfully!
) else (
    echo Model retraining failed with error code %errorlevel%
)

REM Deactivate virtual environment
call deactivate

echo Automated retraining finished at %date% %time%
pause
