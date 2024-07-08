@echo off
setlocal

:: Set the Anaconda path based on the current user
set "ANACONDA_PATH=C:\Users\%USERNAME%\anaconda3\Scripts\activate.bat"

:: Activate the Anaconda environment
call "%ANACONDA_PATH%"

:: Execute the Jupyter Notebook
jupyter nbconvert --execute --to html "C:\GitHub\Carbon_Report\EUA_Position_Report.ipynb" --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True

:: Pause to keep the console window open
pause