call "C:\Users\sbcur\anaconda3\Scripts\activate.bat"

:: Execute the Jupyter Notebook
jupyter nbconvert --execute --to html "C:\GitHub\Carbon_Report\RGGI_Position_Report.ipynb" --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True

:: Pause to keep the console window open
pause