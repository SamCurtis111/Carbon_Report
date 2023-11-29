call C:\Users\SamCurtis.AzureAD\anaconda3\Scripts\activate.bat
jupyter nbconvert --execute --to html "C:\GitHub\Carbon_Report\Position_Summary.ipynb" --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True
pause