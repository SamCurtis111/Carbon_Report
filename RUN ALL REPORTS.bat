call C:\Users\SamCurtis.AzureAD\anaconda3\Scripts\activate.bat
jupyter nbconvert --execute --to html "C:\GitHub\Carbon_Report\ACCU_Position_Report.ipynb" --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True
jupyter nbconvert --execute --to html "C:\GitHub\Carbon_Report\EUA_Position_Report.ipynb" --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True
jupyter nbconvert --execute --to html "C:\GitHub\Carbon_Report\NZU_Position_Report.ipynb" --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True
jupyter nbconvert --execute --to html "C:\GitHub\Carbon_Report\Position_Summary.ipynb" --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True
pause