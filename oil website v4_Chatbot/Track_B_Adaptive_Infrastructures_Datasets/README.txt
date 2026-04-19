Track B dataset package
=======================

Included directly in this zip:
- Global_Fuel_Prices_Database.xlsx
- Global_Fuel_Subsidies_and_Price_Control_Measures_Database.xlsx
- OWID_Energy_Data.csv

Also included:
- source_manifest.csv
- download_doeb_remaining.py
- download_doeb_remaining.ps1

Why the DOEB files are not already inside the zip:
- In this environment, direct downloads from data.doeb.go.th could not be completed because the host could not be resolved from the container.
- I still identified official DOEB direct-download URLs and put them into the manifest and download scripts.

How to get the DOEB files on your computer:
1) Extract this zip.
2) On Windows, right-click PowerShell and run download_doeb_remaining.ps1 inside the extracted folder.
   Or run: python download_doeb_remaining.py
3) The DOEB files will be saved into a folder named doeb_downloads next to the script.

Notes:
- source_manifest.csv lists every file and its source URL.
- The DOEB URLs are official dataset download links gathered from DOEB dataset pages/search results.
