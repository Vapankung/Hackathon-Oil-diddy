$ErrorActionPreference = 'Continue'
$OutDir = Join-Path $PSScriptRoot 'doeb_downloads'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$files = @{
    'DOEB_Imported_Refined_Oil_Prices_2566.xlsx' = 'https://data.doeb.go.th/dataset/7aa42839-cfe4-4383-a1ed-f85fb42ff9ad/resource/515eb811-873b-4862-93b5-f90b5b6f68dc/download/34-2566.xlsx'
    'DOEB_Imported_Crude_Oil_Prices_2566.xlsx'   = 'https://data.doeb.go.th/dataset/d28d9622-91ad-468f-adaa-b275cf57d336/resource/e87efa41-6b36-413b-9728-b4dac3a022ee/download/35-2566.xlsx'
    'DOEB_Fuel_Supply_and_Distribution_Report_2565.pdf' = 'https://data.doeb.go.th/dataset/78091a3d-840a-4d68-8aa5-c0de1d70e9e1/resource/b6cbb7eb-9b17-4327-afb3-425425c3ae58/download/2.report_year-2565.pdf'
    'DOEB_Fuel_Supply_and_Distribution_Report_2564.pdf' = 'https://data.doeb.go.th/dataset/78091a3d-840a-4d68-8aa5-c0de1d70e9e1/resource/7d78321c-bcfd-4f8d-bde5-8f1879985b65/download/2.report_year-2564.pdf'
    'DOEB_Gasoline_Sales_Monthly_2568.xlsx' = 'https://data.doeb.go.th/dataset/a332e7c8-0bab-4639-a3e7-ad06a5cfa910/resource/0c4d7ca4-46c4-4dc1-96f8-35cb288a6d45/download/untitled.xlsx'
    'DOEB_Gasoline_Sales_Monthly_2564.csv'  = 'https://data.doeb.go.th/dataset/a332e7c8-0bab-4639-a3e7-ad06a5cfa910/resource/4d83b0ab-69e9-46f6-814b-edcc1a07f6db/download/003_diesel_ulg_2564.csv'
}

foreach ($name in $files.Keys) {
    $url = $files[$name]
    $dest = Join-Path $OutDir $name
    try {
        Write-Host "Downloading $name ..."
        Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
        Write-Host "Saved: $dest"
    } catch {
        Write-Warning "Failed $name : $($_.Exception.Message)"
    }
}

Write-Host 'Done.'
