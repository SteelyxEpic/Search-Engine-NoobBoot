Drop your source files in this folder. The ingestion script
(ingest.py in the project root) will scan everything here and
write one row per file into data.csv.

Supported file types:
  - PDFs ................ .pdf
  - Excel spreadsheets .. .xlsx .xlsm .xls
  - Word / minutes ...... .docx
  - Plain text / minutes  .txt .md .csv
  - Images (OCR) ........ .png .jpg .jpeg .tif .tiff .bmp .gif

You can organize files into subfolders; the script scans recursively.
