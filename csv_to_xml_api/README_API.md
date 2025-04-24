# CSV to XML Converter API

A FastAPI application that converts CSV files to XML format based on the FORM15CB structure.

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure you have the following files in your directory:
   - api.py
   - csv_to_xml.py
   - code_mappings.py (required by csv_to_xml.py)

## Usage

1. Start the server:

```bash
python api.py
```

2. The API will be available at http://localhost:8000

3. Visit http://localhost:8000/docs for the interactive API documentation

## Endpoints

- GET `/`: API information
- POST `/convert/`: Upload a CSV file to convert it to XML
- GET `/outputs/`: List all generated XML files in the output folder

## Output Storage

All generated XML files are:
1. Returned directly as a response for download
2. Automatically saved in the `output/` folder with unique filenames
   - Format: `{original_filename}_{timestamp}_{unique_id}.xml`

## Example Usage

### Using curl

```bash
curl -X POST "http://localhost:8000/convert/" \
     -H "accept: application/xml" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@form_fields_with_data.csv"
```

### Listing all output files

```bash
curl -X GET "http://localhost:8000/outputs/"
```

### Using Python requests

```python
import requests

url = "http://localhost:8000/convert/"
files = {"file": open("form_fields_with_data.csv", "rb")}

response = requests.post(url, files=files)
with open("output.xml", "wb") as f:
    f.write(response.content)
```

## CSV Format

The API expects a CSV file with the same structure as `form_fields_with_data.csv`, containing fields like:
- Examiner Prefix
- Remitter Name
- Beneficiary Name
- etc.

## Notes

- The API will return the XML file directly for download
- All generated XML files are also saved in the `output/` directory
- For testing, you can use the Swagger UI at http://localhost:8000/docs 