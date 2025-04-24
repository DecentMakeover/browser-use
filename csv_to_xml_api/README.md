# CSV to XML Converter API

A FastAPI application that converts CSV data to XML format for FORM15CB tax documents.

## Overview

This API provides a simple interface for converting CSV data into a structured XML format that complies with the FORM15CB tax form requirements. It's built using FastAPI and leverages Python's XML processing capabilities.

## Features

- Convert CSV data to XML format via a REST API
- Store generated XML files in an output directory
- Use reference XML mode for consistent output
- List all generated XML files
- Easy to deploy and integrate with other systems

## Files in this Repository

- `api.py` - The FastAPI application
- `csv_to_xml.py` - Core conversion logic
- `code_mappings.py` - Code lookups for various fields
- `generate_dummy_data.py` - Utility to generate test data
- `requirements.txt` - Python dependencies
- `output.xml` - Reference XML file
- `form_fields.csv` - Example CSV structure
- `form_fields_with_data.csv` - Example CSV data

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Starting the Server

```bash
python api.py
```

The API will be available at http://localhost:8000

### API Endpoints

- GET `/`: API information
- POST `/convert/`: Upload a CSV file to convert it to XML
- GET `/outputs/`: List all generated XML files in the output folder

### Using the API

#### Converting CSV to XML

By default, the API uses a reference XML file (output.xml) to ensure consistent output structure:

```bash
curl -X POST "http://localhost:8000/convert/" \
     -H "accept: application/xml" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@form_fields_with_data.csv"
```

To use the full processing pipeline instead of the reference file:

```bash
curl -X POST "http://localhost:8000/convert/?use_reference=false" \
     -H "accept: application/xml" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@form_fields_with_data.csv"
```

#### Listing Generated Files

```bash
curl -X GET "http://localhost:8000/outputs/"
```

## Development

The API uses FastAPI's hot reloading feature, so changes to the code will automatically reload the server.

## Notes

- All generated XML files are saved in the `output/` directory
- The API uses a unique naming scheme: `{original_filename}_{timestamp}_{unique_id}.xml`
- For testing, you can use the Swagger UI at http://localhost:8000/docs 