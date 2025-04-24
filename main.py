############################
# Combined single file
############################

import csv
import datetime
import importlib.util
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
# --------------------------
# All necessary imports
# --------------------------
from typing import List

# Google Generative AI
import google.generativeai as genai
from fastapi import FastAPI, Form
from fastapi import UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from google.ai.generativelanguage_v1beta.types import content
# LangChain / Browser Agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from validation import validate_csv_data

# --------------------------
# Create one FastAPI instance
# --------------------------
app = FastAPI()

# --------------------------
# CORS Middleware
# --------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://soulsearching.in"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------
# 1) Endpoint: run-test
# --------------------------

class TestStep(BaseModel):
    description: str  # Description of the test step


class TaskRequest(BaseModel):
    url: str  # The URL to open
    credentials: dict  # Credentials (optional)
    test_steps: List[TestStep]  # List of test steps


# Initialize the browser for the Agent
browser = Browser(
    config=BrowserConfig(headless=True),
)


@app.post("/run-test")
async def run_test(request: TaskRequest):
    """
    Endpoint that uses an LLM agent to run test steps on a given URL.
    """
    try:
        # Construct the task
        current_task = f"1. Open {request.url}\n"
        # if request.credentials:
        #     current_task += (
        #         f"2. Sign in using these credentials: "
        #         f"{request.credentials.get('email')} as email and "
        #         f"{request.credentials.get('password')} as password\n"
        #     )

        for idx, step in enumerate(request.test_steps, start=3):
            current_task += f"{idx}. {step.description}\n"

        print("Generated Task:\n", current_task)

        # Create and run the agent
        agent = Agent(
            task=current_task,
            llm=ChatOpenAI(model="gpt-4o"),
            browser=browser
        )
        result = await agent.run()

        return {"status": "success", "result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------
# 2) Endpoint: generate-scenarios
# --------------------------

# Define prompt and generation config
scenarios_prompt = """You are a world-class tester. 
    1. You will be provided with the text of an application user manual or application documentation. 
    2. Split that into categories/modules that you might think are useful.
    3. For every category list out all the scenarios that need to be tested.
    4. Make sure your coverage is 100%.
"""

generation_config_scenarios = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_schema": content.Schema(
        type=content.Type.OBJECT,
        required=["categories"],
        properties={
            "categories": content.Schema(
                type=content.Type.ARRAY,
                description="Categories or modules derived from the manual text.",
                items=content.Schema(
                    type=content.Type.OBJECT,
                    required=["categoryName", "scenarios"],
                    properties={
                        "categoryName": content.Schema(
                            type=content.Type.STRING,
                            description="The name of the category/module.",
                        ),
                        "scenarios": content.Schema(
                            type=content.Type.ARRAY,
                            description="List of all scenarios for the given category.",
                            items=content.Schema(
                                type=content.Type.STRING,
                                description="A test scenario description.",
                            ),
                        ),
                    },
                ),
            ),
        },
    ),
    "response_mime_type": "application/json",
}

model_scenarios = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config_scenarios,
    system_instruction=scenarios_prompt
)


@app.post("/generate-scenarios/")
async def upload_pdf(file: UploadFile, mime_type: str = Form("application/pdf")):
    """
    Endpoint to upload a PDF file and generate test scenarios.
    """
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            file_location = temp_file.name
            temp_file.write(file.file.read())

        # Upload file to Generative AI
        uploaded_file = genai.upload_file(file_location, mime_type=mime_type)

        # Generate content using the model
        response = model_scenarios.generate_content([scenarios_prompt, uploaded_file])
        print(response.text)

        # Return JSON
        return json.loads(response.text)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")


# --------------------------
# 3) Endpoint: generate-test-cases
# --------------------------

# Define another prompt for test-case generation
test_case_prompt_template = """You are a world-class tester. 
    1. You will be provided with the text of an application user manual or application documentation. 
    2. You will also be given one or more scenarios that need to be tested. 
    3. For each scenario, generate comprehensive test cases ensuring 100% coverage.
"""

# Define the generation configuration for test cases
generation_config_testcases = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_schema": content.Schema(
        type=content.Type.OBJECT,
        required=["testCases"],
        properties={
            "testCases": content.Schema(
                type=content.Type.ARRAY,
                description=(
                    "A list of key-value pairs where each key is a scenario "
                    "and the value contains test cases."
                ),
                items=content.Schema(
                    type=content.Type.OBJECT,
                    required=["key", "value"],
                    properties={
                        "key": content.Schema(
                            type=content.Type.STRING,
                            description="The scenario identifier."
                        ),
                        "value": content.Schema(
                            type=content.Type.OBJECT,
                            required=["testCases"],
                            properties={
                                "testCases": content.Schema(
                                    type=content.Type.ARRAY,
                                    description=(
                                        "A list of objects, each containing a scenario "
                                        "and its corresponding test cases."
                                    ),
                                    items=content.Schema(
                                        type=content.Type.OBJECT,
                                        required=["scenario", "testCases"],
                                        properties={
                                            "scenario": content.Schema(
                                                type=content.Type.STRING,
                                                description=(
                                                    "The specific scenario for which "
                                                    "test cases are generated."
                                                ),
                                            ),
                                            "testCases": content.Schema(
                                                type=content.Type.ARRAY,
                                                description=(
                                                    "A list of detailed test cases "
                                                    "for the given scenario."
                                                ),
                                                items=content.Schema(
                                                    type=content.Type.STRING,
                                                    description=(
                                                        "A detailed description of a single test case."
                                                    ),
                                                ),
                                            ),
                                        },
                                    ),
                                ),
                            },
                        ),
                    },
                ),
            ),
        },
    ),
    "response_mime_type": "application/json",
}

model_testcases = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config_testcases,
    system_instruction=test_case_prompt_template,
)


@app.post("/generate-test-cases/")
async def upload_pdf_with_scenarios(
        file: UploadFile,
        scenarios: List[str] = Form(...),
        mime_type: str = Form("application/pdf")
):
    """
    Endpoint to upload a PDF file and generate test cases for given scenarios.
    """
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            file_location = temp_file.name
            temp_file.write(file.file.read())

        # Upload file to Generative AI
        uploaded_file = genai.upload_file(file_location, mime_type=mime_type)

        # Create a prompt for each scenario
        test_case_prompts = [
            f"Generate test cases for the scenario: '{scenario}' using the content of the uploaded document."
            for scenario in scenarios
        ]

        # Generate test cases using the model
        responses = [
            model_testcases.generate_content(
                [test_case_prompt_template, uploaded_file, scenario_prompt]
            )
            for scenario_prompt in test_case_prompts
        ]

        # Parse the JSON responses (one per scenario)
        test_cases = {
            scenario: json.loads(response.text)
            for scenario, response in zip(scenarios, responses)
        }

        return {"testCases": test_cases}

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file or generating test cases: {e}")


# --------------------------
# That's it! Now you have:
# - A single FastAPI app
# - All three endpoints /run-test, /generate-scenarios, /generate-test-cases
# --------------------------


# Create output directory if it doesn't exist
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Check if conversion module exists
if os.path.exists("csv_to_xml.py"):
    # Import the csv_to_xml module dynamically
    spec = importlib.util.spec_from_file_location("csv_to_xml", "csv_to_xml.py")
    csv_to_xml = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(csv_to_xml)
else:
    raise ImportError("csv_to_xml.py module not found in the current directory")


@app.post("/convert/", response_class=FileResponse)
async def convert_csv_to_xml(
        file: UploadFile = File(...),
        background_tasks: BackgroundTasks = None,
        skip_validation: bool = Query(False, description="Skip data validation")
):
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        # Generate a unique filename based on timestamp and UUID
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        original_filename = file.filename.replace('.csv', '')
        output_filename = f"{original_filename}_{timestamp}_{unique_id}.xml"

        # Create a temporary directory to work with
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            # Save the uploaded file to temp directory
            temp_csv_path = temp_dir_path / "original_upload.csv"
            with open(temp_csv_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Process the uploaded file which should be in form_fields_with_data.csv format
            # into the data_for_conversion.csv format (single row with headers as column names)
            try:
                with open(temp_csv_path, 'r', newline='') as input_file:
                    csv_reader = csv.reader(input_file)
                    # First row should contain "Field Name,Value"
                    header = next(csv_reader)

                    if len(header) != 2 or header[0].strip() != "Field Name" or header[1].strip() != "Value":
                        raise ValueError("CSV file must have 'Field Name' and 'Value' columns")

                    # Read the field names and values
                    field_names = []
                    values = []
                    for row in csv_reader:
                        if len(row) >= 2:
                            field_names.append(row[0])
                            values.append(row[1])

                # Create the data_for_conversion.csv file with the correct format
                data_file_path = temp_dir_path / "data_for_conversion.csv"
                with open(data_file_path, 'w', newline='') as output_file:
                    # Write field names as the header row
                    csv_writer = csv.writer(output_file)
                    csv_writer.writerow(field_names)
                    # Write values as the data row
                    csv_writer.writerow(values)

                print("✓ CSV reformatted successfully")
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to process CSV file: {str(e)}"
                )

            # Copy all necessary files to temp directory
            original_dir = os.getcwd()

            # Copy only local files from the current directory
            files_to_copy = ["csv_to_xml.py", "code_mappings.py", "validation.py"]
            for file_name in files_to_copy:
                if os.path.exists(file_name):
                    shutil.copy(file_name, temp_dir_path / file_name)
                else:
                    # Skip validation.py if not found and validation is skipped
                    if file_name == "validation.py" and skip_validation:
                        continue
                    else:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Required file {file_name} not found in the current directory"
                        )

            # Change to temp directory for processing
            os.chdir(temp_dir)

            # Validate data if required
            if not skip_validation:
                try:
                    print("Validating CSV data before conversion...")

                    # Import validation here to ensure we get the local version
                    from validation import validate_csv_data

                    validation_errors = validate_csv_data(str(data_file_path))

                    if validation_errors:
                        error_message = "Validation failed with the following errors:\n"
                        for row, errors in validation_errors.items():
                            error_message += f"\n{row}:\n"
                            for field, error in errors.items():
                                error_message += f"  - {field}: {error}\n"

                        print(f"Validation failed: {error_message}")
                        os.chdir(original_dir)
                        return JSONResponse(
                            status_code=400,
                            content={
                                "status": "error",
                                "detail": "Validation failed",
                                "validation_errors": validation_errors
                            }
                        )

                    print("✓ Validation passed successfully")
                except Exception as e:
                    os.chdir(original_dir)
                    raise HTTPException(status_code=500, detail=f"Error during validation: {str(e)}")

            # Now execute the XML creation function
            try:
                # This will use data_for_conversion.csv to create output.xml
                csv_to_xml.create_xml()
                print("✓ XML file created successfully")
            except Exception as e:
                os.chdir(original_dir)
                raise HTTPException(status_code=500, detail=f"Error in XML generation: {str(e)}")

            # Check if XML was created
            xml_path = Path(temp_dir) / "output.xml"
            if not xml_path.exists():
                os.chdir(original_dir)
                raise HTTPException(status_code=500, detail="Failed to generate XML file")

            # Go back to original directory
            os.chdir(original_dir)

            # Save a copy to the output directory
            output_file_path = OUTPUT_DIR / output_filename
            shutil.copy(xml_path, output_file_path)
            print(f"XML saved to {output_file_path}")

            # Copy the output file to a location where it's accessible for the response
            temp_output_path = Path("temp_output.xml")
            shutil.copy(xml_path, temp_output_path)

            # Return the XML file
            temp_output_path_str = str(temp_output_path)
            if background_tasks:
                background_tasks.add_task(os.remove, temp_output_path_str)

            return FileResponse(
                path=temp_output_path,
                filename=output_filename,
                media_type="application/xml"
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during conversion: {str(e)}")


@app.get("/outputs/")
async def list_outputs():
    """List all XML files in the output directory"""
    files = [f.name for f in OUTPUT_DIR.glob("*.xml")]
    return {"files": files, "count": len(files)}


@app.post("/validate-csv/")
async def validate_csv(file: UploadFile = File(...)):
    """
    Validate CSV data without conversion to XML.
    This endpoint can be used to check if the data is valid before proceeding with conversion.
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            # Save the uploaded file
            csv_path = temp_dir_path / "data_to_validate.csv"
            with open(csv_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Run validation
            validation_errors = validate_csv_data(str(csv_path))

            if validation_errors:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "detail": "Validation failed",
                        "validation_errors": validation_errors
                    }
                )
            else:
                return {"status": "success", "message": "CSV data is valid"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during validation: {str(e)}")
