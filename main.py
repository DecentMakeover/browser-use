############################
# Combined single file
############################

import json
# --------------------------
# All necessary imports
# --------------------------
import os
import tempfile
from typing import List

# Google Generative AI
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from google.ai.generativelanguage_v1beta.types import content
# LangChain / Browser Agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig

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
