# Deploying MedExtract API to AWS: A Serverless Guide

This guide details the process of deploying the **MedExtract** application using a modern, serverless architecture on AWS. The key components of this deployment are:

  * **AWS S3**: Used for both storing conversation memory and hosting the static frontend application.
  * **AWS Lambda**: Executes the Python backend code in a serverless environment.
  * **API Gateway (HTTP API)**: Acts as the public-facing RESTful API endpoint, routing requests to Lambda and managing CORS.
  * **CloudFront**: Provides a global Content Delivery Network (CDN) for fast, secure content delivery and enforces HTTPS for the frontend.

-----

## Part 1: Preparing the Server Code for AWS

The initial steps involve modifying the application's dependencies and structure to be compatible with the AWS Lambda environment, leveraging `boto3` for S3 access and `mangum` for wrapping FastAPI.

### Step 1: Update Requirements

Ensure all necessary dependencies, including AWS SDK (`boto3`) and the serverless adapter (`mangum`), are listed in `backend/requirements.txt`.

```
fastapi
uvicorn
python-dotenv
python-multipart
boto3
pypdf
mangum
```

### Step 2: Update Server for AWS

The `backend/server.py` file is updated to integrate S3 memory management and a conditional startup for local vs. serverless execution.

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict
import json
import uuid
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from context import prompt
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

app = FastAPI()

# Configure CORS
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set. Please add it to your .env file.")

## Initialize the ChatGroq model
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
)

## Memory storage configuration
USE_S3 = os.getenv("USE_S3", "false").lower() == "true"
S3_BUCKET = os.getenv("S3_BUCKET", "")
MEMORY_DIR = os.getenv("MEMORY_DIR", "memory") # Using a relative path for local storage

## Initialize S3 client if needed
if USE_S3:
    s3_client = boto3.client("s3")


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


class Message(BaseModel):
    role: str
    content: str
    timestamp: str

## Memory Management Functions

def get_memory_path(session_id: str) -> str:
    """Generates the file/key path for a given session ID."""
    return f"{session_id}.json"


def load_conversation(session_id: str) -> List[Dict]:
    """Load conversation history from storage (S3 or local file)."""
    if USE_S3:
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=get_memory_path(session_id))
            return json.loads(response["Body"].read().decode("utf-8"))
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return []
            raise
    else:
        ## Local file storage
        file_path = os.path.join(MEMORY_DIR, get_memory_path(session_id))
        print(f"Loading conversation from local file: {file_path}")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        return []


def save_conversation(session_id: str, messages: List[Dict]):
    """Save conversation history to storage (S3 or local file)."""
    if USE_S3:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=get_memory_path(session_id),
            Body=json.dumps(messages, indent=2),
            ContentType="application/json",
        )
    else:
        ## Local file storage
        os.makedirs(MEMORY_DIR, exist_ok=True)
        file_path = os.path.join(MEMORY_DIR, get_memory_path(session_id))
        with open(file_path, "w") as f:
            json.dump(messages, f, indent=2) 


def to_lc_messages(conversation: List[Dict], system_prompt: str) -> List:
    """Converts a list of dict messages into a list of LangChain message objects."""
    lc_messages = [SystemMessage(content=system_prompt)]
    for msg in conversation:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
    return lc_messages

## API Endpoints
@app.get("/")
async def root():
    return {
        "message": "MedExtract is running.",
        "memory_enabled": True,
        "storage": "S3" if USE_S3 else "local",
        "model_backend": "Groq/LangChain",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "use_s3": USE_S3}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        ## Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        ## Load conversation history
        conversation = load_conversation(session_id)

        lc_messages = to_lc_messages(conversation[-10:], prompt())

        ## Add current user message
        lc_messages.append(HumanMessage(content=request.message))

        ## Call Groq API via ChatGroq (LangChain's invoke method)
        response = llm.invoke(lc_messages)
        assistant_response = response.content

        ## Update conversation history (using the stored dictionary format)
        current_time = datetime.now().isoformat()
        conversation.append(
            {"role": "user", "content": request.message, "timestamp": current_time}
        )
        conversation.append(
            {
                "role": "assistant",
                "content": assistant_response,
                "timestamp": current_time,
            }
        )

        ## Save conversation
        save_conversation(session_id, conversation)

        return ChatResponse(response=assistant_response, session_id=session_id)

    except Exception as e:
        import traceback
        print(f"Error in chat endpoint: {str(e)}")
        print(traceback.format_exc()) 
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.get("/conversation/{session_id}")
async def get_conversation(session_id: str):
    """Retrieve conversation history"""
    try:
        conversation = load_conversation(session_id)
        return {"session_id": session_id, "messages": conversation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    if not USE_S3:
        os.makedirs(MEMORY_DIR, exist_ok=True)
        print(f"Local memory directory created/checked: {os.path.abspath(MEMORY_DIR)}")
        
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Step 3: Create Lambda Handler

The `lambda_handler.py` file uses **Mangum** to bridge the ASGI framework (FastAPI) to the AWS Lambda environment.

```python
from mangum import Mangum
from server import app

handler = Mangum(app)
```

### Step 4: Update Dependencies and Test Locally

Before deployment, install dependencies and verify local operation:

```bash
cd backend
uv add -r requirements.txt
uv run uvicorn server:app --reload
```

In a new terminal, run the frontend:

```bash
cd frontend
npm run dev
```

-----

## Part 2: AWS Environment Setup

This section outlines the initial AWS setup, including creating an IAM user with necessary permissions.

### Step 1: Create Environment Configuration

Create a root-level `.env` file to hold project and AWS configuration variables.

```bash
AWS_ACCOUNT_ID=aws_account_id    # actual AWS account ID with 12 digits
DEFAULT_AWS_REGION=region
PROJECT_NAME=MedExtract
GROQ_API_KEY=your_groq_key
MEMORY_DIR=../memory
```

### Step 2: Sign In to AWS Console

Access the AWS console via [aws.amazon.com](https://aws.amazon.com) and sign in as the **root user**.

### Step 3: Create an IAM User

Create a dedicated IAM user for deployment tasks to follow the principle of least privilege.

1.  Search for **IAM** in the AWS Console.
2.  Navigate to **Users** → **Create user**.
3.  Set the Username: `thilina-aiengineer`.
4.  Check ✅ **Provide user access to the AWS Management Console**.
5.  Select **I want to create an IAM user**.
6.  Set a **Custom password** and uncheck the forced password change option.
7.  Click **Next**.

### Step 4: Create IAM User Group with Permissions

Create a group with the necessary full-access policies required for a streamlined deployment process.

1.  In **IAM**, go to **User groups** → **Create group**.
2.  Group name: `MedExtract`.
3.  Attach the following policies:
      * `AWSLambda_FullAccess`
      * `AmazonS3FullAccess`
      * `AmazonAPIGatewayAdministrator`
      * `CloudFrontFullAccess`
      * `IAMReadOnlyAccess`
      * `AmazonDynamoDBFullAccess_v2`
4.  Click **Create group**.

### Step 5: Add User to Group

Associate the created user with the new permissions group.

1.  In IAM, go to **Users** → Select `thilina-aiengineer`.
2.  Click **Add to groups** and select the `MedExtract` group.
3.  Click **Add to groups**.

### Step 6: Sign In as IAM User

For security, sign out of the root account and sign in using the new `thilina-aiengineer` IAM credentials.

-----

## Part 3: Packaging the Lambda Function

This stage involves bundling the application code, dependencies, and resources into a `zip` file suitable for the Lambda runtime environment.

### Step 1: Create Deployment Script

The `backend/deploy.py` script utilizes **Docker** to ensure dependencies are compiled for the **Lambda runtime environment (Linux/x86\_64)**, avoiding common dependency issues.

```python
import os
import shutil
import zipfile
import subprocess


def main():
    print("Creating Lambda deployment package...")

    if os.path.exists("lambda-package"):
        shutil.rmtree("lambda-package")
    if os.path.exists("lambda-deployment.zip"):
        os.remove("lambda-deployment.zip")

    ## Create package directory
    os.makedirs("lambda-package")

    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{os.getcwd()}:/var/task",
            "--platform",
            "linux/amd64",  # Force x86_64 architecture
            "--entrypoint",
            "",  # Override the default entrypoint
            "public.ecr.aws/lambda/python:3.12",
            "/bin/sh",
            "-c",
            "pip install --target /var/task/lambda-package -r /var/task/requirements.txt --platform manylinux2014_x86_64 --only-binary=:all: --upgrade",
        ],
        check=True,
    )

    ## Copy application files
    print("Copying application files...")
    for file in ["server.py", "lambda_handler.py", "context.py", "resources.py"]:
        if os.path.exists(file):
            shutil.copy2(file, "lambda-package/")
    
    ## Copy data directory
    if os.path.exists("data"):
        shutil.copytree("data", "lambda-package/data")

    ## Create zip
    print("Creating zip file...")
    with zipfile.ZipFile("lambda-deployment.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk("lambda-package"):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, "lambda-package")
                zipf.write(file_path, arcname)

    size_mb = os.path.getsize("lambda-deployment.zip") / (1024 * 1024)
    print(f"✓ Created lambda-deployment.zip ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
```

### Step 2: Update .gitignore

Prevent deployment files from being tracked in version control.

```
lambda-deployment.zip
lambda-package/
```

### Step 3: Build the Lambda Package

Ensure Docker is running and execute the deployment script:

```bash
cd backend
uv run deploy.py
```

This generates the **`lambda-deployment.zip`** file.

-----

## Part 4: Deploying the Lambda Function

The zipped package is now uploaded to AWS Lambda, and the function is configured.

### Step 1: Create Lambda Function in AWS console

1.  In the AWS Console, search for **Lambda**.
2.  Click **Create function**.
3.  Choose **Author from scratch**.
4.  Configuration:
      * Function name: `MedExtract-api`
      * Runtime: **Python 3.12**
      * Architecture: **x86\_64**
5.  Click **Create function**.

### Step 2: Upload the Code

1.  On the Lambda function page, in the **Code source** section, click **Upload from** → **.zip file**.
2.  Upload the `backend/lambda-deployment.zip` file.
3.  Click **Save**.

### Step 3: Configure Handler

The handler must be set to point to the `handler` object in `lambda_handler.py`.

1.  In **Runtime settings**, click **Edit**.
2.  Change Handler to: `lambda_handler.handler`.
3.  Click **Save**.

### Step 4: Configure Environment Variables

Set up essential environment variables, including the Groq API key and S3 configuration.

1.  Go to **Configuration** → **Environment variables** → **Edit**.
2.  Add variables (Note: `S3_BUCKET` is a placeholder for now):
      * `GROQ_API_KEY` = *your\_groq\_api\_key* (Note: The provided code uses Groq, so this may be a typo and should be **GROQ\_API\_KEY** - ensure you use your Groq key)
      * `CORS_ORIGINS` = `*`
      * `USE_S3` = `true`
      * `S3_BUCKET` = `MedExtract-memory`
3.  Click **Save**.

### Step 5: Increase Timeout

Extend the default execution time to prevent timeouts during complex LLM interactions.

1.  In **Configuration** → **General configuration** → **Edit**.
2.  Set Timeout to **30 seconds**.
3.  Click **Save**.

### Step 6: Test the Lambda Function

Test the function's health endpoint using a mock API Gateway event.

1.  Click **Test** tab.

2.  Create a new test event named `HealthCheck`.

3.  Select **API Gateway AWS Proxy** template.

4.  Modify the Event JSON:

    ```json
    {
      "version": "2.0",
      "routeKey": "GET /health",
      "rawPath": "/health",
      "headers": {
        "accept": "application/json",
        "content-type": "application/json",
        "user-agent": "test-invoke"
      },
      "requestContext": {
        "http": {
          "method": "GET",
          "path": "/health",
          "protocol": "HTTP/1.1",
          "sourceIp": "127.0.0.1",
          "userAgent": "test-invoke"
        },
        "routeKey": "GET /health",
        "stage": "$default"
      },
      "isBase64Encoded": false
    }
    ```

5.  Click **Save** → **Test**. The expected result is `{"status": "healthy", "use_s3": true}`.

-----

## Part 5: Creating S3 Buckets

Two S3 buckets are required: one for persistent conversation memory and one for static frontend hosting.

### Step 1: Create Memory Bucket

This bucket stores conversation history JSON files.

1.  In AWS Console, search for **S3** → **Create bucket**.
2.  Configuration:
      * Bucket name: `MedExtract-memory` (Must be globally unique).
      * Region: Match the Lambda function region.
3.  Leave default settings and **Create bucket**.

### Step 2: Update Lambda Environment

Update the Lambda environment variable with the confirmed bucket name.

1.  Lambda → **Configuration** → **Environment variables**.
2.  Update `S3_BUCKET` to the exact name: `MedExtract-memory`.
3.  Click **Save**.

### Step 3: Add S3 Permissions to Lambda

The Lambda execution role needs permission to read and write to the S3 memory bucket.

1.  Lambda → **Configuration** → **Permissions**.
2.  Click the execution role name to open IAM.
3.  Click **Add permissions** → **Attach policies**.
4.  Attach: `AmazonS3FullAccess`.

### Step 4: Create Frontend Bucket

This bucket will host the static web application.

1.  In S3, click **Create bucket**.
2.  Configuration:
      * Bucket name: `MedExtract-frontend`.
      * Region: Match the Lambda region.
      * **Uncheck** "**Block all public access**" and acknowledge.
3.  Click **Create bucket**.

### Step 5: Enable Static Website Hosting

1.  Go to the frontend bucket → **Properties** tab.
2.  Scroll to **Static website hosting** → **Edit**.
3.  Enable: **Host a static website**.
      * Index document: `index.html`
      * Error document: `404.html`
4.  **Save changes**. Note the **Bucket website endpoint** URL.

### Step 6: Configure Bucket Policy

Set a policy to allow public read access to the bucket's objects.

1.  Go to **Permissions** tab → **Bucket policy** → **Edit**.
2.  Add the policy, replacing `YOUR-BUCKET-NAME`:

<!-- end list -->

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadGetObject",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::YOUR-BUCKET-NAME/*"
        }
    ]
}
```

3.  **Save changes**.

-----

## Part 6: Setting Up API Gateway

The API Gateway is configured to expose the Lambda function as a public REST API.

### Step 1: Create HTTP API with Integration

1.  In AWS Console, search for **API Gateway** → **Create API**.
2.  Choose **HTTP API** → **Build**.
3.  **Step 1 - Create and configure integrations:**
      * Click **Add integration** → Integration type: **Lambda**.
      * Select `MedExtract-api` function.
      * API name: `MedExtract-api-gateway`.
      * Click **Next**.

### Step 2: Configure Routes

Define the necessary routes and map them to the Lambda integration.

1.  **Step 2 - Configure routes:**
2.  **Update the existing default route:**
      * Method: `ANY`
      * Resource path: `/{proxy+}`
      * Integration target: `MedExtract-api`
3.  **Add additional explicit routes (important for static API definitions):**
      * Route 1: `GET /`
      * Route 2: `GET /health`
      * Route 3: `POST /chat`
      * Route 4 (for CORS pre-flight): `OPTIONS /{proxy+}`
4.  Click **Next**.

### Step 3: Configure Stages

1.  **Step 3 - Configure stages:** Leave Stage name as `$default` and auto-deploy enabled.
2.  Click **Next**.

### Step 4: Review and Create

1.  Review and click **Create**.

### Step 5: Configure CORS

Enable Cross-Origin Resource Sharing (CORS) to allow the frontend to communicate with the API.

1.  In the new API, go to **CORS** in the left menu.
2.  Click **Configure**.
3.  Set the following (click **Add** after typing each value):
      * Access-Control-Allow-Origin: `*`
      * Access-Control-Allow-Headers: `*`
      * Access-Control-Allow-Methods: `*` (or `GET, POST, OPTIONS`)
      * Access-Control-Max-Age: `300`
4.  Click **Save**.

### Step 6: Test Your API

Copy the **Invoke URL** from the API details and test the health endpoint using `curl`.

```bash
curl https://YOUR-API-ID.execute-api.us-east-1.amazonaws.com/health
```

Expected output: `{"status": "healthy", "use_s3": true}`.

-----

## Part 7: Building and Deploying Frontend 

The Next.js frontend is built as static assets and uploaded to the S3 hosting bucket.

### Step 1: Update Frontend API URL

Modify the frontend to point to the new API Gateway URL.

In `frontend/components/bot.tsx`, update the fetch call:

```typescript
// With your API Gateway URL:
const response = await fetch('https://YOUR-API-ID.execute-api.us-east-1.amazonaws.com/chat', {
```

### Step 2: Configure for Static Export

Enable Next.js static export functionality.

Update `frontend/next.config.ts`:

```typescript
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export',
  images: {
    unoptimized: true
  }
};

export default nextConfig;
```

### Step 3: Build Static Export

Generate the static HTML/CSS/JS files in the `out` directory.

```bash
cd frontend
npm run build
```

### Step 4: Upload to S3

Use the AWS CLI to synchronize the local `out` directory with the S3 frontend bucket.

```bash
cd frontend
aws s3 sync out/ s3://YOUR-FRONTEND-BUCKET-NAME/ --delete
```

### Step 5: Test the Static Site

Verify the basic loading of the application using the **Bucket website endpoint** URL from S3 properties.

-----

## Part 8: Setting Up CloudFront

CloudFront is deployed as a CDN to provide global speed, caching, and a crucial **HTTPS** layer for the static frontend.

### Step 1: Get the S3 Website Endpoint

Retrieve the S3 Static Website Hosting endpoint URL (e.g., `http://MedExtract-frontend-xxx.s3-website-us-east-1.amazonaws.com`).

### Step 2: Create CloudFront Distribution

1.  In AWS Console, search for **CloudFront** → **Create distribution**.
2.  **Origin:**
      * Origin domain name: **Paste the S3 website endpoint (without `http://`)**.
      * **Origin protocol policy**: Select **HTTP only** (S3 static websites do not support HTTPS).
      * Click **Add origin**.
3.  **Default cache behavior:**
      * Viewer protocol policy: **Redirect HTTP to HTTPS** (This is how we enforce HTTPS).
4.  **Settings:**
      * Default root object: `index.html`.
5.  **Create distribution**.

### Step 3: Wait for Deployment

The CloudFront distribution will take several minutes to deploy globally.

### Step 4: Update CORS Settings

Restrict the Lambda CORS to only accept requests originating from the new CloudFront domain.

1.  Find the **Distribution domain name** (`https://YOUR-CLOUDFRONT-DOMAIN.cloudfront.net`).
2.  Go to Lambda → **Configuration** → **Environment variables**.
3.  Update `CORS_ORIGINS`:
      * New value: `https://YOUR-CLOUDFRONT-DOMAIN.cloudfront.net`
4.  Click **Save**.

### Step 5: Invalidate CloudFront Cache

Clear the cached content to ensure the latest frontend changes are served.

1.  In CloudFront, select the distribution → **Invalidations** tab.
2.  Click **Create invalidation**.
3.  Add path: `/*`.
4.  **Create invalidation**.

-----

## Part 9: Final Testing and Verification

### Step 1: Access the MedExtract

Use the final **CloudFront URL** (`https://YOUR-DISTRIBUTION.cloudfront.net`). The application should load via HTTPS and interact with the API.

### Step 2: Verify Memory in S3

Check the S3 memory bucket to confirm that conversation history is being saved as JSON files for each session.

### Step 3: Monitor CloudWatch Logs

Use CloudWatch to debug any runtime issues with the Lambda function.

1.  Go to CloudWatch → **Log groups**.
2.  Find `/aws/lambda/MedExtract-api` and check recent logs.

-----

### Key Serverless Components Summary

| Component | Role in MedExtract Deployment |
| :--- | :--- |
| **CloudFront** | Global CDN, provides **HTTPS**, caching, and high-speed delivery of static assets. |
| **S3 Frontend Bucket** | Hosts the static Next.js/React files (`index.html`, etc.). |
| **API Gateway** | Manages API routes, handles pre-flight **CORS**, and routes requests to the Lambda function. |
| **AWS Lambda** | Runs the Python **FastAPI backend serverlessly** (via Mangum). |
| **S3 Memory Bucket** | Provides **persistent storage** for conversation history (JSON files). |

-----