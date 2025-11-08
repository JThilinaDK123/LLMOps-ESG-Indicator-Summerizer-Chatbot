# Deploy MedExtract API to AWS

- **AWS S3 Buckets** for store memory and the frontend application
- **AWS Lambda** for serverless backend deployment
- **API Gateway** for RESTful API management
- **S3 buckets** for memory storage and static hosting
- **CloudFront** for global content delivery

## Part 1: Update the Server code for AWS

### Step 1: Update Requirements

Update `backend/requirements.txt`:

```
fastapi
uvicorn
openai
python-dotenv
python-multipart
boto3
pypdf
mangum
```

### Step 2: Update Server for AWS

Replace `backend/server.py` with this AWS version:

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

Create `backend/lambda_handler.py`:

```python
from mangum import Mangum
from server import app

# Create the Lambda handler
handler = Mangum(app)
```

### Step 4: Update Dependencies and Test Locally

```bash
cd backend
uv add -r requirements.txt
uv run uvicorn server:app --reload
```

1. Open a new terminal
2. `cd frontend`
3. `npm run dev`


## Part 2: Set Up AWS Environment

### Step 1: Create Environment Configuration

Create a `.env` file in the root directory:

```bash
AWS_ACCOUNT_ID=aws_account_id    # actual AWS account ID with 12 digits
DEFAULT_AWS_REGION=region
PROJECT_NAME=MedExtract
GROQ_API_KEY=your_groq_key
MEMORY_DIR=../memory
```

### Step 2: Sign In to AWS Console

1. Go to [aws.amazon.com](https://aws.amazon.com)
2. Sign in as **root user** 

### Step 3: Create an IAM User 

1. Search for IAM in the AWS Console
2. Click **Users**, **Create user**
3. Username: `thilina-aiengineer`
4. Check ✅ Provide user access to the AWS Management Console
5. Select **I want to create an IAM user**
6. Choose Custom password and set a strong password
7. Uncheck **Users must create a new password** at next sign-in
8. Click Next


### Step 4: Create IAM User Group with Permissions

1. In AWS Console, search for **IAM**
2. Click **User groups** → **Create group**
3. Group name: `MedExtract`
4. Attach the following policies
   - `AWSLambda_FullAccess` - For Lambda operations
   - `AmazonS3FullAccess` - For S3 bucket operations
   - `AmazonAPIGatewayAdministrator` - For API Gateway
   - `CloudFrontFullAccess` - For CloudFront distribution
   - `IAMReadOnlyAccess` - To view roles
   - `AmazonDynamoDBFullAccess_v2`
5. Click **Create group**

### Step 5: Add User to Group

1. In IAM, click **Users** → Select `thilina-aiengineer`
2. Click **Add to groups**
3. Select `MedExtract`
4. Click **Add to groups**

### Step 6: Sign In as IAM User

1. Sign out from root account
2. Sign in as `thilina-aiengineer` with the IAM credentials

## Part 3: Package Lambda Function

This stage will help to package all the codes and data sources into a zip file that can add as a lambda function

### Step 1: Create Deployment Script

Create `backend/deploy.py`:

```python
import os
import shutil
import zipfile
import subprocess


def main():
    print("Creating Lambda deployment package...")

    # Clean up
    if os.path.exists("lambda-package"):
        shutil.rmtree("lambda-package")
    if os.path.exists("lambda-deployment.zip"):
        os.remove("lambda-deployment.zip")

    # Create package directory
    os.makedirs("lambda-package")

    # Install dependencies using Docker with Lambda runtime image
    print("Installing dependencies for Lambda runtime...")

    # Use the official AWS Lambda Python 3.12 image
    # This ensures compatibility with Lambda's runtime environment
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

    # Copy application files
    print("Copying application files...")
    for file in ["server.py", "lambda_handler.py", "context.py", "resources.py"]:
        if os.path.exists(file):
            shutil.copy2(file, "lambda-package/")
    
    # Copy data directory
    if os.path.exists("data"):
        shutil.copytree("data", "lambda-package/data")

    # Create zip
    print("Creating zip file...")
    with zipfile.ZipFile("lambda-deployment.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk("lambda-package"):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, "lambda-package")
                zipf.write(file_path, arcname)

    # Show package size
    size_mb = os.path.getsize("lambda-deployment.zip") / (1024 * 1024)
    print(f"✓ Created lambda-deployment.zip ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
```

### Step 2: Update .gitignore

```
lambda-deployment.zip
lambda-package/
```

### Step 3: Build the Lambda Package

This creates `lambda-deployment.zip` containing the Lambda function and all dependencies.
Make sure Docker Desktop is running, then:

```bash
cd backend
uv run deploy.py
```

## Part 4: Deploy Lambda Function

### Step 1: Create Lambda Function in AWS console

1. In AWS Console, search for **Lambda**
2. Click **Create function**
3. Choose **Author from scratch**
4. Configuration:
   - Function name: `MedExtract-api`
   - Runtime: **Python 3.12**
   - Architecture: **x86_64**
5. Click **Create function**

### Step 2: Upload the Code

**Direct Upload**

1. In the Lambda function page, under **Code source**
2. Click **Upload from** → **.zip file**
3. Click **Upload** and select your `backend/lambda-deployment.zip`
4. Click **Save**

This will show some error at the first stage. Below step will sort out that issue

### Step 3: Configure Handler

1. In **Runtime settings**, click **Edit**
2. Change Handler to: `lambda_handler.handler`
3. Click **Save**

### Step 4: Configure Environment Variables

1. Click **Configuration** tab → **Environment variables**
2. Click **Edit** → **Add environment variable**
3. Add these variables:
   - `OPENAI_API_KEY` = your_openai_api_key
   - `CORS_ORIGINS` = `*` (this will updated in next steps)
   - `USE_S3` = `true`
   - `S3_BUCKET` = `MedExtract-memory` (S3 bucket will create in next step. update this with that bucket name)
4. Click **Save**

### Step 5: Increase Timeout

1. In **Configuration** → **General configuration**
2. Click **Edit**
3. Set Timeout to **30 seconds**
4. Click **Save**

### Step 6: Test the Lambda Function

1. Click **Test** tab
2. Create new test event:
   - Event name: `HealthCheck`
   - Event template: **API Gateway AWS Proxy** (scroll down to find it)
   - Modify the Event JSON to:
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
3. Click **Save** → **Test**
4. It should see a successful response with a body containing `{"status": "healthy", "use_s3": true}`


## Part 5: Create S3 Buckets

### Step 1: Create Memory Bucket

1. In AWS Console, search for **S3**
2. Click **Create bucket**
3. Configuration:
   - Bucket name: `MedExtract-memory` (must be a unique namespace)
   - Region: Same as the Lambda function stays(e.g., us-east-1)
   - Leave all other settings as default
4. Click **Create bucket**
5. Copy the exact bucket name

### Step 2: Update Lambda Environment

1. Go back to Lambda → **Configuration** → **Environment variables**
2. Update `S3_BUCKET` with the actual bucket name (`MedExtract-memory`)
3. Click **Save**

### Step 3: Add S3 Permissions to Lambda

1. In Lambda → **Configuration** → **Permissions**
2. Click on the execution role name (opens IAM)
3. Click **Add permissions** → **Attach policies**
4. Search and select: `AmazonS3FullAccess`
5. Click **Attach policies**

### Step 4: Create Frontend Bucket

1. Back in S3, click **Create bucket**
2. Configuration:
   - Bucket name: `MedExtract-frontend`
   - Region: Same as Lambda
   - **Uncheck** "Block all public access"
   - Check the acknowledgment box
3. Click **Create bucket**

### Step 5: Enable Static Website Hosting

1. Click on your frontend bucket
2. Go to **Properties** tab
3. Scroll to **Static website hosting** → **Edit**
4. Enable static website hosting:
   - Hosting type: **Host a static website**
   - Index document: `index.html`
   - Error document: `404.html`
5. Click **Save changes**
6. Note the **Bucket website endpoint** URL

### Step 6: Configure Bucket Policy

1. Go to **Permissions** tab
2. Under **Bucket policy**, click **Edit**
3. Add this policy (replace `YOUR-BUCKET-NAME`):

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

4. Click **Save changes**

## Part 6: Set Up API Gateway

### Step 1: Create HTTP API with Integration

1. In AWS Console, search for **API Gateway**
2. Click **Create API**
3. Choose **HTTP API** → **Build**
4. **Step 1 - Create and configure integrations:**
   - Click **Add integration**
   - Integration type: **Lambda**
   - Lambda function: Select `MedExtract-api` from the dropdown
   - API name: `MedExtract-api-gateway`
   - Click **Next**

### Step 2: Configure Routes

1. **Step 2 - Configure routes:**
2. The default route already created. Click **Add route** to add more:

**Existing route (update it):**
- Method: `ANY`
- Resource path: `/{proxy+}`
- Integration target: `MedExtract-api`

**Add these additional routes (click Add route for each):**

Route 1:
- Method: `GET`
- Resource path: `/`
- Integration target: `MedExtract-api`

Route 2:
- Method: `GET`
- Resource path: `/health`
- Integration target: `MedExtract-api`

Route 3:
- Method: `POST`
- Resource path: `/chat`
- Integration target: `MedExtract-api`

Route 4 (for CORS):
- Method: `OPTIONS`
- Resource path: `/{proxy+}`
- Integration target: `MedExtract-api`

3. Click **Next**

### Step 3: Configure Stages

1. **Step 3 - Configure stages:**
   - Stage name: `$default` (leave as is)
   - Auto-deploy: Leave enabled
2. Click **Next**

### Step 4: Review and Create

1. **Step 4 - Review and create:**
   - Review the configuration
2. Click **Create**

### Step 5: Configure CORS

After creation, configure CORS:

1. In the newly created API, go to **CORS** in the left menu
2. Click **Configure**
3. Settings:
   - Access-Control-Allow-Origin: Type `*` and **click Add** 
   - Access-Control-Allow-Headers: Type `*` and **click Add**
   - Access-Control-Allow-Methods: Type `*` and **click Add** (or add `GET, POST, OPTIONS` individually)
   - Access-Control-Max-Age: `300`
4. Click **Save**


### Step 6: Test Your API

1. Go to **API details** (or **Stages** → **$default**)
2. Copy your **Invoke URL** (looks like: `https://abc123xyz.execute-api.us-east-1.amazonaws.com`)
3. Test with curl or browser:

```bash
curl https://YOUR-API-ID.execute-api.us-east-1.amazonaws.com/health
```

It should see: `{"status": "healthy", "use_s3": true}`


## Part 7: Build and Deploy Frontend

### Step 1: Update Frontend API URL

Update `frontend/components/bot.tsx` - find the fetch call and update:

```typescript
// Replace this line:
const response = await fetch('http://localhost:8000/chat', {

// With your API Gateway URL:
const response = await fetch('https://YOUR-API-ID.execute-api.us-east-1.amazonaws.com/chat', {
```

### Step 2: Configure for Static Export

First, update `frontend/next.config.ts` to enable static export:

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

```bash
cd frontend
npm run build
```

This creates an `out` directory with static files.


### Step 4: Upload to S3

Use the AWS CLI to upload the static files:

```bash
cd frontend
aws s3 sync out/ s3://YOUR-FRONTEND-BUCKET-NAME/ --delete
```


### Step 5: Test the Static Site

1. Go to the S3 frontend bucket → **Properties** → **Static website hosting**
2. Click the **Bucket website endpoint** URL
3. The MedExtract app should load

## Part 8: Set Up CloudFront

### Step 1: Get the S3 Website Endpoint

1. Go to S3 → the frontend bucket
2. Click **Properties** tab
3. Scroll to **Static website hosting**
4. Copy the **Bucket website endpoint** (looks like: `http://MedExtract-frontend-xxx.s3-website-us-east-1.amazonaws.com`)
5. Save this URL - This will need it for CloudFront

### Step 2: Create CloudFront Distribution

1. In AWS Console, search for **CloudFront**
2. Click **Create distribution**
3. **Step 1 - Origin:**
   - Distribution name: `MedExtract-distribution`
   - Click **Next**
4. **Step 2 - Add origin:**
   - Choose origin: Select **Other** (not Amazon S3!)
   - Origin domain name: Paste the S3 website endpoint WITHOUT the http://
     - Example: `MedExtract-frontend-xxx.s3-website-us-east-1.amazonaws.com`
   - **Origin protocol policy**: Select **HTTP only** (not HTTPS!)
     - This is because S3 static website hosting doesn't support HTTPS
     - If select HTTPS, it will get 504 Gateway Timeout errors
   - Origin name: `s3-static-website` (or leave auto-generated)
   - Leave other settings as default
   - Click **Add origin**
5. **Step 3 - Default cache behavior:**
   - Path pattern: Leave as `Default (*)`
   - Origin and origin groups: Select the origin
   - Viewer protocol policy: **Redirect HTTP to HTTPS**
   - Allowed HTTP methods: **GET, HEAD**
   - Cache policy: **CachingOptimized**
   - Click **Next**
6. **Step 4 - Web Application Firewall (WAF):**
   - Select **Do not enable security protections** 
   - Click **Next**
7. **Step 5 - Settings:**
   - Price class: **Use only North America and Europe**
   - Default root object: `index.html`
   - Click **Next**
8. **Review** and click **Create distribution**

### Step 3: Wait for Deployment

CloudFront takes 5-15 minutes to deploy globally.

### Step 4: Update CORS Settings

Update the Lambda to accept requests from CloudFront:

1. Go to Lambda → **Configuration** → **Environment variables**
2. Find the CloudFront distribution domain:
   - Go to CloudFront → Your distribution
   - Copy the **Distribution domain name** (like `d1234abcd.cloudfront.net`)
3. Edit the `CORS_ORIGINS` environment variable:
   - Current value: `*`
   - New value: `https://YOUR-CLOUDFRONT-DOMAIN.cloudfront.net`
   - Example: `https://d1234abcd.cloudfront.net`
4. Click **Save**

This allows Lambda function to accept requests only from the CloudFront distribution, improving security.

### Step 5: Invalidate CloudFront Cache

1. In CloudFront, select the distribution
2. Go to **Invalidations** tab
3. Click **Create invalidation**
4. Add path: `/*`
5. Click **Create invalidation**

## Part 9: Testing

### Step 1: Access the MedExtract

1. Go to the CloudFront URL: `https://YOUR-DISTRIBUTION.cloudfront.net`
2. The MedExtract should load with HTTPS.

### Step 2: Verify Memory in S3

1. Go to S3 → The memory bucket
2. The JSON files for each conversation session

### Step 3: Monitor CloudWatch Logs

1. Go to CloudWatch → **Log groups**
2. Find `/aws/lambda/MedExtract-api`
3. View recent logs to debug any issues


### Key Components

1. **CloudFront**: Global CDN, provides HTTPS, caches static content
2. **S3 Frontend Bucket**: Hosts static Next.js files
3. **API Gateway**: Manages API routes, handles CORS
4. **Lambda**: Runs your Python backend serverlessly
5. **S3 Memory Bucket**: Stores conversation history as JSON files
