from typing import Optional
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, desc, DateTime, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from utils import *
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from typing import Dict

load_dotenv()  # take environment variables from .env.
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Create SQLite database engine
SQLALCHEMY_DATABASE_URL = "sqlite:///./Database.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for SQLAlchemy models
Base = declarative_base()
Base.metadata.create_all(bind=engine)


class TestRagbyDataAsLinkPydanticModel(BaseModel):
    url: str
    action: str  #loadhostedxml,usesitemap,usesinglelink,crawl,useutubelink
    max_pages_to_crawl: Optional[int] = 10
    number_of_testcases: int
    api_endpoint: str
    api_sample_data: Dict
    api_method: str


class TestRagbyDataAsFilePydanticModel(BaseModel):
    number_of_testcases: int
    api_endpoint: str
    api_sample_data: Dict
    api_method: str


# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
UPLOAD_DIRECTORY = "uploads"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


@app.post("/TestRagWithDataAsFile/")
async def TestRagWithDataAsFile(
        number_of_testcases: int = Form(...),
        api_endpoint: str = Form(...),
        api_sample_data: str = Form(...),  # We will parse this to dict
        api_method: str = Form(...),
        file: UploadFile = File(...)):
    try:
        # Convert the api_sample_data from JSON string to dict

        api_sample_data_dict = json.loads(api_sample_data)

        # Create the Pydantic model instance
        entry = TestRagbyDataAsFilePydanticModel(
            number_of_testcases=number_of_testcases,
            api_endpoint=api_endpoint,
            api_sample_data=api_sample_data_dict,
            api_method=api_method)
        file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_location, "wb") as f:
            f.write(file.file.read())
        documents, fileformat = load_documents(file_location)
        documents = RetunCHunkedocs(documents, 1, 1, "chroma",
                                    "SemanticChunker using OpenAI Embeddings")
        resultant_frame = pipeline(documents, entry.number_of_testcases,
                                   entry.api_endpoint, entry.api_sample_data,
                                   entry.api_method)
        df_json = resultant_frame.to_json(orient="records")
        metrics = getAllMetrics(resultant_frame)
        # Prepare the combined response
        results = {
            "Complete_Results_Table":
            df_json,  # Convert JSON string to dictionary
            "Evaluation_Metrics": str(metrics)
        }
        print(
            "--------------------------------------------------------------------"
        )

        return JSONResponse(content=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)


@app.post("/TestRagWithDataAsLink/")
async def TestRagWithDataAsLink(entry: TestRagbyDataAsLinkPydanticModel):
    try:
        documents, fileformat = loadContentFromWeb(entry.url, entry.action)
        documents = RetunCHunkedocs(documents, 1, 1, "chroma",
                                    "SemanticChunker using OpenAI Embeddings")
        resultant_frame = pipeline(documents, entry.number_of_testcases,
                                   entry.api_endpoint,
                                   f"""{str(entry.api_sample_data)}""",
                                   entry.api_method)
        df_json = resultant_frame.to_json(orient="records")
        metrics = getAllMetrics(resultant_frame)
        # Prepare the combined response
        results = {
            "Complete_Results_Table":
            df_json,  # Convert JSON string to dictionary
            "Evaluation_Metrics": str(metrics)
        }
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
