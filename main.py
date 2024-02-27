from fastapi import FastAPI, File, UploadFile
from fastapi import Body, Cookie, File, Form, Header, Path, Query
from typing import Annotated
import facial_recognition.comparison as face_comparison

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/upload_file/")
async def create_upload_file(file1: UploadFile, file2: UploadFile):
    print(file1.filename)
    print(file2.filename)
    try:
        results = await face_comparison.compare_faces(file1, file2)
        return results
    except Exception as e:
        print("Error:", e)
        return {"Error": "Face detection failed"}
