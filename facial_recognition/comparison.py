from deepface import DeepFace
from starlette.datastructures import UploadFile
import numpy as np
from PIL import Image
import io


async def upload_file_to_numpy_array(upload_file: UploadFile) -> np.ndarray:
    """Convert FastAPI UploadFile -> Numpy Array"""
    print("ENTERING")
    # Read the content of the uploaded file
    contents = await upload_file.read()

    # Use BytesIO to convert bytes data to a file-like object for PIL
    img = Image.open(io.BytesIO(contents))

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Don't forget to close the file (UploadFile)
    await upload_file.close()

    return img_array


async def compare_faces(image1: UploadFile, image2: UploadFile):
    # use DeepFace to verify the two images are the same
    numpy1 = await upload_file_to_numpy_array(image1)
    numpy2 = await upload_file_to_numpy_array(image2)
    result = DeepFace.verify(numpy1, numpy2)
    return result
    