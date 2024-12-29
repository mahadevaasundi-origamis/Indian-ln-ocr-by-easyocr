import fitz  # PyMuPDF
import easyocr
from PIL import Image
import io
import numpy as np  # Import numpy

# Initialize the EasyOCR reader for Kannada and English
reader = easyocr.Reader(['en', 'kn'])  # 'kn' is for Kannada

# Path to the PDF containing images
pdf_path = r'optima-secure-plan_brochure_page-0001.pdf'  # Specify your local file path here

# Open the PDF using PyMuPDF
doc = fitz.open(pdf_path)

# Define a scaling factor to increase resolution (DPI)
scale_factor = 2  # Increase the scaling factor for higher DPI (2x -> 144 DPI, 3x -> 216 DPI, etc.)

# Initialize a list to store the content in the specified format
page_wise_content = []

# Loop through each page of the PDF
for page_num in range(len(doc)):
    page = doc.load_page(page_num)  # Load each page
    
    # Create a transformation matrix to scale the page
    matrix = fitz.Matrix(scale_factor, scale_factor)  # Scale by factor (default 1.0)
    
    # Convert the page to a pixmap (image) using the matrix for higher resolution
    pix = page.get_pixmap(matrix=matrix)  # Get the pixmap at the desired resolution

    # Convert the pixmap to a PIL Image object
    img_bytes = io.BytesIO(pix.tobytes("png"))  # Use PNG format for higher quality
    img = Image.open(img_bytes)

    # Convert the PIL Image to a NumPy array
    img_np = np.array(img)  # Convert to numpy array

    # Use EasyOCR to extract text from the image
    result = reader.readtext(img_np)  # Pass numpy array to readtext

    # Initialize a list to store the content for this page
    page_content = []

    # Collect text blocks for the current page
    page_text = ""
    for detection in result:
        coordinates = detection[0]  # Bounding box coordinates
        text = detection[1]  # Extracted text
        
        # We will concatenate all text into one string for each page
        page_text += text + "\n"  # Add a newline to separate the text blocks
        
        # For position, using the x-coordinate of the bounding box
        page_position = int(coordinates[0][0])  # Taking the first x-coordinate as the position
        page_content.append((page_num + 1, page_position, page_text.strip()))  # Store the page number, position, and text

    # Append the page's content to the page_wise_content list
    page_wise_content.append(page_content)

# Print the formatted page-wise content
print("------------------------------ page_wise_content -------------------------------------")
for page_content in page_wise_content:
    for content in page_content:
        print(content)
