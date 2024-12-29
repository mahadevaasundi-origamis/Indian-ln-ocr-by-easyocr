import logging
import easyocr
from io import BytesIO
from PIL import Image
import numpy as np
import fitz  # PyMuPDF

class OCRProcessor:
    def __init__(self, file_path, languages=None):
        self.file_path = file_path
        self.languages = languages if languages else ['en', 'kn']  # Default to English and Kannada
        
        # Initialize the EasyOCR Reader
        self.reader = easyocr.Reader(self.languages)  # Pass languages dynamically

    def increase_resolution(self, img, scale_factor=2):
        """Increase the image resolution (DPI) to improve OCR accuracy."""
        width, height = img.size
        img_resized = img.resize((int(width * scale_factor), int(height * scale_factor)), Image.LANCZOS)
        return img_resized

    def preprocess_image(self, img):
        """Preprocess image by converting to grayscale, increasing resolution, etc."""
        # Convert image to grayscale to improve OCR accuracy
        img_gray = img.convert('L')
        
        # Increase the resolution to enhance OCR
        img_high_res = self.increase_resolution(img_gray)

        # Optional: You can add thresholding or other techniques here if needed
        return img_high_res

    def extract_text_using_ocr(self, image_bytes, page_number):
        """Perform OCR using EasyOCR and return results in the required format."""
        try:
            # Convert the image bytes to a PIL Image object
            img = Image.open(BytesIO(image_bytes))

            # Preprocess the image (increase resolution, grayscale, etc.)
            img_preprocessed = self.preprocess_image(img)
            
            # Perform OCR using EasyOCR
            result = self.reader.readtext(np.array(img_preprocessed))  # Convert PIL image to numpy array for EasyOCR
            
            # Store text in the desired format: (page_number, position, text)
            page_map = []
            for detection in result:
                position = detection[0]  # The OCR bounding box coordinates (top-left and bottom-right)
                text = detection[1]  # The OCR detected text
                page_map.append((page_number, position[0][1], text))  # Position[0][1] is the Y-coordinate of the text
                
            return page_map
        except Exception as e:
            logging.error(f"Error during EasyOCR: {e}", exc_info=True)
            return []

    def process_pdf(self):
        """Process the PDF locally by converting it to images and running OCR."""
        try:
            # Open the PDF using fitz (PyMuPDF)
            pdf_document = fitz.open(self.file_path)

            page_maps = []
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)

                # Render the page to an image (pixmap)
                pix = page.get_pixmap(dpi=300)  # DPI set to 300 for better resolution

                # Convert the pixmap to a PIL Image object
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Convert image to bytes for OCR processing
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format="JPEG")
                img_bytes = img_byte_arr.getvalue()

                # Perform OCR on the image and get results in the required format
                page_map = self.extract_text_using_ocr(img_bytes, page_num + 1)
                page_maps.extend(page_map)

            return page_maps
        except Exception as e:
            logging.error(f"Error during PDF processing: {e}", exc_info=True)
            return []

# Main testing code
if __name__ == "__main__":
    # path = "Kannada_Images_pdf.pdf"  # Replace with your local PDF file path
    # path = "optima-secure-plan_brochure_page-0001.pdf"
    path = input("Enter the PDF file path: ")
    processor = OCRProcessor(path)
    try:
        result = processor.process_pdf()
        print("Processing Result:")
        print(result)
    except Exception as e:
        print(f"Error during testing: {e}")
