import logging
import easyocr
from io import BytesIO
from PIL import Image
import numpy as np
import fitz  # PyMuPDF
import re

class OCRProcessor:
    def __init__(self, file_path, languages=None, scale_factor=1.5, dpi=200):
        """
        Initialize OCRProcessor with scaling and DPI optimization.
        """
        self.file_path = file_path
        self.languages = languages if languages else ['en', 'kn']  # Default to English and Kannada
        self.scale_factor = scale_factor  # Optimized scale factor
        self.dpi = dpi  # Optimized DPI
        
        # Initialize EasyOCR Reader
        self.reader = easyocr.Reader(self.languages)

    def increase_resolution(self, img):
        """
        Increase image resolution with optimized scaling.
        """
        width, height = img.size
        img_resized = img.resize(
            (int(width * self.scale_factor), int(height * self.scale_factor)), 
            Image.LANCZOS
        )
        return img_resized

    def preprocess_image(self, img):
        """
        Preprocess image to optimize OCR performance.
        """
        # Convert to grayscale
        img_gray = img.convert('L')
        # Increase resolution
        img_high_res = self.increase_resolution(img_gray)
        return img_high_res

    def extract_text_using_ocr(self, image_bytes, page_number):
        """
        Extract text from preprocessed image and chunk it by lines and sentences.
        """
        try:
            img = Image.open(BytesIO(image_bytes))
            img_preprocessed = self.preprocess_image(img)
            
            result = self.reader.readtext(
                np.array(img_preprocessed), 
                detail=0  # Disable bounding box details to speed up OCR
            )
            
            # Combine all text into one string
            combined_text = " ".join(result)
            
            # Split text into logical chunks (lines and sentences)
            # Split by newline and sentence-ending punctuation
            lines_and_sentences = re.split(r'(?<=[.!?])\s+|\n', combined_text)
            
            # Format the result as (page_number, position, text_chunk)
            chunks = []
            position = 0
            for chunk in lines_and_sentences:
                chunk = chunk.strip()
                if chunk:
                    chunks.append((page_number, position, chunk))
                    position += len(chunk) + 1  # Update position for the next chunk
            
            return chunks
        
        except Exception as e:
            logging.error(f"Error during OCR: {e}", exc_info=True)
            return []

    def process_pdf(self):
        """
        Process PDF using optimized rendering and OCR.
        """
        try:
            pdf_document = fitz.open(self.file_path)
            page_maps = []
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                
                # Render page with optimized DPI
                pix = page.get_pixmap(dpi=self.dpi)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Convert image to bytes for OCR
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format="JPEG", quality=70)  # Lower image quality for faster OCR
                img_bytes = img_byte_arr.getvalue()
                
                # Perform OCR
                page_map = self.extract_text_using_ocr(img_bytes, page_num + 1)
                page_maps.extend(page_map)
            
            return page_maps
        
        except Exception as e:
            logging.error(f"Error during PDF processing: {e}", exc_info=True)
            return []

# Main testing code
if __name__ == "__main__":
    # path = "optima-secure-plan_brochure_page-0001.pdf"
    path = input("Enter the PDF file path: ")
    processor = OCRProcessor(path, scale_factor=1.5, dpi=200)
    
    try:
        result = processor.process_pdf()
        print("Processing Result:")
        for page_chunk in result:
            print(page_chunk)
    except Exception as e:
        print(f"Error during testing: {e}")


