import os
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageEnhance
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredFileLoader
)
from langchain.text_splitter import CharacterTextSplitter

# Configure absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = os.path.join(BASE_DIR, "temp_files")
os.makedirs(TEMP_DIR, exist_ok=True)

# Windows-specific configurations
POPPLER_PATH = r"E:\GCEK 22-26\4th YEAR\SEM-7\Seminar\poppler-24.08.0\Library\bin"
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Configure OCR engines
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Uncomment and configure for Windows if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def is_scanned_pdf(file_path):
    """Improved scanned PDF detection with better error handling"""
    try:
        with fitz.open(file_path) as doc:
            if len(doc) == 0:
                return True

            text_pages = sum(
                1 for page in doc
                if page.get_text().strip()
            )
            return (text_pages / len(doc)) < 0.2
    except Exception as e:
        print(f"PDF analysis error: {str(e)}")
        return True


def enhance_image_for_ocr(image):
    """Pre-process image to improve OCR accuracy"""
    try:
        # Convert to grayscale
        image = image.convert('L')

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)

        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)

        return image
    except Exception as e:
        print(f"Image enhancement failed: {str(e)}")
        return image


def ocr_pdf(file_path):
    """Robust OCR processing with image pre-processing"""
    try:
        # Create unique directory for this PDF
        pdf_name = os.path.splitext(os.path.basename(file_path))[0]
        pdf_temp_dir = os.path.join(TEMP_DIR, pdf_name)
        os.makedirs(pdf_temp_dir, exist_ok=True)

        # Convert PDF to images
        images = convert_from_path(
            file_path,
            dpi=400,  # Higher resolution
            output_folder=pdf_temp_dir,
            fmt="jpeg",
            thread_count=4,
            poppler_path=POPPLER_PATH,
            grayscale=True  # Convert to grayscale early
        )

        full_text = []
        for i, image in enumerate(images, 1):
            try:
                # Enhance image quality
                processed_image = enhance_image_for_ocr(image)

                # Save processed image for debugging
                debug_img_path = os.path.join(pdf_temp_dir, f"processed_page_{i}.jpg")
                processed_image.save(debug_img_path, "JPEG")

                # Try multiple OCR configurations
                configs = [
                    "--psm 6",  # Assume uniform block of text
                    "--psm 11",  # Sparse text
                    "--oem 3"  # Default OCR engine mode
                ]

                for config in configs:
                    try:
                        text = pytesseract.image_to_string(
                            processed_image,
                            lang="eng",
                            config=config
                        )
                        if text.strip():
                            break
                    except:
                        continue

                if text.strip():
                    full_text.append(text)
                    print(f"Successfully extracted text from page {i}")
                else:
                    print(f"No text found on page {i}")

            except Exception as e:
                print(f"Page {i} processing failed: {str(e)}")
                continue

        if not full_text:
            print("Warning: No text extracted from any page")
            return None

        # Save combined text
        ocr_text_path = os.path.join(pdf_temp_dir, "ocr_output.txt")
        with open(ocr_text_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(full_text))

        print(f"OCR results saved to: {ocr_text_path}")
        return ocr_text_path

    except Exception as e:
        print(f"OCR processing failed: {str(e)}")
        return None


def load_document(file_path):
    """Document loader with comprehensive error handling and OCR text cleaning"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        if file_ext == ".pdf":
            if is_scanned_pdf(file_path):
                print(f"Processing scanned PDF: {file_path}")
                ocr_path = ocr_pdf(file_path)

                if ocr_path and os.path.exists(ocr_path):
                    try:
                        # --- Clean OCR text before loading ---
                        with open(ocr_path, "r", encoding="utf-8", errors="ignore") as f:
                            cleaned_text = f.read().replace('\x00', '').replace('\ufffd', '')
                            cleaned_text = ' '.join(cleaned_text.split())  # normalize spaces

                        with open(ocr_path, "w", encoding="utf-8") as f:
                            f.write(cleaned_text)

                        # Load single file
                        loader = TextLoader(ocr_path, encoding="utf-8")
                        docs = loader.load()

                        # --- Split into smaller chunks (simulate multiple pages) ---
                        splitter = CharacterTextSplitter(
                            separator="\n\n",  # split by double newlines
                            chunk_size=1500,   # adjust as needed
                            chunk_overlap=100
                        )
                        docs = splitter.split_documents(docs)

                        print(f"Successfully split into {len(docs)} chunks from OCR")
                        return docs
                    except Exception as e:
                        print(f"Failed to load OCR results: {str(e)}")
            else:
                return PDFPlumberLoader(file_path).load()

        elif file_ext == ".docx":
            return Docx2txtLoader(file_path).load()

        elif file_ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            splitter = CharacterTextSplitter(
                separator="\n\n", chunk_size=1500, chunk_overlap=100
            )
            return splitter.split_documents(docs)

        else:
            print(f"Unsupported file type: {file_ext}")
            return None

    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None



def preprocess_documents(documents):
    """Enhanced document cleaning"""
    if not documents:
        return []

    processed = []
    for doc in documents:
        try:
            # Advanced cleaning pipeline
            content = doc.page_content
            content = content.replace('\x00', '').replace('\ufffd', '')
            content = ' '.join(content.split())  # Normalize whitespace
            doc.page_content = content
            processed.append(doc)
        except Exception as e:
            print(f"Document processing error: {str(e)}")
            continue

    return processed