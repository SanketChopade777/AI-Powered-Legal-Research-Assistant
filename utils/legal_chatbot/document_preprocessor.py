import os
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageEnhance
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain.text_splitter import CharacterTextSplitter
import gc  # Garbage collection
import psutil  # For memory monitoring

# Configure absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = os.path.join(BASE_DIR, "temp_files")
os.makedirs(TEMP_DIR, exist_ok=True)

# Windows-specific configurations
POPPLER_PATH = r"E:\GCEK 22-26\4th YEAR\SEM-7\Seminar\poppler-24.08.0\Library\bin"
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Configure OCR engines
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Memory management settings
MAX_MEMORY_PERCENT = 85  # Stop if memory usage exceeds this


def get_system_memory():
    """Check current memory usage"""
    return psutil.virtual_memory().percent


def should_continue_processing():
    """Check if system has enough memory to continue"""
    return get_system_memory() < MAX_MEMORY_PERCENT


def is_scanned_pdf(file_path):
    """Optimized scanned PDF detection for large files"""
    try:
        with fitz.open(file_path) as doc:
            if len(doc) == 0:
                return True

            # Sample pages for large documents (check first, middle, last)
            total_pages = len(doc)
            sample_pages = []

            # Always check first page
            sample_pages.append(0)

            # Check middle page for large documents
            if total_pages > 10:
                sample_pages.append(total_pages // 2)

            # Check last page
            if total_pages > 1:
                sample_pages.append(total_pages - 1)

            text_found = 0
            for page_num in sample_pages:
                text = doc[page_num].get_text().strip()
                if len(text) > 100:  # Reasonable text threshold
                    text_found += 1

            # If no text found in samples, consider it scanned
            return text_found == 0

    except Exception as e:
        print(f"PDF analysis error: {str(e)}")
        return True


def enhance_image_for_ocr(image):
    """Optimized image pre-processing"""
    try:
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')

        # Moderate enhancements to avoid over-processing
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)

        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.3)

        return image
    except Exception as e:
        print(f"Image enhancement failed: {str(e)}")
        return image


def ocr_pdf(file_path):
    """High-performance OCR for unlimited pages with memory management"""
    try:
        # Check memory before starting
        if not should_continue_processing():
            print("⚠️ High memory usage - pausing OCR")
            return None

        # Create unique directory for this PDF
        pdf_name = os.path.splitext(os.path.basename(file_path))[0]
        pdf_temp_dir = os.path.join(TEMP_DIR, pdf_name)
        os.makedirs(pdf_temp_dir, exist_ok=True)

        print(f"🔄 Starting OCR for: {os.path.basename(file_path)}")

        # Get total pages first
        with fitz.open(file_path) as doc:
            total_pages = len(doc)
        print(f"📄 Total pages to process: {total_pages}")

        # Process in batches to manage memory
        batch_size = 20  # Process 20 pages at a time
        full_text = []

        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)

            # Check memory before each batch
            if not should_continue_processing():
                print("⚠️ Memory limit reached - stopping OCR batch")
                break

            print(f"🔍 Processing pages {batch_start + 1} to {batch_end}")

            try:
                # Convert current batch to images
                images = convert_from_path(
                    file_path,
                    dpi=300,  # Balanced quality/performance
                    first_page=batch_start + 1,
                    last_page=batch_end,
                    output_folder=pdf_temp_dir,
                    fmt="jpeg",
                    thread_count=2,
                    poppler_path=POPPLER_PATH,
                    grayscale=True,
                )

                batch_text = []
                for i, image in enumerate(images, batch_start + 1):
                    try:
                        # Enhance and OCR
                        processed_image = enhance_image_for_ocr(image)

                        text = pytesseract.image_to_string(
                            processed_image,
                            lang="eng",
                            config="--psm 6 --oem 3"
                        )

                        if text.strip():
                            batch_text.append(f"Page {i}:\n{text}")
                            print(f"✅ Page {i} processed")
                        else:
                            print(f"⚠️ No text on page {i}")

                        # Clean up
                        del processed_image

                    except Exception as e:
                        print(f"❌ Page {i} OCR failed: {str(e)}")
                        continue

                full_text.extend(batch_text)

                # Force garbage collection after each batch
                del images
                gc.collect()

                print(f"✅ Batch {batch_start // batch_size + 1} completed")

            except Exception as e:
                print(f"❌ Batch processing failed: {str(e)}")
                continue

        if not full_text:
            print("❌ No text extracted from any page")
            return None

        # Save combined text
        ocr_text_path = os.path.join(pdf_temp_dir, "ocr_output.txt")
        with open(ocr_text_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(full_text))

        print(f"💾 OCR completed: {len(full_text)} pages extracted")
        print(f"📁 Output saved to: {ocr_text_path}")
        return ocr_text_path

    except Exception as e:
        print(f"❌ OCR processing failed: {str(e)}")
        return None


def clean_ocr_text(text):
    """Enhanced text cleaning for OCR output"""
    if not text:
        return ""

    # Remove null characters and replacement characters
    text = text.replace('\x00', '').replace('\ufffd', '')

    # Fix common OCR errors (context-aware)
    corrections = [
        ('\n\n', '\n'),  # Reduce excessive newlines
        ('  ', ' '),  # Reduce multiple spaces
    ]

    for wrong, correct in corrections:
        text = text.replace(wrong, correct)

    # Normalize line breaks but preserve paragraph structure
    lines = text.split('\n')
    cleaned_lines = []

    current_paragraph = []
    for line in lines:
        line = line.strip()
        if line:
            current_paragraph.append(line)
        elif current_paragraph:
            # Empty line indicates paragraph break
            cleaned_lines.append(' '.join(current_paragraph))
            current_paragraph = []

    # Add last paragraph if exists
    if current_paragraph:
        cleaned_lines.append(' '.join(current_paragraph))

    return '\n\n'.join(cleaned_lines)


def load_document(file_path):
    """Unlimited document loader with optimized memory usage"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        if file_ext == ".pdf":
            if is_scanned_pdf(file_path):
                print(f"🔍 Processing scanned PDF: {file_path}")
                ocr_path = ocr_pdf(file_path)

                if ocr_path and os.path.exists(ocr_path):
                    try:
                        # Clean OCR text
                        with open(ocr_path, "r", encoding="utf-8", errors="ignore") as f:
                            raw_text = f.read()

                        cleaned_text = clean_ocr_text(raw_text)

                        # Save cleaned text
                        with open(ocr_path, "w", encoding="utf-8") as f:
                            f.write(cleaned_text)

                        # Load and split
                        loader = TextLoader(ocr_path, encoding="utf-8")
                        docs = loader.load()

                        if docs and docs[0].page_content.strip():
                            # Adaptive chunking based on content size
                            content_length = len(docs[0].page_content)
                            chunk_size = min(1000, max(500, content_length // 50))

                            splitter = CharacterTextSplitter(
                                separator="\n\n",
                                chunk_size=chunk_size,
                                chunk_overlap=100,
                                length_function=len
                            )

                            chunks = splitter.split_documents(docs)
                            print(f"✅ Split into {len(chunks)} chunks from OCR")
                            return chunks
                        else:
                            print("❌ No content after OCR cleaning")
                            return None

                    except Exception as e:
                        print(f"❌ Failed to process OCR results: {str(e)}")
                        return None
            else:
                # For text-based PDFs - process large files efficiently
                print(f"📄 Processing text-based PDF: {file_path}")

                # Get file size to decide processing strategy
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB

                if file_size > 50:  # Large file - process in memory-efficient way
                    print(f"📊 Large file detected: {file_size:.1f}MB")

                loader = PDFPlumberLoader(file_path)
                docs = loader.load()

                if docs:
                    splitter = CharacterTextSplitter(
                        chunk_size=1200,
                        chunk_overlap=150,
                        separator="\n\n"
                    )
                    chunks = splitter.split_documents(docs)
                    print(f"✅ Split into {len(chunks)} chunks from text PDF")
                    return chunks
                return docs

        elif file_ext == ".docx":
            print(f"📝 Processing DOCX: {file_path}")
            loader = Docx2txtLoader(file_path)
            docs = loader.load()

            if docs:
                splitter = CharacterTextSplitter(
                    chunk_size=1200,
                    chunk_overlap=150
                )
                chunks = splitter.split_documents(docs)
                print(f"✅ Split into {len(chunks)} chunks from DOCX")
                return chunks
            return docs

        elif file_ext == ".txt":
            print(f"📄 Processing TXT: {file_path}")
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()

            if docs:
                splitter = CharacterTextSplitter(
                    chunk_size=1200,
                    chunk_overlap=150
                )
                chunks = splitter.split_documents(docs)
                print(f"✅ Split into {len(chunks)} chunks from TXT")
                return chunks
            return docs

        else:
            print(f"❌ Unsupported file type: {file_ext}")
            return None

    except Exception as e:
        print(f"❌ Error loading {file_path}: {str(e)}")
        return None


def preprocess_documents(documents):
    """Memory-efficient document preprocessing"""
    if not documents:
        print("⚠️ No documents to preprocess")
        return []

    processed = []
    total_size = 0

    for i, doc in enumerate(documents):
        try:
            # Check memory periodically
            if i % 100 == 0 and not should_continue_processing():
                print("⚠️ Memory high - stopping preprocessing")
                break

            content = doc.page_content

            # Skip empty documents
            if not content or not content.strip():
                continue

            # Enhanced cleaning
            content = clean_ocr_text(content)

            # Skip if content is too short after cleaning
            if len(content.strip()) < 20:
                continue

            doc.page_content = content
            processed.append(doc)
            total_size += len(content)

        except Exception as e:
            print(f"❌ Document {i} processing error: {str(e)}")
            continue

    print(f"✅ Preprocessed {len(processed)} documents ({total_size / 1024 / 1024:.1f}MB)")
    return processed


def batch_process_files(file_paths, batch_size=5):
    """Process files in batches to manage memory"""
    all_documents = []

    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i + batch_size]
        print(f"\n🔄 Processing batch {i // batch_size + 1}/{(len(file_paths) - 1) // batch_size + 1}")

        # Check memory before batch
        if not should_continue_processing():
            print("⚠️ Memory limit reached - stopping batch processing")
            break

        for file_path in batch:
            print(f"\n📂 Processing: {os.path.basename(file_path)}")
            docs = load_document(file_path)
            if docs:
                all_documents.extend(docs)

        # Clean up between batches
        gc.collect()

    return all_documents