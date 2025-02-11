import pytesseract
from pdf2image import convert_from_path
import os

def ocr_pdf(pdf_file, output_txt_path=None):
    """
    Converts each page of a scanned PDF into images, then runs Tesseract OCR.
    If output_txt_path is provided, writes the recognized text there.
    Otherwise returns the text as a single string.
    """
    try:
        # Convert PDF pages to images (300-500 dpi is typical, here it's 500 for clarity).
        pages = convert_from_path(pdf_file, 500)
        all_text = []

        for page_number, page in enumerate(pages):
            image_file = f"page_{page_number + 1}.jpg"
            # Save each page as a temporary image
            page.save(image_file, 'JPEG')

            # OCR the image
            text = pytesseract.image_to_string(image_file)
            print(f"Text from page {page_number + 1}:")
            print(text)
            all_text.append(text)

            # Cleanup: remove the image file if desired
            if os.path.exists(image_file):
                os.remove(image_file)

        final_text = "\n".join(all_text)

        if output_txt_path:
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(final_text)

        return final_text

    except FileNotFoundError:
        print(f"The file {pdf_file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")