import pytesseract
from pdf2image import convert_from_path
import os

def ocr_pdf(pdf_path, output_dir):
    # Convert PDF pages to images
    pages = convert_from_path(pdf_path, 300)
    
    # Perform OCR on each page
    text = ""
    for i, page in enumerate(pages):
        image_path = os.path.join(output_dir, f'page_{i+1}.png')
        page.save(image_path, 'PNG')
        text += pytesseract.image_to_string(image_path)
        os.remove(image_path)  # Clean up image file after OCR
    
    return text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Perform OCR on scanned PDFs.')
    parser.add_argument('pdf_path', type=str, help='Path to the scanned PDF file.')
    parser.add_argument('output_file', type=str, help='Path to the output text file.')
    args = parser.parse_args()
    
    # Perform OCR and save to output file
    ocr_text = ocr_pdf(args.pdf_path, 'temp_images')
    with open(args.output_file, 'w') as f:
        f.write(ocr_text)