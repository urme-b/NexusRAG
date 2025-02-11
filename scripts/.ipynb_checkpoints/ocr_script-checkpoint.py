import pytesseract
from pdf2image import convert_from_path

# Absolute path to the PDF file
pdf_file = '/Users/urmebose/Documents/NexusRAG/data/raw/Naval.pdf'

try:
    # Convert PDF to images
    pages = convert_from_path(pdf_file, 500)

    # Iterate through each page and extract text
    for page_number, page in enumerate(pages):
        # Save the page as an image file
        image_file = f'page_{page_number + 1}.jpg'
        page.save(image_file, 'JPEG')
        
        # Extract text from the image
        text = pytesseract.image_to_string(image_file)
        print(f"Text from page {page_number + 1}:")
        print(text)

except FileNotFoundError:
    print(f"The file {pdf_file} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")