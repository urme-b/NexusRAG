import fitz

def extract_text_from_pdf(pdf_file, output_txt_path=None):
    """
    Extract text from a non-scanned (normal) PDF using PyMuPDF.
    Writes to output_txt_path if provided, otherwise returns text as a string.
    """
    try:
        doc = fitz.open(pdf_file)
        all_text = []

        for page in doc:
            page_text = page.get_text("text")
            all_text.append(page_text)

        doc.close()
        final_text = "\n".join(all_text)

        if output_txt_path:
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(final_text)

        return final_text

    except FileNotFoundError:
        print(f"The file {pdf_file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")