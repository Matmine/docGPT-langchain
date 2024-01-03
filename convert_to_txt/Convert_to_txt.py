import PyPDF2

class PDFConverter:
    def convert_pdf_to_txt(self, pdf_file, txt_file):
        try:
            with open(pdf_file, 'rb') as pdf:
                pdf_reader = PyPDF2.PdfReader(pdf)
                text = ''
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
                
                with open(txt_file, 'w', encoding='utf-8') as txt:
                    txt.write(text)
                print(f"PDF converted to TXT. Saved as {txt_file}")
        except FileNotFoundError:
            print("File not found.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

# Create an instance of the PDFConverter class
converter = PDFConverter()

# Call the convert_pdf_to_txt method by using the instance
converter.convert_pdf_to_txt("convert_to_txt/m--douchin.pdf", "convert_to_txt/m--douchin.txt")