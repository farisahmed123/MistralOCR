import requests
import json
import time
from pathlib import Path

class MistralOCRProcessor:
    def __init__(self, mistral_api_key, groq_api_key):
        """
        Initialize the OCR processor with API keys
        
        Args:
            mistral_api_key: API key for Mistral AI
            groq_api_key: API key for Groq (FREE - get from console.groq.com)
        """
        self.mistral_api_key = mistral_api_key
        self.groq_api_key = groq_api_key
        self.mistral_base_url = "https://api.mistral.ai/v1"
        self.groq_base_url = "https://api.groq.com/openai/v1"
    
    def upload_to_mistral(self, file_path):
        """
        Upload file to Mistral AI for OCR processing
        
        Args:
            file_path: Path to the document file
            
        Returns:
            dict: Response containing file ID
        """
        url = f"{self.mistral_base_url}/files"
        headers = {
            "Authorization": f"Bearer {self.mistral_api_key}"
        }
        
        with open(file_path, 'rb') as f:
            files = {
                'file': f,
                'purpose': (None, 'ocr')
            }
            response = requests.post(url, headers=headers, files=files)
        
        response.raise_for_status()
        return response.json()
    
    def get_signed_url(self, file_id, expiry_hours=24):
        """
        Get a signed URL for the uploaded file
        
        Args:
            file_id: ID of the uploaded file
            expiry_hours: URL expiry time in hours
            
        Returns:
            dict: Response containing signed URL
        """
        url = f"{self.mistral_base_url}/files/{file_id}/url"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.mistral_api_key}"
        }
        params = {
            "expiry": str(expiry_hours)
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_ocr_results_image(self, signed_url):
        """
        Get OCR results for an image
        
        Args:
            signed_url: Signed URL of the image
            
        Returns:
            dict: OCR results
        """
        url = f"{self.mistral_base_url}/ocr"
        headers = {
            "Authorization": f"Bearer {self.mistral_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "mistral-ocr-latest",
            "document": {
                "type": "image_url",
                "image_url": signed_url
            },
            "include_image_base64": True
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_ocr_results_pdf(self, signed_url):
        """
        Get OCR results for a PDF
        
        Args:
            signed_url: Signed URL of the PDF
            
        Returns:
            dict: OCR results
        """
        url = f"{self.mistral_base_url}/ocr"
        headers = {
            "Authorization": f"Bearer {self.mistral_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "mistral-ocr-latest",
            "document": {
                "type": "document_url",
                "document_url": signed_url
            },
            "include_image_base64": True
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    
    def extract_medical_info(self, markdown_text):
        """
        Extract medical information using Groq AI (FREE)
        
        Args:
            markdown_text: OCR text in markdown format
            
        Returns:
            str: Extracted medical information
        """
        url = f"{self.groq_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        system_message = """Extract only the following from the medical document:
- Patient details (name, age, gender, etc.)
- Medicine name(s) with strength (e.g., Paracetamol 500mg)
- Dosage(s)

Ignore all other information.

Format your response as:
Patient Name: [name or "Not found"]
Age: [age or "Not found"]
Gender: [gender or "Not found"]
Medicine: [medicine name with strength]
Dosage: [dosage instructions]

If there are multiple medicines, list each with its strength and dosage."""
        
        payload = {
            "model": "llama-3.3-70b-versatile",  # FREE Groq model
            "messages": [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": f"Medical Document:\n\n{markdown_text}"
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1024
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    
    def clean_data(self, ocr_result):
        """
        Clean and extract text from OCR result
        
        Args:
            ocr_result: Raw OCR result
            
        Returns:
            str: Cleaned extracted text
        """
        if 'output' in ocr_result:
            return ocr_result['output']
        elif 'pages' in ocr_result and len(ocr_result['pages']) > 0:
            return ocr_result['pages'][0].get('markdown', '')
        return ''
    
    def save_to_file(self, data, output_path='ocr_output.txt'):
        """
        Save extracted data to a text file
        
        Args:
            data: Data to save
            output_path: Path to output file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(data)
    
    def process_document(self, file_path, output_path='ocr_output.txt'):
        """
        Process a document end-to-end: upload, OCR, extract info, save
        
        Args:
            file_path: Path to the document
            output_path: Path to save the output
            
        Returns:
            str: Extracted medical information
        """
        file_extension = Path(file_path).suffix.lower()
        is_pdf = file_extension == '.pdf'
        is_image = file_extension in ['.jpg', '.jpeg', '.png']
        
        if not (is_pdf or is_image):
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        upload_response = self.upload_to_mistral(file_path)
        file_id = upload_response['id']
        
        url_response = self.get_signed_url(file_id)
        signed_url = url_response['url']
        
        if is_pdf:
            ocr_result = self.get_ocr_results_pdf(signed_url)
        else:
            ocr_result = self.get_ocr_results_image(signed_url)
        
        markdown_text = self.clean_data(ocr_result)
        
        if is_pdf and 'pages' in ocr_result:
            markdown_text = ocr_result['pages'][0].get('markdown', '')
        
        extracted_info = self.extract_medical_info(markdown_text)
        self.save_to_file(extracted_info, output_path)
        
        return extracted_info


if __name__ == "__main__":
    import os
    
    # Add your API keys here
    MISTRAL_API_KEY = "**************************************"
    GROQ_API_KEY = "**************************************"
    
    processor = MistralOCRProcessor(MISTRAL_API_KEY, GROQ_API_KEY)
    document_path = r"C:\Users\PC\OneDrive\Documents\aaa.png"
    
    if os.path.exists(document_path):
        try:
            result = processor.process_document(document_path, "medical_info_output.txt")
            print("\n" + result)
            print("\n✅ Saved to: medical_info_output.txt")
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print(f"File not found: {document_path}")