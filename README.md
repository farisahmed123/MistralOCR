# MistralOCR - Medical Document OCR Processor

A Python tool that uses Mistral AI's OCR capabilities and Groq AI to extract and process medical information from documents (images and PDFs).

## Features

- 📄 Process PDF and image files (JPG, JPEG, PNG)
- 🔍 OCR powered by Mistral AI
- 🤖 AI-powered medical information extraction using Groq (FREE)
- 💊 Extracts patient details, medicine names, and dosages
- 📝 Saves results to text file

## Prerequisites

- Python 3.7+
- Mistral API key ([Get it here](https://console.mistral.ai/api-keys/))
- Groq API key ([Get it FREE here](https://console.groq.com/keys))

## Installation

1. Clone the repository:
```bash
git clone https://github.com/farisahmed123/MistralOCR.git
cd MistralOCR
```

2. Install required package:
```bash
pip install requests
```

## Configuration

Open `OCR.py` and add your API keys:

```python
MISTRAL_API_KEY = "your_mistral_api_key_here"
GROQ_API_KEY = "your_groq_api_key_here"
```

Update the document path to your file:

```python
document_path = r"C:\path\to\your\document.png"
```

## Usage

Run the script:

```bash
python OCR.py
```

The extracted medical information will be:
- Displayed in the console
- Saved to `medical_info_output.txt`

## Output Format

The tool extracts and formats:
- Patient Name
- Age
- Gender
- Medicine name(s) with strength
- Dosage instructions

## API Keys

### Mistral AI
- Sign up at [console.mistral.ai](https://console.mistral.ai/api-keys/)
- Create a new API key
- Used for OCR processing

### Groq AI (100% FREE)
- Sign up at [console.groq.com](https://console.groq.com/keys)
- Free tier available, no credit card required
- Used for AI-powered information extraction

## Supported File Types

- Images: `.jpg`, `.jpeg`, `.png`
- Documents: `.pdf`

## Example

```python
from OCR import MistralOCRProcessor

processor = MistralOCRProcessor(MISTRAL_API_KEY, GROQ_API_KEY)
result = processor.process_document("prescription.png", "output.txt")
print(result)
```

## License

MIT License

## Author

Faris Ahmed
