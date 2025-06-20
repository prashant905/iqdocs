# 🏥 Medical Document Analysis App

A powerful Python application for analyzing medical documents using cutting-edge AI models. Extract structured information from medical PDFs and images with support for multiple AI providers including OpenAI, Google Gemini, and Google's specialized MedGemma model.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Gradio](https://img.shields.io/badge/gradio-web--interface-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- 📎 **Multi-format Support**: Upload PDFs, images (JPG, PNG, GIF, BMP, WebP)
- 🤖 **Multiple AI Models**: 
  - **OpenAI GPT-4o**: Best-in-class multimodal analysis
  - **Google Gemini 1.5 Flash**: Fast and efficient processing
  - **MedGemma-4B**: Google's medical specialist model for local processing
- 🏥 **Medical Focus**: Specialized prompts for medical document analysis
- 📋 **Structured Extraction**: Extract specific medical fields with high accuracy
- 👀 **Visual Preview**: Real-time preview of uploaded documents
- 🔒 **Privacy Options**: Local processing with MedGemma for sensitive data
- 🎨 **Modern UI**: Clean, intuitive Gradio web interface
- ⚡ **System Prompts**: Pre-configured medical expert instructions

## 📋 Requirements

- **Python 3.8+** (Python 3.9+ recommended)
- **8GB+ RAM** (for MedGemma local processing)
- **Internet connection** (for OpenAI/Gemini APIs)
- **API Keys** (OpenAI and/or Gemini) or **Hugging Face account** (for MedGemma)

## 🚀 Quick Setup

### Method 1: Automated Setup (Recommended)

```bash
# Clone or download the project
cd medical-document-analysis

# Run the automated setup script
./setup.sh

# Set up your API keys
cp env.example .env
# Edit .env file with your API keys

# Start the application
source venv/bin/activate
python app.py
```

### Method 2: Manual Setup

Follow the detailed instructions below for complete control over the setup process.

## 📦 Detailed Installation

### Step 1: Environment Setup

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Core Dependencies:**
- `gradio>=4.0.0` - Web interface framework
- `openai>=1.0.0` - OpenAI API client
- `google-generativeai>=0.3.0` - Google Gemini API client
- `transformers>=4.50.0` - Hugging Face transformers (for MedGemma)
- `torch>=2.0.0` - PyTorch (for local model processing)
- `Pillow>=9.0.0` - Image processing
- `PyMuPDF>=1.23.0` - PDF processing
- `python-dotenv>=1.0.0` - Environment variable management
- `huggingface_hub>=0.25.0` - Hugging Face model access
- `accelerate>=0.20.0` - Model optimization

### Step 3: API Key Configuration

#### Option A: Using .env File (Recommended)

```bash
# Copy the example environment file
cp env.example .env

# Edit the .env file and add your API keys
nano .env  # or use your preferred editor
```

Add the following to your `.env` file:
```env
# OpenAI API Key (required for OpenAI model)
OPENAI_API_KEY=your-openai-api-key-here

# Google Gemini API Key (required for Gemini model)
GEMINI_API_KEY=your-gemini-api-key-here
```

#### Option B: Environment Variables

```bash
# Set environment variables directly
export OPENAI_API_KEY='your-openai-api-key-here'
export GEMINI_API_KEY='your-gemini-api-key-here'
```

#### How to Get API Keys

**OpenAI API Key:**
1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the generated key

**Google Gemini API Key:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key

### Step 4: MedGemma Setup (Optional)

MedGemma is Google's medical specialist model that runs locally for enhanced privacy.

#### Automated MedGemma Setup

```bash
# Run the MedGemma setup assistant
python setup_medgemma.py
```

This script will guide you through:
- Installing required packages
- Hugging Face authentication
- License acceptance
- Model access verification

#### Manual MedGemma Setup

```bash
# 1. Install Hugging Face CLI
pip install huggingface_hub

# 2. Login to Hugging Face
huggingface-cli login
# Enter your Hugging Face token when prompted

# 3. Accept the MedGemma license
# Visit: https://huggingface.co/google/medgemma-4b-it
# Click "Agree and access repository"

# 4. Test model access
python -c "from transformers import AutoTokenizer; print('Testing...'); AutoTokenizer.from_pretrained('google/medgemma-4b-it'); print('✅ MedGemma access successful!')"
```

**Getting a Hugging Face Token:**
1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Give it a name and select "Read" permissions
4. Copy the generated token

## 🎯 Running the Application

```bash
# Activate the virtual environment
source venv/bin/activate

# Start the application
python app.py

# The app will be available at: http://localhost:7860
```

**If port 7860 is in use:**
```bash
# Run on a different port
GRADIO_SERVER_PORT=7861 python app.py

# Or kill existing processes
lsof -ti:7860 | xargs kill -9
```

## 🎮 How to Use

### Basic Workflow

1. **Upload Documents**: 
   - Click the file upload area
   - Select PDF or image files (supports multiple files)

2. **Choose AI Model**:
   - **OpenAI**: Best for complex analysis, requires API key
   - **Gemini**: Fast and capable, requires API key
   - **MedGemma**: Medical specialist, runs locally (requires setup)

3. **Specify Fields to Extract**:
   ```
   1. Hospital/Facility Name
   2. Registration Number/Care Facility ID
   3. Complete Address
   4. Email Address
   5. Phone Number
   6. Provisional Diagnosis
   7. Encounter ID & Type
   8. Emergency Status (Yes/No)
   ```

4. **Analyze**: Click the "🚀 Analyze" button

5. **Review Results**: View extracted information and document previews

### Medical Field Examples

**Default Fields (Pre-filled):**
- Hospital/Facility information
- Registration and contact details
- Medical diagnoses and encounters
- Emergency status

**Custom Field Examples:**
```
- Patient Demographics (Name, Age, Gender, DOB)
- Vital Signs (Blood Pressure, Heart Rate, Temperature)
- Medications (Current medications, Dosages, Frequencies)
- Test Results (Lab values, Imaging findings)
- Treatment Plans (Procedures, Follow-up instructions)
- Provider Information (Doctor names, Specialties, Contact)
- Insurance Information (Policy numbers, Coverage details)
```

## 🔧 Configuration

### Default Settings

The application uses these default configurations:

```python
# Server Configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 7860

# AI Model Settings
OPENAI_MODEL = "gpt-4o"
GEMINI_MODEL = "gemini-1.5-flash"
MEDGEMMA_MODEL = "google/medgemma-4b-it"

# Processing Settings
MAX_TOKENS = 1000
TEMPERATURE = 0.7
```

### Customization

You can modify these settings in `app.py`:

```python
# Change the server port
app.launch(server_port=8080)

# Modify model parameters
response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    max_tokens=1500,  # Increase for longer responses
    temperature=0.3   # Lower for more consistent results
)
```

## 🛠️ Troubleshooting

### Common Issues & Solutions

#### 1. API Key Errors
```bash
# Error: OpenAI API key not found
# Solution: Check your .env file or environment variables
cat .env | grep OPENAI
export OPENAI_API_KEY='your-key-here'
```

#### 2. Port Already in Use
```bash
# Error: Address already in use
# Solution: Kill existing processes or use different port
lsof -ti:7860 | xargs kill -9
# or
GRADIO_SERVER_PORT=7861 python app.py
```

#### 3. MedGemma Loading Issues
```bash
# Error: Model not found or access denied
# Solution: Complete authentication
python setup_medgemma.py

# Or manually:
huggingface-cli login
# Then visit: https://huggingface.co/google/medgemma-4b-it
```

#### 4. Memory Issues (MedGemma)
```bash
# Error: CUDA out of memory or insufficient RAM
# Solution: Close other applications or use smaller model
# System Requirements: 8GB+ RAM recommended
```

#### 5. File Upload Errors
```bash
# Error: Unsupported file format
# Supported formats: PDF, JPG, JPEG, PNG, GIF, BMP, WebP
# Solution: Convert your file to a supported format
```

#### 6. Dependencies Issues
```bash
# Error: Module not found
# Solution: Reinstall requirements
pip install -r requirements.txt --force-reinstall

# For specific issues:
pip install transformers>=4.50.0
pip install torch>=2.0.0
```

### Debug Mode

Run with debug information:
```bash
# Enable verbose logging
GRADIO_DEBUG=1 python app.py

# Python debug mode
python -u app.py
```

### Getting Help

1. **Check Console Output**: Look for detailed error messages
2. **Verify Dependencies**: Ensure all packages are installed correctly
3. **Test API Keys**: Verify your API keys have sufficient credits
4. **Check System Resources**: Ensure adequate RAM/disk space
5. **Update Packages**: Keep dependencies up to date

## 📊 Model Comparison

| Feature | OpenAI GPT-4o | Gemini 1.5 Flash | MedGemma-4B |
|---------|---------------|-------------------|-------------|
| **Speed** | Fast | Very Fast | Moderate |
| **Accuracy** | Excellent | Very Good | Good (Medical) |
| **Medical Specialization** | General | General | ⭐ Specialized |
| **Privacy** | Cloud | Cloud | ⭐ Local |
| **Cost** | Pay per use | Pay per use | ⭐ Free |
| **Setup Complexity** | Easy | Easy | Moderate |
| **Internet Required** | Yes | Yes | ⭐ No |
| **Multimodal** | ⭐ Yes | ⭐ Yes | ⭐ Yes |

### When to Use Each Model

- **OpenAI GPT-4o**: Best overall performance, complex analysis
- **Gemini 1.5 Flash**: Quick processing, cost-effective
- **MedGemma-4B**: Privacy-sensitive medical data, offline processing

## 🔒 Privacy & Security

### Data Privacy Considerations

- **Cloud Models (OpenAI/Gemini)**: 
  - Data sent to external APIs
  - Subject to provider privacy policies
  - Not recommended for highly sensitive data

- **Local Model (MedGemma)**:
  - Processing happens entirely on your machine
  - No data leaves your system
  - ⭐ **Recommended for sensitive medical documents**

### Security Best Practices

1. **API Key Management**:
   ```bash
   # Use .env files (never commit to version control)
   echo ".env" >> .gitignore
   
   # Set proper file permissions
   chmod 600 .env
   ```

2. **Network Security**:
   ```bash
   # Run on localhost only for production
   # Change server_name="127.0.0.1" in app.py
   ```

3. **Data Handling**:
   - Clear uploaded files after processing
   - Use MedGemma for HIPAA-sensitive documents
   - Regular security updates

## 🚀 Advanced Usage

### Batch Processing

Process multiple documents programmatically:

```python
import os
from app import process_files_and_prompt

# Process multiple files
files = ["doc1.pdf", "doc2.jpg", "doc3.png"]
fields = "Hospital name, Patient ID, Diagnosis"
model = "MedGemma"  # For privacy

for file in files:
    result, images = process_files_and_prompt([file], fields, model)
    print(f"Results for {file}: {result}")
```

### Custom Medical Templates

Create specialized extraction templates:

```python
# Radiology Report Template
radiology_fields = """
1. Study Type (X-ray, CT, MRI, etc.)
2. Body Part Examined
3. Contrast Used (Yes/No)
4. Findings Summary
5. Impression/Diagnosis
6. Radiologist Name
7. Study Date and Time
"""

# Lab Report Template
lab_fields = """
1. Patient ID
2. Collection Date/Time
3. Test Names and Values
4. Reference Ranges
5. Critical Values (if any)
6. Ordering Physician
7. Lab Facility Name
"""
```

### Integration Examples

Integrate with other systems:

```python
# Example: Save results to database
import sqlite3

def save_to_database(patient_id, extracted_data):
    conn = sqlite3.connect('medical_records.db')
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO extractions (patient_id, data, timestamp)
        VALUES (?, ?, datetime('now'))
    """, (patient_id, extracted_data))
    conn.commit()
    conn.close()
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

- **Issues**: Report bugs via [GitHub Issues](https://github.com/yourusername/medical-document-analysis/issues)
- **Discussions**: Join conversations in [GitHub Discussions](https://github.com/yourusername/medical-document-analysis/discussions)
- **Email**: support@example.com

## 🙏 Acknowledgments

- **OpenAI** for GPT-4o multimodal capabilities
- **Google** for Gemini and MedGemma models
- **Hugging Face** for model hosting and transformers library
- **Gradio** for the intuitive web interface
- **PyMuPDF** for PDF processing capabilities

---

## 📊 Quick Reference

### Essential Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp env.example .env  # Add your API keys

# MedGemma Setup
python setup_medgemma.py

# Run Application
python app.py

# Troubleshooting
lsof -ti:7860 | xargs kill -9  # Kill port conflicts
GRADIO_SERVER_PORT=7861 python app.py  # Use different port
```

### File Formats Supported
- **Documents**: PDF
- **Images**: JPG, JPEG, PNG, GIF, BMP, WebP

### Memory Requirements
- **Minimum**: 4GB RAM
- **Recommended**: 8GB+ RAM (for MedGemma)
- **Optimal**: 16GB+ RAM (for multiple large documents)

---

**🎯 Ready to analyze medical documents with AI? Follow the setup guide above and start extracting structured information from your medical files!** 
>>>>>>> 82e2d87 (Initial setup)
