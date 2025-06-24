# 📄 Document Intelligence Hub

A powerful and streamlined Python application for extracting structured data from medical documents. This tool uses a sophisticated dual-model approach, leveraging both GPT-4o and Gemini 2.5 Flash for comprehensive text extraction, with Gemini handling the final, intelligent synthesis of the results.

![Python](https://img.shields.io/badge/python-v3.12+-blue.svg)
![Gradio](https://img.shields.io/badge/gradio-web--interface-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

-   📎 **Multi-format Support**: Upload PDFs and all common image types (JPG, PNG, etc.).
-   🤖 **Dual-Model AI Analysis**: Automatically processes documents with both **OpenAI GPT-4o** and **Google Gemini 2.5 Flash** to capture the most comprehensive data.
-   🧠 **Intelligent Synthesis**: Uses Gemini 2.5 Flash to intelligently analyze, disambiguate, and merge the outputs from both initial models into a single, accurate report.
-   📋 **Advanced Field Recognition**: Capable of extracting text from complex, handwritten forms, including checkboxes, ratings, and circled items.
-   👀 **Visual Preview**: Instantly see a preview of your uploaded documents.
-   🐛 **Debug Mode**: An optional interface to view the raw text output from each model side-by-side for comparison and troubleshooting.
-   🎨 **Modern UI**: A clean and intuitive web interface powered by Gradio.

## 📋 Requirements

-   **Python 3.12+**
-   [Poppler](https://poppler.freedesktop.org/): A PDF rendering library.
-   An active internet connection.
-   API keys for OpenAI and Google Gemini.

## 🚀 Setup Instructions

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### Step 2: Install System Dependencies (Poppler)

The application requires `poppler` to convert PDF pages into images.

**On macOS (using Homebrew):**

```bash
brew install poppler
```

**On Debian/Ubuntu:**

```bash
sudo apt-get update && sudo apt-get install -y poppler-utils
```

**On Windows:**
Follow the instructions from this [stackoverflow guide](https://stackoverflow.com/questions/18381713/how-to-install-poppler-on-windows) to install Poppler and add it to your system's PATH.

### Step 3: Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment using Python 3.12
python3.12 -m venv venv-py312

# Activate the virtual environment
# On macOS/Linux:
source venv-py312/bin/activate
# On Windows:
# venv-py312\Scripts\activate
```

### Step 4: Install Python Packages

Install all the required Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Step 5: Configure API Keys

The application loads API keys from a `.env` file in the project root.

1.  **Create the .env file:**
    ```bash
    touch .env
    ```

2.  **Add your API keys to the `.env` file:** Open the file in a text editor and add your keys like this:

    ```env
    OPENAI_API_KEY="your-openai-api-key-here"
    GEMINI_API_KEY="your-google-gemini-api-key-here"
    ```

## 🎯 Running the Application

Once the setup is complete, you can start the Gradio web server.

```bash
# Make sure your virtual environment is activated
source venv-py312/bin/activate

# Run the application
python app.py
```

The application will now be running and accessible at a local URL, typically `http://127.0.0.1:7860`.

## 🎮 How to Use

1.  **Upload Files**: Drag and drop or click to upload one or more PDF or image files. You will see a preview of the pages.
2.  **Define Fields**: In the "Fields to Extract" text box, list each piece of information you want the AI to find, with one field per line.
3.  **Run Analysis**: Click the "🚀 Run AI Analysis" button.
4.  **Review Results**: The final, synthesized information will appear in the "Final Extracted Fields" box.

## 🔧 Configuration

### Default Settings

The application uses these default configurations:

```python
# Server Configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 7860

# AI Model Settings
OPENAI_MODEL = "gpt-4o"
GEMINI_MODEL = "gemini-2.5-flash"

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

#### 3. Poppler Installation Issues
```bash
# Error: Poppler not found
# Solution: Install Poppler
brew install poppler  # macOS
sudo apt-get update && sudo apt-get install -y poppler-utils  # Debian/Ubuntu
```

#### 4. Memory Issues
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

| Feature | OpenAI GPT-4o | Gemini 2.5 Flash |
|---------|---------------|-------------------|
| **Speed** | Fast | Very Fast |
| **Accuracy** | Excellent | Very Good |
| **Medical Specialization** | General | General |
| **Privacy** | Cloud | Cloud |
| **Cost** | Pay per use | Pay per use |
| **Setup Complexity** | Easy | Easy |
| **Internet Required** | Yes | Yes |
| **Multimodal** | ⭐ Yes | ⭐ Yes |

### When to Use Each Model

- **OpenAI GPT-4o**: Best overall performance, complex analysis
- **Gemini 2.5 Flash**: Quick processing, cost-effective

## 🔒 Privacy & Security

### Data Privacy Considerations

- **Cloud Models (OpenAI/Gemini)**: 
  - Data sent to external APIs
  - Subject to provider privacy policies
  - Not recommended for highly sensitive data

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
   - Use Gemini for HIPAA-sensitive documents
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
model = "Gemini"  # For privacy

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
- **Google** for Gemini models
- **Gradio** for the intuitive web interface
- **PyMuPDF** for PDF processing capabilities

---

## 📊 Quick Reference

### Essential Commands

```bash
# Setup
python3.12 -m venv venv-py312 && source venv-py312/bin/activate
pip install -r requirements.txt
cp env.example .env  # Add your API keys

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
- **Recommended**: 8GB+ RAM (for Gemini)
- **Optimal**: 16GB+ RAM (for multiple large documents)

---

**🎯 Ready to analyze medical documents with AI? Follow the setup guide above and start extracting structured information from your medical files!**
