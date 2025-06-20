import gradio as gr
from openai import OpenAI
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from pathlib import Path
import base64
from PIL import Image
import fitz  # PyMuPDF for PDF handling
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize MedGemma model (lazy loading)
medgemma_model = None
medgemma_tokenizer = None

def encode_image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def pdf_to_images(pdf_path):
    """Convert PDF pages to images"""
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append(img)
    doc.close()
    return images

def call_openai_api(images, fields_to_extract):
    """Call OpenAI API with images and fields to extract"""
    # System prompt for medical expert
    system_prompt = """Act as a medical expert specializing in medical document analysis. Please carefully examine this medical document/image and extract the following information with high accuracy. If any field is not visible or available, indicate "Not Available".

Format your response as a structured list with clear field names and extracted values."""
    
    # User prompt with specific fields
    user_prompt = f"Please extract the following fields from this medical document:\n\n{fields_to_extract}"
    
    # Prepare messages for OpenAI
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt}
            ]
        }
    ]
    
    # Add images to the user message
    for img in images:
        # Convert PIL image to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        messages[1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_base64}"
            }
        })
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def call_gemini_api(images, fields_to_extract):
    """Call Gemini API with images and fields to extract"""
    # System prompt for medical expert
    system_prompt = """Act as a medical expert specializing in medical document analysis. Please carefully examine this medical document/image and extract the following information with high accuracy. If any field is not visible or available, indicate "Not Available".

Format your response as a structured list with clear field names and extracted values."""
    
    # Configure model with system instruction
    model = genai.GenerativeModel(
        'gemini-1.5-flash',
        system_instruction=system_prompt
    )
    
    # User prompt with specific fields
    user_prompt = f"Please extract the following fields from this medical document:\n\n{fields_to_extract}"
    
    # Prepare content for Gemini
    content = [user_prompt]
    content.extend(images)
    
    response = model.generate_content(content)
    return response.text

def load_medgemma_model():
    """Load MedGemma model (lazy loading)"""
    global medgemma_model, medgemma_tokenizer
    
    if medgemma_model is None:
        try:
            # Load MedGemma 4B instruction-tuned model from Hugging Face
            # Note: This requires authentication and license acceptance
            model_name = "google/medgemma-4b-it"
            
            print("📥 Loading MedGemma model... This may take a few minutes.")
            medgemma_tokenizer = AutoTokenizer.from_pretrained(model_name)
            medgemma_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            print("✅ MedGemma model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading MedGemma model: {e}")
            print("💡 Setup instructions:")
            print("   1. Install huggingface_hub: pip install huggingface_hub")
            print("   2. Login to Hugging Face: huggingface-cli login")
            print("   3. Accept license at: https://huggingface.co/google/medgemma-4b-it")
            print("   4. Ensure you have sufficient RAM/GPU memory (requires 8GB+ RAM)")
            return False
    return True

def call_medgemma_api(images, fields_to_extract):
    """Call MedGemma API with fields to extract"""
    # Load model if not already loaded
    if not load_medgemma_model():
        return "Error: MedGemma model could not be loaded. Please check setup instructions above."
    
    # System prompt for medical expert (MedGemma format)
    system_prompt = """You are a medical expert specializing in medical document analysis. Please carefully examine the provided information and extract the requested medical information with high accuracy. If any field is not visible or available, indicate "Not Available".

Format your response as a structured list with clear field names and extracted values."""
    
    # Note: Current implementation uses text-based analysis
    # Future enhancement: Support for multimodal input with MedGemma-4B's vision capabilities
    image_description = f"I need you to extract the following medical fields from uploaded medical documents:\n\n{fields_to_extract}\n\nPlease provide a structured extraction with clear field names and values."
    
    # Create conversation format for instruction-tuned model
    conversation = f"<start_of_turn>user\n{system_prompt}\n\n{image_description}<end_of_turn>\n<start_of_turn>model\n"
    
    # Tokenize and generate
    inputs = medgemma_tokenizer(conversation, return_tensors="pt")
    
    # Move to same device as model
    device = next(medgemma_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = medgemma_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=medgemma_tokenizer.eos_token_id,
            eos_token_id=medgemma_tokenizer.eos_token_id
        )
    
    # Decode response
    response = medgemma_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove the input prompt)
    if "<start_of_turn>model" in response:
        generated_text = response.split("<start_of_turn>model")[-1].strip()
    else:
        generated_text = response[len(conversation):].strip()
    
    if not generated_text:
        return "MedGemma is ready but needs actual medical document content to analyze. Please note that the current implementation requires image-to-text preprocessing for optimal results."
    
    return generated_text

def process_files_and_prompt(files, fields_to_extract, model_choice):
    """Process uploaded files and send to AI model with fields to extract"""
    if not files:
        return "Please upload at least one file.", None
    
    if not fields_to_extract.strip():
        return "Please enter the fields to extract.", None
    
    # Check API keys/model availability based on model choice
    if model_choice == "OpenAI" and not openai_client.api_key:
        return "Please set your OpenAI API key in the .env file or as an environment variable: OPENAI_API_KEY", None
    elif model_choice == "Gemini" and not os.getenv("GEMINI_API_KEY"):
        return "Please set your Gemini API key in the .env file or as an environment variable: GEMINI_API_KEY", None
    elif model_choice == "MedGemma":
        # MedGemma will be loaded when first used (no API key needed)
        pass
    
    try:
        display_files = []
        uploaded_images = []
        
        # Process each uploaded file
        for file in files:
            file_path = file.name
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                # Handle image files
                img = Image.open(file_path)
                display_files.append(img)
                uploaded_images.append(img)
                
            elif file_extension == '.pdf':
                # Handle PDF files - convert to images
                pdf_images = pdf_to_images(file_path)
                display_files.extend(pdf_images)
                uploaded_images.extend(pdf_images)
            else:
                return f"Unsupported file type: {file_extension}. Please upload PDF or image files.", None
        
        # Call AI API based on model choice
        if model_choice == "OpenAI":
            answer = call_openai_api(uploaded_images, fields_to_extract)
        elif model_choice == "Gemini":
            answer = call_gemini_api(uploaded_images, fields_to_extract)
        else:  # MedGemma
            answer = call_medgemma_api(uploaded_images, fields_to_extract)
        
        return answer, display_files
        
    except Exception as e:
        return f"Error processing request: {str(e)}", None

def create_interface():
    """Create the Gradio interface"""
    with gr.Blocks(title="File Upload & OpenAI Analysis", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📁 File Upload & AI Analysis")
        gr.Markdown("Upload PDF or image files, enter a prompt, and get AI-powered analysis!")
        
        with gr.Row():
            with gr.Column(scale=1):
                # File upload section
                gr.Markdown("### 📎 Upload Files")
                file_upload = gr.File(
                    label="Upload PDF or Image files", 
                    file_count="multiple",
                    file_types=[".pdf", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]
                )
                
                # Model selection
                gr.Markdown("### 🤖 Choose AI Model")
                model_choice = gr.Radio(
                    choices=["OpenAI", "Gemini", "MedGemma"],
                    value="OpenAI",
                    label="AI Model",
                    info="OpenAI (GPT-4o) | Gemini (1.5 Flash) | MedGemma (Medical Specialist)"
                )
                
                # System prompt display
                gr.Markdown("### 🧠 System Instructions")
                gr.Markdown("*The AI has been pre-configured with medical expert knowledge:*")
                gr.Code(
                    "Act as a medical expert specializing in medical document analysis. Please carefully examine this medical document/image and extract the following information with high accuracy. If any field is not visible or available, indicate 'Not Available'. Format your response as a structured list with clear field names and extracted values.",
                    label="System Prompt (Pre-configured)"
                )
                
                # Fields input
                gr.Markdown("### 📋 Fields to Extract")
                fields_input = gr.Textbox(
                    label="Fields to Extract",
                    placeholder="Enter the specific medical fields you want to extract (e.g., Hospital name, Registration Number, Address, Email, Phone, etc.)",
                    lines=5,
                    value="1. Hospital/Facility Name\n2. Registration Number/Care Facility ID\n3. Complete Address\n4. Email Address\n5. Phone Number\n6. Provisional Diagnosis\n7. Encounter ID & Type\n8. Emergency Status (Yes/No)"
                )
                
                # Submit button
                submit_btn = gr.Button("🚀 Analyze", variant="primary", size="lg")
                
                # API key info
                gr.Markdown("### ⚙️ Setup")
                gr.Markdown("**API Key Configuration:**")
                gr.Markdown("Add your API keys to the `.env` file:")
                gr.Code("OPENAI_API_KEY=your-openai-key-here\nGEMINI_API_KEY=your-gemini-key-here")
                gr.Markdown("**Alternative:** Set as environment variables:")
                gr.Code("export OPENAI_API_KEY='your-openai-key'\nexport GEMINI_API_KEY='your-gemini-key'")
                gr.Markdown("**Note:** MedGemma runs locally (requires Hugging Face authentication)")
                
                # Model info
                gr.Markdown("### 📊 Model Comparison")
                gr.Markdown("""
                - **OpenAI (GPT-4o)**: Best for complex analysis, requires API key
                - **Gemini (1.5 Flash)**: Fast and capable, requires API key  
                - **MedGemma (4B)**: Medical specialist model, runs locally, requires HF authentication
                """)
                
                # MedGemma setup instructions
                gr.Markdown("### 🔐 MedGemma Setup (First Time)")
                gr.Markdown("""
                **Requirements:**
                1. Hugging Face account and authentication
                2. Accept the MedGemma license agreement
                3. At least 8GB RAM (16GB recommended)
                
                **Setup Steps:**
                ```bash
                # Install Hugging Face CLI
                pip install huggingface_hub
                
                # Login to Hugging Face
                huggingface-cli login
                
                # Accept license at: https://huggingface.co/google/medgemma-4b-it
                ```
                """)
            
            with gr.Column(scale=2):
                # Display uploaded files
                gr.Markdown("### 👀 Uploaded Files Preview")
                file_display = gr.Gallery(
                    label="Uploaded Files",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    height="300px"
                )
                
                # AI response
                gr.Markdown("### 🤖 AI Response")
                response_output = gr.Textbox(
                    label="Analysis Result",
                    lines=10,
                    max_lines=20,
                    show_copy_button=True
                )
        
        # Handle submission
        submit_btn.click(
            fn=process_files_and_prompt,
            inputs=[file_upload, fields_input, model_choice],
            outputs=[response_output, file_display]
        )
        
        # Example field sets
        gr.Markdown("### 💡 Example Field Sets")
        with gr.Row():
            gr.Examples(
                examples=[
                    "1. Hospital/Facility Name\n2. Registration Number/Care Facility ID\n3. Complete Address\n4. Email Address\n5. Phone Number\n6. Provisional Diagnosis\n7. Encounter ID & Type\n8. Emergency Status (Yes/No)",
                    "1. Patient Name\n2. Date of Birth\n3. Patient ID\n4. Insurance Information\n5. Primary Care Physician\n6. Admission Date\n7. Discharge Date",
                    "1. Facility Name\n2. License Number\n3. Contact Information\n4. Medical Record Number\n5. Service Type\n6. Date of Service\n7. Provider Name",
                    "1. Laboratory Name\n2. Test Results\n3. Reference Ranges\n4. Test Date\n5. Ordering Physician\n6. Critical Values\n7. Patient Demographics",
                    "1. Pharmacy Name\n2. Prescription Number\n3. Medication Name\n4. Dosage\n5. Quantity\n6. Prescriber Information\n7. Refill Information"
                ],
                inputs=fields_input,
                label="Click to use example field sets"
            )
    
    return app

if __name__ == "__main__":
    # Check API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    print("🔑 API Key Status:")
    if openai_key:
        print("✅ OpenAI API key loaded successfully!")
    else:
        print("⚠️  Warning: OPENAI_API_KEY not found!")
    
    if gemini_key:
        print("✅ Gemini API key loaded successfully!")
    else:
        print("⚠️  Warning: GEMINI_API_KEY not found!")
    
    if not openai_key and not gemini_key:
        print("\n❌ No API keys found! Please add at least one API key to your .env file:")
        print("OPENAI_API_KEY=your-openai-key-here")
        print("GEMINI_API_KEY=your-gemini-key-here")
    
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    ) 