import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gradio as gr
from openai import OpenAI
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path
import base64
from PIL import Image
from pdf2image import convert_from_path
import io
from dotenv import load_dotenv
import asyncio
import concurrent.futures

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
    """Convert PDF pages to images using pdf2image"""
    try:
        images = convert_from_path(pdf_path)
        print(f"[DEBUG] Extracted {len(images)} images from PDF: {pdf_path}")
        return images
    except Exception as e:
        print(f"[ERROR] Failed to convert PDF to images: {pdf_path} | Error: {e}")
        if "No such file or directory: 'pdftoppm'" in str(e):
             raise gr.Error("Poppler not found. Please install Poppler to process PDF files. On macOS: `brew install poppler`")
        else:
            raise gr.Error(f"An error occurred while processing the PDF: {e}")
        return []

def get_medgemma_model():
    """Initializes and returns the MedGemma model and tokenizer."""
    global medgemma_model, medgemma_tokenizer
    if medgemma_model is None or medgemma_tokenizer is None:
        print("Initializing MedGemma model...")
        model_id = "google/medgemma-2b"
        medgemma_tokenizer = AutoTokenizer.from_pretrained(model_id)
        medgemma_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("MedGemma model initialized.")
    return medgemma_model, medgemma_tokenizer

def call_gpt4(file_paths, progress=gr.Progress()):
    """Call OpenAI API with files (PDFs and images) directly"""
    system_prompt = "You are an expert medical document analyzer. Your task is to  extract ALL  content from the provided documents. Focus on accuracy and completeness, including handwritten notes."
    user_prompt = """Please extract all content from these medical documents, including any handwritten notes, printed text, and form fields. Pay special attention to:

1. CHECKBOXES & SELECTIONS: Look for checked boxes, circled options, crossed marks, or any visual indicators of selected choices (Yes/No, checkmarks, X marks, etc.)
2. RATING SCALES: Identify any circled or marked numbers on rating scales (0-10 pain scales, etc.)
3. FORM FIELDS: Capture both the field labels and any filled-in values
4. TABLE DATA: Extract information from tables including any marked or filled cells
5. HANDWRITTEN ENTRIES: Include any handwritten text, numbers, or symbols

For any selections or markings, clearly indicate what was selected (e.g., 'Pain scale: 5 circled', 'Tobacco: Y marked', 'Negative column checked').

Be thorough and accurate."""
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": [{"type": "text", "text": user_prompt}]}]
    
    for file_path in file_paths:
        file_extension = Path(file_path).suffix.lower()
        if file_extension == '.pdf':
            images = pdf_to_images(file_path)
            for img in images:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                messages[1]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}})
        else:
            img = Image.open(file_path)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            messages[1]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}})

    try:
        response = openai_client.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=4000)
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API Error: {str(e)}"

def call_gemini(file_paths, progress=gr.Progress()):
    """Call Gemini API with files (PDFs and images) directly"""
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = """You are an expert medical document analyzer. Extract ALL text content from the provided documents, including handwritten notes. Pay special attention to:

1. CHECKBOXES & SELECTIONS: Look for checked boxes, circled options, crossed marks, or any visual indicators of selected choices (Yes/No, checkmarks, X marks, etc.)
2. RATING SCALES: Identify any circled or marked numbers on rating scales (0-10 pain scales, etc.)  
3. FORM FIELDS: Capture both the field labels and any filled-in values
4. TABLE DATA: Extract information from tables including any marked or filled cells
5. HANDWRITTEN ENTRIES: Include any handwritten text, numbers, or symbols

For any selections or markings, clearly indicate what was selected (e.g., 'Pain scale: 5 circled', 'Tobacco: Y marked', 'Negative column checked').

Prioritize accuracy and completeness."""
    content = [prompt]
    
    for file_path in file_paths:
        file_extension = Path(file_path).suffix.lower()
        if file_extension == '.pdf':
            try:
                pdf_images = pdf_to_images(file_path)
                content.extend(pdf_images)
            except Exception as e:
                print(f"[WARNING] Could not convert PDF for Gemini: {file_path} | Error: {e}")
        else:
            img = Image.open(file_path)
            content.append(img)
    try:
        response = model.generate_content(content)
        return response.text
    except Exception as e:
        return f"Gemini API Error: {str(e)}"

def combine_and_refine_text(gpt_text, gemini_text, fields, progress=gr.Progress()):
    """Combine and refine text from both models, then extract specified fields using Gemini."""
    
    prompt = f"""You are an AI-powered data specialist with deep expertise in medical documentation. Your task is to act as a meticulous assembler of information from two text sources (PASS 1 and PASS 2).

For each field you need to extract, follow this two-step process:

**Step 1: Apply Expert Filtering.** 
Use your expert medical knowledge to identify *only the relevant information* for that field from the source text.
- *Example:* When asked for "Vitals", you must first scan both Passes and gather only text related to actual vital signs (BP, HR, Temp, RR, SpO2, etc.). You must explicitly **ignore** and discard demographic data like 'Age' or 'Sex' from your consideration for the 'Vitals' field, even if the raw text lists them nearby.

**Step 2: Assemble the Filtered Data.** 
Once you have a collection of only the relevant text snippets, apply the 'Synthesis & Conflict Resolution Hierarchy' below to assemble them into the final value for the field.

---
**Synthesis & Conflict Resolution Hierarchy**
For each field you need to extract, follow these steps in order:

**Step A: Check for Identical Information**
If PASS 1 and PASS 2 provide the exact same text for a field, you must use the version from PASS 2.

**Step B: Check for a Superior Version**
If the information is for the same field but one version is clearly more complete or accurate (e.g., "Akola" vs. "Akola, District: Akola"), you must select and use the text from the more comprehensive version.

**Step C: Perform an Intelligent Combination (No Overlap)**
If each Pass provides unique, non-overlapping information for a field, your task is to combine them.
- **CRITICAL:** You must merge them logically to create a single, coherent value without duplicating any words.
- *Example:* If PASS 1 has "Narayan Apartment Flat No. 31" and PASS 2 has "Gurudatta Nagar, Daski Bhuyad", the correct final output is "Narayan Apartment Flat No. 31, Gurudatta Nagar, Daski Bhuyad".
- *Anti-Example:* If PASS 1 is "Dr. Smith" and PASS 2 is "Dr. Smith, MD", the correct combination is "Dr. Smith, MD", **not** "Dr. Smith, Dr. Smith, MD".

**Step D: Handle True Conflicts**
If, and only if, the information is for the same field but is fundamentally contradictory (e.g., two completely different phone numbers), you must present both options clearly, labeled by their source pass.
- *Example:* `Patient Mobile No.: 9975763569 (from PASS 1) / 9772202564 (from PASS 2)`.

**Step E: The "Not Found" Rule**
Only output "Not found" if, after following all the steps above, no information for the field can be found in either Pass.

---
**Output Format & Final Rules:**
- Provide a structured list with clear field names and the extracted values.
- **CRITICAL:** Do NOT add any extra commentary. Your final output must not contain the words 'GPT', 'Gemini', 'PASS 1', or 'PASS 2'. It should be a clean, final report.

---
**FIELDS TO EXTRACT:**
{fields}

---
**DOCUMENT TEXT TO ANALYZE**

=== PASS 1 ===
{gpt_text}

=== PASS 2 ===
{gemini_text}
"""
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error during field extraction: {str(e)}"

def create_file_previews(files):
    """
    Processes uploaded files and converts them to a list of PIL Images for display.
    """
    if not files:
        return []

    display_files = []
    for file in files:
        file_path = file.name
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                img = Image.open(file_path)
                display_files.append(img)
            elif file_extension == '.pdf':
                pdf_images = pdf_to_images(file_path)
                if pdf_images:
                    display_files.extend(pdf_images)
        except Exception as e:
            print(f"[WARN] Skipping file due to preview generation error: {file_path} | {e}")
            
    if not display_files:
        gr.Warning("Could not generate previews for the uploaded files.")

    return display_files

def run_ai_analysis(files, fields, progress=gr.Progress()):
    """
    Main function to run the full AI analysis pipeline.
    """
    progress(0, desc="Starting AI Analysis...")
    if not files:
        raise gr.Error("Please upload at least one file for AI Analysis.")
    if not fields:
        raise gr.Error("Please enter the fields you want to extract.")
        
    file_paths = [file.name for file in files]
    
    progress(0.1, desc="Extracting text...")
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_gpt = executor.submit(call_gpt4, file_paths)
        future_gemini = executor.submit(call_gemini, file_paths)
        
        gpt_text = future_gpt.result()
        gemini_text = future_gemini.result()

    progress(0.7, desc="Refining extracted text...")
    
    final_result = combine_and_refine_text(gpt_text, gemini_text, fields)
    
    progress(1, desc="Analysis Complete!")
    # return final_result
    return final_result, gpt_text, gemini_text

# --- Gradio UI ---

with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="neutral")) as demo:
    gr.Markdown("# 📄 Document Intelligence Hub")
    gr.Markdown("An all-in-one tool for extracting structured data from your documents using advaneced multi-pass AI.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload Files")
            file_upload = gr.Files(
                label="Upload PDFs or Images",
                file_types=["image", ".pdf"],
                height=200,
            )
            
            gr.Markdown("### 2. Configure AI Analysis")
            fields_input = gr.Textbox(
                label="Fields to Extract (one per line)",
                lines=10,
                placeholder="e.g., Patient Name\nDate of Birth\n..."
            )
            
            with gr.Accordion("Example Field Sets", open=False):
                gr.Examples(
                    [["Chief Complaint\nHistory of Present Illness\nPast Medical History\nMedications\nAllergies"]],
                    inputs=fields_input,
                    label="Medical Chart Fields"
                )
                
            ai_submit_btn = gr.Button("🚀 Run AI Analysis", variant="primary", size="lg")

        with gr.Column(scale=2):
            gr.Markdown("### Document Preview")
            gallery_output = gr.Gallery(
                label="File Previews",
                show_label=False,
                elem_id="gallery",
                columns=[2],
                rows=[2],
                object_fit="contain",
                height="auto"
            )

            gr.Markdown("### AI Analysis Result")
            ai_output_textbox = gr.Textbox(
                label="Final Extracted Fields",
                lines=15,
                show_copy_button=True,
                interactive=False
            )
            
            if os.getenv("DEBUG") == "True":
                gr.Markdown("### Debug: Individual Model Outputs")
                with gr.Row():
                    gpt_debug_textbox = gr.Textbox(
                        label="GPT-4o Raw Output",
                        lines=10,
                        show_copy_button=True,
                        interactive=False
                    )
                    gemini_debug_textbox = gr.Textbox(
                        label="Gemini 2.5 Flash Raw Output",
                        lines=10,
                        show_copy_button=True,
                        interactive=False
                    )

    # --- Event Handlers ---
    
    file_upload.upload(
        fn=create_file_previews,
        inputs=[file_upload],
        outputs=[gallery_output]
    )
    
    ai_submit_outputs = [ai_output_textbox]
    if os.getenv("DEBUG") == "True":
        ai_submit_outputs.extend([gpt_debug_textbox, gemini_debug_textbox])
    
    ai_submit_btn.click(
        fn=run_ai_analysis,
        inputs=[file_upload, fields_input],
        outputs=ai_submit_outputs
    )

def check_api_keys():
    """Checks for API keys and prints their status."""
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    print("🔑 API Key Status:")
    if openai_key:
        print(" OpenAI API key loaded successfully!")
    else:
        print("🚨 Warning: OPENAI_API_KEY not found in .env file!")
    
    if gemini_key:
        print(" Gemini API key loaded successfully!")
    else:
        print("🚨 Warning: GEMINI_API_KEY not found in .env file!")
    
    if not openai_key or not gemini_key:
        print("\n👉 Please add your API keys to the .env file to ensure the application works correctly.")
        print("Example .env file:\nOPENAI_API_KEY=your-openai-key-here\nGEMINI_API_KEY=your-gemini-key-here")

if __name__ == "__main__":
    # check_api_keys()
    demo.launch(debug=True, show_error=True) 