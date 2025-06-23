#!/usr/bin/env python3
"""
PDF Splitter Tool

A command-line tool to split PDF files into individual pages.
Usage: python pypdf.py <input_pdf> [options]
"""

import os
import sys
import argparse
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter


def split_pdf(input_path, output_folder=None, prefix="page"):
    """
    Split a PDF file into individual pages.
    
    Args:
        input_path (str): Path to the input PDF file
        output_folder (str): Output folder for split pages (default: 'split_pages')
        prefix (str): Prefix for output files (default: 'page')
    """
    # Validate input file
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        return False
    
    if not input_path.lower().endswith('.pdf'):
        print(f"Error: '{input_path}' is not a PDF file.")
        return False
    
    # Set default output folder if not specified
    if output_folder is None:
        input_name = Path(input_path).stem
        output_folder = f"{input_name}_split_pages"
    
    # Create output folder
    try:
        os.makedirs(output_folder, exist_ok=True)
    except Exception as e:
        print(f"Error creating output folder: {e}")
        return False
    
    try:
        # Read the input PDF
        print(f"Reading PDF: {input_path}")
        reader = PdfReader(input_path)
        total_pages = len(reader.pages)
        
        if total_pages == 0:
            print("Error: PDF file appears to be empty or corrupted.")
            return False
        
        print(f"Found {total_pages} page(s)")
        
        # Split and save each page
        for i, page in enumerate(reader.pages, start=1):
            writer = PdfWriter()
            writer.add_page(page)
            output_path = os.path.join(output_folder, f"{prefix}_{i:03d}.pdf")
            
            with open(output_path, "wb") as f:
                writer.write(f)
            
            print(f"Saved: {output_path}")
        
        print(f"\nSuccessfully split {total_pages} page(s) into '{output_folder}' folder.")
        return True
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return False


def main():
    """Main function to handle command-line arguments and execute the split operation."""
    parser = argparse.ArgumentParser(
        description="Split a PDF file into individual pages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pypdf.py document.pdf
  python pypdf.py document.pdf -o my_pages
  python pypdf.py document.pdf -o my_pages -p slide
        """
    )
    
    parser.add_argument(
        "pdf_file",
        help="Path to the input PDF file"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output folder for split pages (default: <filename>_split_pages)"
    )
    
    parser.add_argument(
        "-p", "--prefix",
        default="page",
        help="Prefix for output files (default: page)"
    )
    
    args = parser.parse_args()
    
    # Execute the split operation
    success = split_pdf(args.pdf_file, args.output, args.prefix)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 