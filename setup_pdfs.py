"""
Helper script to download WHO PDFs for RAG chatbot
You can manually download PDFs and place them in who_pdfs/ directory
"""

import os
from pathlib import Path
import urllib.request
from typing import List

BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "who_pdfs"
PDF_DIR.mkdir(exist_ok=True)

# Example WHO PDF URLs (you can add more)
WHO_PDF_URLS = [
    # Add WHO pneumonia/chest X-ray related PDF URLs here
    # Example:
    # "https://www.who.int/publications/i/item/WHO-2019-nCoV-Clinical-2021-1",
]

def download_pdf(url: str, filename: str) -> bool:
    """Download a PDF from URL"""
    try:
        output_path = PDF_DIR / filename
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {filename}: {e}")
        return False

def main():
    """Main function"""
    print("=" * 50)
    print("WHO PDF Setup for RAG Chatbot")
    print("=" * 50)
    
    print(f"\nPDF directory: {PDF_DIR}")
    
    if not WHO_PDF_URLS:
        print("\nNo PDF URLs configured.")
        print("\nTo add PDFs manually:")
        print(f"1. Download WHO pneumonia/chest X-ray guidelines PDFs")
        print(f"2. Place them in: {PDF_DIR}")
        print(f"3. The RAG chatbot will automatically process them on first run")
        
        # Check if any PDFs already exist
        existing_pdfs = list(PDF_DIR.glob("*.pdf"))
        if existing_pdfs:
            print(f"\n✓ Found {len(existing_pdfs)} existing PDF(s):")
            for pdf in existing_pdfs:
                print(f"  - {pdf.name}")
        else:
            print(f"\n⚠ No PDFs found in {PDF_DIR}")
            print("\nYou can download WHO guidelines from:")
            print("  - https://www.who.int/publications")
            print("  - Search for 'pneumonia', 'chest X-ray', 'respiratory infections'")
    else:
        print(f"\nDownloading {len(WHO_PDF_URLS)} PDF(s)...")
        for i, url in enumerate(WHO_PDF_URLS, 1):
            filename = f"who_guideline_{i}.pdf"
            download_pdf(url, filename)
    
    print("\n" + "=" * 50)
    print("Setup complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()



