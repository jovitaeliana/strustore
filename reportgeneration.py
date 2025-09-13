"""
GDINO Detection and Classification Comparison Report Generator

This script generates PDF reports comparing original auction images with GDINO detection results
and enhanced classifications. It processes items from CSV and creates comprehensive comparison
reports showing detection results, tokens, enhanced classifications, and similarity scores.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv, json
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader

# Configuration
WRITE_PER_ID_PDFS = True
COMBINED_PDF_NAME = "gdino_comparison_report.pdf"

# Data paths - Update these to match your directory structure
TEXT_PROMPT_CSV = "./zm_scraper/items-prompt.csv"  # CSV with item IDs
GDINO_FINAL_ORIGINAL = "./gdinoOutput/final-original"  # Original GDINO JSON results (gdino_readable, tokens)
GDINO_FINAL = "./gdinoOutput/final"  # Enhanced GDINO JSON results (enhanced_classification only)
GDINO_OUTPUT_IMAGES = "./gdinoOutput/output"  # GDINO detection images with bounding boxes
COMPILED = "./gdino_reports"  # Output directory for PDFs

# Page layout - A4 Portrait
PAGE_W, PAGE_H = A4
MARGIN = 15 * mm
GAP_COL = 10 * mm
HEADER_GAP = 6 * mm
TEXT_IMG_GAP = 4 * mm
SECTION_GAP = 8 * mm

# Fonts
FONT_REG = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
SIZE_TITLE = 14
SIZE_SUB = 11
SIZE_SMALL = 7
SIZE_TEXT = 8
SIZE_JSON = 6

VALID_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# --- HELPER FUNCTIONS ---
def ensure_dir(p: Path):
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)

def read_ids_from_csv(csv_path: Path) -> List[str]:
    """Read item IDs from CSV file."""
    if not csv_path.exists():
        print(f"Warning: CSV file not found: {csv_path}")
        return []
    
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    
    headers = [h.strip().lower() for h in rows[0]]
    body = rows[1:] if headers else rows
    candidates = ["id", "item_id", "listing_id", "sku"]
    
    if headers:
        id_col = next((headers.index(c) for c in candidates if c in headers), 0)
    else:
        id_col = 0
    
    ids = [r[id_col].strip() for r in body if r and len(r) > id_col and r[id_col].strip()]
    
    # Remove duplicates while preserving order
    seen, out = set(), []
    for _id in ids:
        if _id not in seen:
            seen.add(_id)
            out.append(_id)
    return out

def collect_images(folder: Path) -> Dict[str, Path]:
    """Collect all valid image files from a folder."""
    out = {}
    if folder.exists():
        for p in folder.iterdir():
            if p.suffix.lower() in VALID_IMG_EXT:
                out[p.stem] = p
    return out

def collect_json(folder: Path) -> Dict[str, Path]:
    """Collect all JSON files from a folder."""
    out = {}
    if folder.exists():
        for p in folder.iterdir():
            if p.suffix.lower() == ".json":
                out[p.stem] = p
    return out

def load_json(p: Optional[Path]) -> dict:
    """Load JSON file safely."""
    if not p or not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error loading JSON {p}: {e}")
        return {}

def fit_image_box(img_path: Path, max_w: float, max_h: float) -> Tuple[float, float]:
    """Calculate image dimensions to fit within box while preserving aspect ratio."""
    try:
        with Image.open(img_path) as im:
            w, h = im.size
        scale = min(max_w / w, max_h / h) if w > 0 and h > 0 else 1.0
        return (w * scale, h * scale)
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return (max_w, max_h)

def wrap_text(c: canvas.Canvas, text: str, max_width: float, font_name: str, font_size: int) -> List[str]:
    """Wrap text to fit within specified width."""
    c.setFont(font_name, font_size)
    lines = []
    for para in (text or "").split("\n"):
        words = para.split()
        if not words:
            lines.append("")
            continue
        cur = words[0]
        for w in words[1:]:
            test = f"{cur} {w}"
            if c.stringWidth(test, font_name, font_size) <= max_width:
                cur = test
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines

def draw_block(c: canvas.Canvas, text: str, x: float, y_top: float, max_width: float,
               font_name: str = FONT_REG, font_size: int = SIZE_TEXT, leading: float = None, 
               min_y: float = None) -> float:
    """Draw a text block and return the y position after the block. Stops at min_y if specified."""
    if leading is None:
        leading = font_size * 1.0  # Reduced line spacing from 1.2 to 1.0
    if min_y is None:
        min_y = MARGIN
    
    lines = wrap_text(c, text, max_width, font_name, font_size)
    c.setFont(font_name, font_size)
    y = y_top
    
    for ln in lines:
        if y - leading < min_y:  # Stop if we would go below min_y
            break
        c.drawString(x, y, ln)
        y -= leading
    return y

def draw_block_with_height_limit(c: canvas.Canvas, text: str, x: float, y_top: float, 
                                max_width: float, max_height: float,
                                font_name: str = FONT_REG, font_size: int = SIZE_TEXT, 
                                leading: float = None) -> Tuple[float, str]:
    """Draw text block within height limit and return (final_y, remaining_text)."""
    if leading is None:
        leading = font_size * 1.0  # Reduced line spacing from 1.2 to 1.0
    
    lines = wrap_text(c, text, max_width, font_name, font_size)
    c.setFont(font_name, font_size)
    y = y_top
    lines_drawn = 0
    
    for i, ln in enumerate(lines):
        if y - leading < y_top - max_height:  # Would exceed height limit
            # Return remaining text
            remaining_lines = lines[i:]
            remaining_text = '\n'.join(remaining_lines)
            return y, remaining_text
        
        c.drawString(x, y, ln)
        y -= leading
        lines_drawn += 1
    
    return y, ""  # All text was drawn

def limit_gdino_tokens(gdino_json: dict, max_tokens: int = 15) -> dict:
    """Limit gdino_tokens to top N results."""
    if not gdino_json or 'gdino_tokens' not in gdino_json:
        return gdino_json
    
    limited_json = gdino_json.copy()
    if isinstance(gdino_json['gdino_tokens'], dict):
        # Limit tokens for each item key
        limited_tokens = {}
        for key, tokens in gdino_json['gdino_tokens'].items():
            if isinstance(tokens, list):
                limited_tokens[key] = tokens[:max_tokens]
            else:
                limited_tokens[key] = tokens
        limited_json['gdino_tokens'] = limited_tokens
    
    return limited_json

def merge_gdino_data(original_json: dict, enhanced_json: dict) -> dict:
    """Merge data from original and enhanced JSON files, keeping only specific fields."""
    merged_data = {}
    
    # From original JSON: gdino_readable and gdino_tokens
    if 'gdino_readable' in original_json:
        merged_data['gdino_readable'] = original_json['gdino_readable']
    
    if 'gdino_tokens' in original_json:
        merged_data['gdino_tokens'] = original_json['gdino_tokens']
    
    # From enhanced JSON: enhanced_classification only
    if 'enhanced_classification' in enhanced_json:
        merged_data['enhanced_classification'] = enhanced_json['enhanced_classification']
    
    return merged_data

def format_gdino_data(merged_json: dict) -> str:
    """Format the merged JSON data for display with limited gdino_tokens."""
    import json
    
    if not merged_json:
        return "No GDINO data available"
    
    # Limit gdino_tokens to top 15 results
    limited_json = limit_gdino_tokens(merged_json, 15)
    
    # Format the JSON with proper indentation
    try:
        formatted_json = json.dumps(limited_json, indent=2, ensure_ascii=False)
        return formatted_json
    except Exception as e:
        return f"Error formatting JSON: {e}\n\nRaw data: {str(limited_json)}"

def calculate_optimal_columns(json_text: str) -> int:
    """Calculate optimal number of columns based on content amount."""
    lines = json_text.split('\n')
    total_lines = len(lines)
    
    # Estimate average line length
    avg_line_length = sum(len(line) for line in lines) / max(1, total_lines)
    
    # Determine columns based on content amount
    if total_lines <= 30 and avg_line_length <= 50:
        return 1  # Single column for small data
    elif total_lines <= 80 or avg_line_length <= 80:
        return 2  # Two columns for medium data
    else:
        return 3  # Three columns for large data

def split_text_into_columns(text: str, num_columns: int, max_height: float, 
                           font_size: int = 6, leading: float = None) -> Tuple[List[str], str]:
    """Split text into columns that fit within height constraints. Returns (columns, remaining_text)."""
    if leading is None:
        leading = font_size * 1.0  # Reduced line spacing
    
    lines = text.split('\n')
    max_lines_per_col = int(max_height / leading)
    total_lines_that_fit = max_lines_per_col * num_columns
    
    # Check if all text fits
    if len(lines) <= total_lines_that_fit:
        # All text fits, distribute into columns
        columns = []
        lines_per_col = len(lines) // num_columns + (1 if len(lines) % num_columns > 0 else 0)
        
        for i in range(num_columns):
            start_idx = i * lines_per_col
            end_idx = min(start_idx + lines_per_col, len(lines))
            if start_idx < len(lines):
                columns.append('\n'.join(lines[start_idx:end_idx]))
            else:
                columns.append('')
        
        return columns, ''
    
    # Text doesn't fit, fill available space and return remainder
    columns = []
    lines_used = 0
    
    for col in range(num_columns):
        col_lines = lines[lines_used:lines_used + max_lines_per_col]
        if col_lines:
            columns.append('\n'.join(col_lines))
            lines_used += len(col_lines)
        else:
            columns.append('')
    
    # Return remaining text for next page
    remaining_lines = lines[lines_used:]
    remaining_text = '\n'.join(remaining_lines) if remaining_lines else ''
    
    return columns, remaining_text

# --- PAGE RENDERING ---
def render_continuation_page_header(c: canvas.Canvas, item_id: str, stem: str, page_num: int):
    """Render header for continuation pages."""
    c.setFont(FONT_BOLD, SIZE_SUB)
    x = MARGIN
    y = PAGE_H - MARGIN
    c.drawString(x, y, f"Item {item_id} — {stem} (continued) — Page {page_num}")
    y -= (SIZE_SUB * 1.5)
    return y

def render_listing_page(c: canvas.Canvas,
                        item_id: str, stem: str,
                        gdino_img: Optional[Path],
                        merged_json: dict) -> int:
    """Render pages showing GDINO detection image and complete JSON data. Returns number of pages used."""
    page_count = 1
    
    # First page with image and initial JSON content
    # Page title
    c.setFont(FONT_BOLD, SIZE_TITLE)
    x = MARGIN
    y = PAGE_H - MARGIN
    c.drawString(x, y, f"Item {item_id} — Detection Analysis: {stem}")
    y -= (SIZE_TITLE * 1.4)

    # Subtitle
    c.setFont(FONT_REG, SIZE_SUB)
    c.drawString(x, y, "GDINO Detection Image & Complete JSON Data")
    y -= (SIZE_SUB * 1.6)

    # Layout: Image at top (centered), JSON data below
    page_width = PAGE_W - 2*MARGIN
    
    # Reserve space for image (reduced to about 20% of page height for more JSON space)
    img_max_h = (PAGE_H - MARGIN * 3) * 0.20
    img_y_top = y
    
    # Center the image horizontally with smaller max width
    if gdino_img and gdino_img.exists():
        try:
            w, h = fit_image_box(gdino_img, page_width * 0.4, img_max_h)  # Use 40% of page width max
            img_x = MARGIN + (page_width - w) / 2  # Center the image
            
            c.drawImage(ImageReader(str(gdino_img)), img_x, img_y_top - h,
                        width=w, height=h, preserveAspectRatio=True, anchor='sw')
            
            # Add source paths directly under image
            c.setFont(FONT_REG, SIZE_SMALL)
            source_path = f"Data from: {Path(GDINO_FINAL_ORIGINAL).absolute()}/{item_id}/{stem}.json + {Path(GDINO_FINAL).absolute()}/{item_id}/{stem}.json"
            c.drawString(MARGIN, img_y_top - h - SIZE_SMALL - 3, source_path)
            
            # Start JSON content below source
            json_y_start = img_y_top - h - SIZE_SMALL - 15
            
        except Exception as e:
            c.setFont(FONT_REG, SIZE_SMALL)
            c.drawString(MARGIN, img_y_top - SIZE_SMALL, f"Error loading image: {e}")
            # Still show source paths
            source_path = f"Data from: {Path(GDINO_FINAL_ORIGINAL).absolute()}/{item_id}/{stem}.json + {Path(GDINO_FINAL).absolute()}/{item_id}/{stem}.json"
            c.drawString(MARGIN, img_y_top - SIZE_SMALL * 2 - 3, source_path)
            json_y_start = img_y_top - SIZE_SMALL * 2 - 15
    else:
        c.setFont(FONT_REG, SIZE_SMALL)
        c.drawString(MARGIN, img_y_top - SIZE_SMALL, "No GDINO detection image found")
        # Still show source paths
        source_path = f"Data from: {Path(GDINO_FINAL_ORIGINAL).absolute()}/{item_id}/{stem}.json + {Path(GDINO_FINAL).absolute()}/{item_id}/{stem}.json"
        c.drawString(MARGIN, img_y_top - SIZE_SMALL * 2 - 3, source_path)
        json_y_start = img_y_top - SIZE_SMALL * 2 - 15

    # JSON Data section - Dynamic column layout with pagination
    c.setFont(FONT_BOLD, SIZE_SUB)
    c.drawString(MARGIN, json_y_start, "Merged JSON Data (gdino_readable + enhanced_classification + Top 15 tokens):")
    json_y_start -= (SIZE_SUB * 1.3)
    
    # Format JSON and determine optimal column layout
    gdino_text = format_gdino_data(merged_json)
    num_columns = calculate_optimal_columns(gdino_text)
    
    # Calculate available height for JSON content on first page
    available_height = json_y_start - MARGIN - 10  # Minimal bottom margin
    
    # Ensure we have enough space for content
    if available_height < 100:  # If very little space, force 3 columns
        num_columns = 3
    
    # Calculate column dimensions
    gap_width = GAP_COL * (num_columns - 1) if num_columns > 1 else 0
    col_width = (page_width - gap_width) / num_columns
    col_positions = [MARGIN + i * (col_width + GAP_COL) for i in range(num_columns)]
    
    # Process content with pagination
    remaining_text = gdino_text
    
    while remaining_text.strip():
        if page_count > 1:
            # Start new page for continuation
            c.showPage()
            json_y_start = render_continuation_page_header(c, item_id, stem, page_count)
            # Full page height available for continuation pages
            available_height = json_y_start - MARGIN - 10
        
        # Split text for current page
        columns, remaining_text = split_text_into_columns(
            remaining_text, num_columns, available_height, SIZE_JSON
        )
        
        # Draw columns with strict height constraints
        leading = SIZE_JSON * 1.0  # Tight line spacing
        for column_text, col_x in zip(columns, col_positions):
            if column_text.strip():  # Only draw non-empty columns
                draw_block(c, column_text, col_x, json_y_start, col_width, 
                         FONT_REG, SIZE_JSON, leading, MARGIN)
        
        # If no remaining text, we're done
        if not remaining_text.strip():
            break
        
        page_count += 1

    c.showPage()
    return page_count

# --- MAIN EXECUTION ---
def main():
    """Main execution function."""
    compiled_root = Path(COMPILED)
    ensure_dir(compiled_root)

    # Read item IDs from CSV
    ids = read_ids_from_csv(Path(TEXT_PROMPT_CSV))
    print(f"Found {len(ids)} IDs from CSV: {TEXT_PROMPT_CSV}")

    if not ids:
        print("No IDs found. Please check your CSV file path and format.")
        return

    all_pages_data = []
    processed_count = 0

    for item_id in ids:
        # Collect original GDINO JSON files (for gdino_readable and gdino_tokens)
        original_gdino_jsons = collect_json(Path(GDINO_FINAL_ORIGINAL) / item_id)
        
        # Collect enhanced GDINO JSON files (for enhanced_classification)
        enhanced_gdino_jsons = collect_json(Path(GDINO_FINAL) / item_id)
        
        if not original_gdino_jsons and not enhanced_gdino_jsons:
            print(f"[skip] {item_id}: no GDINO JSON files found in either {GDINO_FINAL_ORIGINAL}/{item_id} or {GDINO_FINAL}/{item_id}")
            continue

        # Collect GDINO output images (with detection boxes)
        gdino_output_imgs = collect_images(Path(GDINO_OUTPUT_IMAGES) / item_id)

        pages_data = []
        # Process files that exist in either directory
        all_stems = set(original_gdino_jsons.keys()) | set(enhanced_gdino_jsons.keys())
        
        for stem in all_stems:
            # Load original data (gdino_readable, gdino_tokens)
            original_json = load_json(original_gdino_jsons.get(stem)) if stem in original_gdino_jsons else {}
            
            # Load enhanced data (enhanced_classification)
            enhanced_json = load_json(enhanced_gdino_jsons.get(stem)) if stem in enhanced_gdino_jsons else {}
            
            # Merge the data keeping only the required fields
            merged_json = merge_gdino_data(original_json, enhanced_json)
            
            gdino_img = gdino_output_imgs.get(stem)
            pages_data.append((stem, merged_json, gdino_img))

        if not pages_data:
            print(f"[skip] {item_id}: no valid data found")
            continue

        # Generate per-ID PDF
        if WRITE_PER_ID_PDFS:
            pdf_path = compiled_root / f"{item_id}_gdino_report.pdf"
            c = canvas.Canvas(str(pdf_path), pagesize=A4)
            total_item_pages = 0
            for stem, merged_json, gdino_img in pages_data:
                pages_used = render_listing_page(c, item_id, stem, gdino_img, merged_json)
                total_item_pages += pages_used
            c.save()
            print(f"✓ Generated {pdf_path} ({total_item_pages} pages)")

        all_pages_data.append((item_id, pages_data))
        processed_count += 1

    # Generate combined PDF
    if all_pages_data:
        combined_path = compiled_root / COMBINED_PDF_NAME
        c = canvas.Canvas(str(combined_path), pagesize=A4)
        total_pages = 0
        
        for item_id, pages_data in all_pages_data:
            for stem, merged_json, gdino_img in pages_data:
                pages_used = render_listing_page(c, item_id, stem, gdino_img, merged_json)
                total_pages += pages_used
        
        c.save()
        print(f"✓ Generated combined report: {combined_path} ({total_pages} total pages)")
    else:
        print("No data to process. Please check your file paths and data.")

    print(f"\nProcessing complete: {processed_count} items processed")
    print(f"Reports saved to: {compiled_root.absolute()}")

if __name__ == "__main__":
    main()
