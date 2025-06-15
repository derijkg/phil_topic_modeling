import os
import re
import csv
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, NavigableString, Tag # Import Tag

# --- Configuration ---
INPUT_EPUB_FOLDER = r"input\epubs"  # Folder containing your EPUB files
OUTPUT_TEXT_FOLDER = "test\extracted_texts"
OUTPUT_METADATA_CSV = "test\gutenberg_metadata.csv"

# --- Helper Functions ---
def sanitize_filename(name):
    """Removes or replaces characters that are invalid in filenames."""
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'\s+', ' ', name).strip() # Consolidate whitespace
    name = name[:150] # Limit filename length to avoid issues
    return name

def extract_year(date_string):
    """Extracts the year from a typical EPUB date string (e.g., YYYY-MM-DD or just YYYY)."""
    if not date_string:
        return ""
    match = re.search(r'^(\d{4})', date_string)
    return match.group(1) if match else ""

def get_first_metadata_value(book, namespace, tag, default="Unknown"):
    """Safely extracts the first metadata value or a default."""
    meta_list = book.get_metadata(namespace, tag)
    if meta_list and meta_list[0] and meta_list[0][0]:
        return meta_list[0][0]
    return default

# --- Main Script ---
def process_gutenberg_epubs(epub_folder, text_output_folder, metadata_csv_file):
    if not os.path.exists(epub_folder):
        print(f"Error: Input folder '{epub_folder}' not found.")
        return

    os.makedirs(text_output_folder, exist_ok=True)

    all_metadata_entries = []
    processed_files_count = 0
    failed_files_count = 0

    print(f"Starting processing of EPUBs in '{epub_folder}'...")

    for filename in os.listdir(epub_folder):
        if filename.lower().endswith(".epub"):
            epub_path = os.path.join(epub_folder, filename)
            print(f"\nProcessing: {filename}")

            try:
                book = epub.read_epub(epub_path)

                # 1. Extract Metadata (same as before)
                title = get_first_metadata_value(book, 'DC', 'title', "Untitled")
                primary_authors = []
                for name, attrs in book.get_metadata('DC', 'creator'):
                    role = attrs.get('opf:role', attrs.get('role', '')).lower()
                    if role == 'aut':
                        primary_authors.append(name)
                if not primary_authors:
                    primary_authors = [item[0] for item in book.get_metadata('DC', 'creator') if item and item[0]]
                original_writer_str = ", ".join(primary_authors) if primary_authors else "Unknown Author"
                translators = []
                other_contributors_info = []
                for name, attrs in book.get_metadata('DC', 'contributor'):
                    role = attrs.get('opf:role', attrs.get('role', '')).lower()
                    if role == 'trl':
                        translators.append(name)
                    elif name:
                        other_contributors_info.append(f"{name}" + (f" ({role})" if role else " (contributor)"))
                for name, attrs in book.get_metadata('DC', 'creator'):
                    if name not in primary_authors:
                        role = attrs.get('opf:role', attrs.get('role', '')).lower()
                        if role and role != 'aut':
                            if not any(name in t for t in translators) and \
                               not any(name in o for o in other_contributors_info):
                                other_contributors_info.append(f"{name} ({role})")
                translator_contributor_display_parts = []
                if translators:
                    translator_contributor_display_parts.append("Translator: " + ", ".join(translators))
                if other_contributors_info:
                    translator_contributor_display_parts.append("Other: " + ", ".join(other_contributors_info))
                translator_contributor_final_str = "; ".join(translator_contributor_display_parts)
                version_pub_date_raw = get_first_metadata_value(book, 'DC', 'date', "")
                version_pub_year = extract_year(version_pub_date_raw)
                current_book_metadata = {
                    "Original Title": title,
                    "Original Writer": original_writer_str,
                    "Publication Year (Original)": "",
                    "Translator/Contributor": translator_contributor_final_str,
                    "Publication Year (Version)": version_pub_year,
                    "Source EPUB": filename
                }
                all_metadata_entries.append(current_book_metadata)

                # 2. Prepare Output Text Filename (same as before)
                safe_title = sanitize_filename(title)
                safe_author = sanitize_filename(original_writer_str)
                output_text_filename = f"{safe_title} - {safe_author}.txt"
                output_text_path = os.path.join(text_output_folder, output_text_filename)

                # 3. Extract and Segment Text with Chapter Detection
                full_text_content_parts = []
                
                # Get all documents from the book's "spine" (the reading order)
                for item in book.spine:
                    doc_item = book.get_item_with_id(item[0])
                    if doc_item.get_type() != ebooklib.ITEM_DOCUMENT:
                        continue

                    soup = BeautifulSoup(doc_item.get_content(), 'html.parser')
                    body_content = soup.find('body')
                    if not body_content:
                        continue
                    
                    # Find all header (h1-h6) and paragraph (p) tags in document order
                    for element in body_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']):
                        # --- Boilerplate Filtering ---
                        # Skip elements that are likely Gutenberg headers/footers/TOCs
                        element_classes = element.get('class', [])
                        text_for_check = element.get_text(" ", strip=True).upper()
                        if 'pg-boilerplate' in element_classes or 'toc' in element_classes or 'pgheader' in element_classes:
                            continue
                        if ("PROJECT GUTENBERG" in text_for_check and "EBOOK" in text_for_check) or \
                           "*** START OF" in text_for_check or "*** END OF" in text_for_check:
                            continue
                        
                        # --- Chapter and Paragraph Processing ---
                        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                            chapter_title = element.get_text(separator=' ', strip=True)
                            # Add chapter marker and title if it looks like a real title
                            if chapter_title and len(chapter_title) > 2:
                                full_text_content_parts.append(f"\n\nCHAPTER_BREAK\n{chapter_title}\n")
                        
                        elif element.name == 'p':
                            paragraph_text = element.get_text(separator=' ', strip=True)
                            # Add paragraph if it has meaningful content
                            if paragraph_text and len(paragraph_text) > 10:
                                full_text_content_parts.append(paragraph_text)
                
                # 4. Write Text File 
                with open(output_text_path, 'w', encoding='utf-8') as f:
                    # Join all parts with a paragraph break marker in between
                    f.write("\nPARAGRAPH_BREAK\n".join(full_text_content_parts))
                # --- MODIFIED SECTION END ---

                print(f"  Extracted text to: {output_text_path}")
                processed_files_count += 1

            except Exception as e:
                print(f"  Error processing {filename}: {e}")
                failed_files_count +=1

    # ... (rest of the script is unchanged)
    if all_metadata_entries:
        csv_fieldnames = list(all_metadata_entries[0].keys())
        with open(metadata_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows(all_metadata_entries)
        print(f"\nMetadata written to: {metadata_csv_file}")
    print(f"\n--- Summary ---\nSuccessfully processed: {processed_files_count}\nFailed: {failed_files_count}")
# --- Run the script ---


if __name__ == "__main__":
    process_gutenberg_epubs(INPUT_EPUB_FOLDER, OUTPUT_TEXT_FOLDER, OUTPUT_METADATA_CSV)
    EXCLUDED_TEXTS = ['Leviathan - Thomas Hobbes.txt',
                  'Nathan the Wise; a dramatic poem in five acts - Gotthold Ephraim Lessing.txt',
                  'The Communist Manifesto - Karl Marx, Friedrich Engels.txt',
                  'The Fable of the Bees; Or, Private Vices, Public Benefits - Bernard Mandeville.txt',
                  "Hegel's Philosophy of Mind - Georg Wilhelm Friedrich Hegel.txt", #NOT ORIGINAL
                  'An Inquiry Into the Nature and Causes of the Wealth of Nations - Adam Smith, M. Garnier.txt',
                  'Primitive culture, vol. 1 (of 2) - Edward B. Tylor.txt',
                  'Primitive culture, vol. 2 (of 2) - Edward B. Tylor.txt',
                  'Second Treatise of Government - John Locke.txt', #old langUAGE
                  'The Writings of Thomas Paine — Volume 4 (1794-1796)_ The Age of Reason - Thomas Paine.txt',
                  'Three Dialogues Between Hylas and Philonous in Opposition to Sceptics and Atheists - George Berkeley.txt', #DIALOGUE
                  'Thus Spake Zarathustra_ A Book for All and None - Friedrich Wilhelm Nietzsche.txt', # dialogue / parable
                  'On the Origin of Species By Means of Natural Selection _ Or, the Preservation of Favoured Races in the Struggle for Life - Charles Darwin.txt', #very different topics
                  'An Inquiry into the Nature and Causes of the Wealth of Nations - Adam Smith.txt',
                  'The Descent of Man, and Selection in Relation to Sex - Charles Darwin.txt',
                  'Auguste Comte and Positivism - John Stuart Mill.txt'

                  ]

    for f in os.listdir(OUTPUT_TEXT_FOLDER):
        if f in EXCLUDED_TEXTS:
            os.remove(os.path.join(OUTPUT_TEXT_FOLDER,f))
            print(f"Removed excluded file: {f}")


# Find all h2 (for chapters) and p (for paragraphs) tags in order
# We will iterate through them to maintain the document flow
# This approach is simpler for the current requirement.
'''
chapter_or_paragraph_elements = []
# Common chapter containers - adjust if needed
chapter_divs = body_content.find_all('div', class_=['chapter', 'tei-div'], recursive=False) # Look at top-level divs first

if chapter_divs: # If specific chapter divs are found, process within them
    for ch_div in chapter_divs:
        # Skip boilerplate if the div itself is a known boilerplate container
        div_classes = ch_div.get('class', [])
        div_id = ch_div.get('id', '')
        if 'pg-boilerplate' in div_classes or \
            'pgheader' in div_classes or \
            'pgfooter' in div_classes or \
            'toc' in div_classes or \
            div_id in ['pg-header', 'pg-footer', 'pgepub-license', 'pg-machine-header'] or \
            'pg-start-separator' in div_id or 'pg-end-separator' in div_id:
            continue
        
        # Find h2 and p tags within this chapter div
        for element in ch_div.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'], recursive=True): # Recursive True here
            chapter_or_paragraph_elements.append(element)
else: # Fallback: process all h2 and p tags directly under body
        for element in body_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'], recursive=True):
        chapter_or_paragraph_elements.append(element)


for element in chapter_or_paragraph_elements:
    # --- Skip Gutenberg boilerplate and unwanted sections based on element's own properties ---
    element_classes = element.get('class', [])
    element_id = element.get('id', '')
    
    # More robust boilerplate check on the element itself
    if 'pg-boilerplate' in element_classes or \
        'pgheader' in element_classes or \
        'pgfooter' in element_classes or \
        'toc' in element_classes or \
        element_id in ['pg-header', 'pg-footer', 'pgepub-license', 'pg-machine-header'] or \
        'pg-start-separator' in element_id or 'pg-end-separator' in element_id:
        continue

    text_for_check = element.get_text(" ", strip=True).upper()
    if ("PROJECT GUTENBERG" in text_for_check and \
        ("EBOOK" in text_for_check or "LICENSE" in text_for_check or "COPYRIGHT" in text_for_check)) or \
        "*** START OF" in text_for_check or "*** END OF" in text_for_check :
        if len(text_for_check) < 300: # Assume small boilerplate sections
            continue
    
    if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
        chapter_title_text = element.get_text(separator=' ', strip=True)
        # Filter out boilerplate titles or very short/non-content titles
        # (e.g., just "Chapter X" might be less useful than a descriptive title)
        if chapter_title_text and len(chapter_title_text) > 3 and \
            not "CONTENTS" in chapter_title_text.upper() and \
            not "INDEX" in chapter_title_text.upper() and \
            not "NOTES" in chapter_title_text.upper() and \
            not "APPENDIX" in chapter_title_text.upper() and \
            not "INTRODUCTION" in chapter_title_text.upper() and \
            not "PREFACE" in chapter_title_text.upper() and \
            not "TRANSCR" in chapter_title_text.upper(): # Avoid transcriber's notes as chapters
            full_text_content_parts.append("CHAPTER_BREAK")
            full_text_content_parts.append(chapter_title_text)
    
    elif element.name == 'p':
        # --- MODIFICATION: Extract inner HTML of <p> tags ---
        # Instead of .get_text(), we get the string representation of children
        paragraph_inner_html = "".join(str(child) for child in element.contents).strip()
        
        # Basic filter for meaningful paragraphs
        # Check length of text content for this filter, not raw HTML
        paragraph_text_for_check = element.get_text(separator=' ', strip=True)
        if paragraph_text_for_check and len(paragraph_text_for_check) > 10 :
            # Filter out image placeholders (often within <p>)
            if not paragraph_text_for_check.startswith("Illustration:") and \
                not paragraph_text_for_check.startswith("[Illustration") and \
                not element.find('img'): # also check for img tags inside
                full_text_content_parts.append("PARAGRAPH_BREAK")
                full_text_content_parts.append(paragraph_inner_html)
# --- END OF MODIFIED SECTION ---
'''
