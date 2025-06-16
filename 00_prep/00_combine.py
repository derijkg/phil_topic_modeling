import os
import re
import csv
import json
from pathlib import Path
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import logging

# --- 1. UNIFIED CONFIGURATION ---

# NOTE: Use a set for EXCLUDED_BOOKS for much faster lookups.
EXCLUDED_BOOKS = {
    'Leviathan - Thomas Hobbes.txt',
    'Nathan the Wise; a dramatic poem in five acts - Gotthold Ephraim Lessing.txt',
    'The Communist Manifesto - Karl Marx, Friedrich Engels.txt',
    'The Fable of the Bees; Or, Private Vices, Public Benefits - Bernard Mandeville.txt',
    "Hegel's Philosophy of Mind - Georg Wilhelm Friedrich Hegel.txt",
    'An Inquiry Into the Nature and Causes of the Wealth of Nations - Adam Smith, M. Garnier.txt',
    'Primitive culture, vol. 1 (of 2) - Edward B. Tylor.txt',
    'Primitive culture, vol. 2 (of 2) - Edward B. Tylor.txt',
    'Second Treatise of Government - John Locke.txt',
    'The Writings of Thomas Paine — Volume 4 (1794-1796)_ The Age of Reason - Thomas Paine.txt',
    'Three Dialogues Between Hylas and Philonous in Opposition to Sceptics and Atheists - George Berkeley.txt',
    'Thus Spake Zarathustra_ A Book for All and None - Friedrich Wilhelm Nietzsche.txt',
    'On the Origin of Species By Means of Natural Selection _ Or, the Preservation of Favoured Races in the Struggle for Life - Charles Darwin.txt',
    'An Inquiry into the Nature and Causes of the Wealth of Nations - Adam Smith.txt',
    'The Descent of Man, and Selection in Relation to Sex - Charles Darwin.txt',
    'Auguste Comte and Positivism - John Stuart Mill.txt',
    'Dialogues Concerning Natural Religion - David Hume.txt'
}

# --- 2. COMPILED REGEX PATTERNS ---
# Grouping simple replacement patterns makes the cleaning function much tidier.
CLEANING_PATTERNS = [
    (re.compile(r' ?—( Author’s Note .) ?', re.IGNORECASE), ' '), # Author notes
    (re.compile(r' ?\(\d+?\) ?'), ' '),                          # (1265)
    (re.compile(r' ?\[.+?\] ?', re.DOTALL), ' '),                 # [references]
    (re.compile(r'^\d+\.?\s*'), ''),                              # 1. Starting number
    (re.compile(r' ?§ ?\d* ?\.? ?'), ' '),                      # § section symbols
    (re.compile(r'( *\t  *)'), ' '),                              # Tabs
    (re.compile(r'( ?[IVXCM]+\. ?)'), ' '),                       # IV. Roman numerals
    (re.compile(r' ?\(\d+?:?\d+?\) ?'), ' '),                     # (1:15)
    (re.compile(r' ?[↑↓↩↪]+ ?'), ' '),                           # Arrows
    (re.compile(r' ?\* ?'), ' '),                                # Asterisks
    (re.compile(r' ?, ?'), ', '),                                 # Space before comma
    (re.compile(r' ?\n ?'), ' '),                                 # Newlines
    (re.compile(r' ?\. ?'), '. '),                                # Space around period
    (re.compile(r' {2,}'), ' '),                                 # Extra spaces
]

# Patterns requiring special handler functions
CONNEXION_PATTERN = re.compile(r'connexion', re.IGNORECASE)
END_NUM_PATTERN = re.compile(r'(?<=[\.?!]) \d+ ?([A-Z])?')


class EpubProcessor:
    """
    A class to process EPUB files from an input directory, extract content
    and metadata, clean it, and save it as structured JSON.
    """
    def __init__(self, input_folder: str, output_folder: str, excluded_books: set):
        self.input_dir = Path(input_folder)
        self.output_dir = Path(output_folder)
        self.excluded_books = excluded_books
        self.all_metadata = []

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)

    # --- Helper and Cleaning Functions ---

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Removes or replaces characters that are invalid in filenames."""
        name = re.sub(r'[<>:"/\\|?*]', '_', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name[:150]

    @staticmethod
    def _end_num_replacer(match: re.Match) -> str:
        """Handles replacement for END_NUM_PATTERN."""
        letter = match.group(1)
        return f' {letter}' if letter else ''

    @staticmethod
    def _preserve_case_connexion(match: re.Match) -> str:
        """Replaces 'connexion' with 'connection', preserving case."""
        original_word = match.group(0)
        replacement = 'connection'
        result = []
        for i, original_char in enumerate(original_word):
            if i < len(replacement):
                replacement_char = replacement[i]
                if original_char.isupper():
                    result.append(replacement_char.upper())
                else:
                    result.append(replacement_char.lower())
        if len(replacement) > len(original_word):
            remaining = replacement[len(original_word):]
            result.append(remaining.upper() if original_word[-1].isupper() else remaining.lower())
        return "".join(result)

    def _clean_text(self, text: str) -> str:
        """Applies all cleaning regex patterns to a string."""
        for pattern, replacement in CLEANING_PATTERNS:
            text = pattern.sub(replacement, text)

        # Apply special patterns
        text = CONNEXION_PATTERN.sub(self._preserve_case_connexion, text)
        text = END_NUM_PATTERN.sub(self._end_num_replacer, text)
        
        return text.strip()

    # --- Core Processing Logic ---

    def _extract_metadata(self, book: epub.EpubBook, epub_filename: str) -> dict:
        """Extracts metadata from the EPUB book object."""
        title = book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else "Untitled"
        
        authors = [item[0] for item in book.get_metadata('DC', 'creator')]
        author_str = ", ".join(authors) if authors else "Unknown Author"
        
        date_raw = book.get_metadata('DC', 'date')[0][0] if book.get_metadata('DC', 'date') else ""
        pub_year = re.search(r'^(\d{4})', date_raw).group(1) if re.search(r'^(\d{4})', date_raw) else ""

        return {
            "title": title,
            "author": author_str,
            "publication_year": pub_year,
            "source_epub": epub_filename,
        }

    def _extract_content(self, book: epub.EpubBook) -> list:
        """Extracts and segments content into a list of chapters and paragraphs."""
        content_parts = []
        for item in book.spine:
            doc = book.get_item_with_id(item[0])
            if doc.get_type() != ebooklib.ITEM_DOCUMENT:
                continue

            soup = BeautifulSoup(doc.get_content(), 'html.parser')
            body = soup.find('body')
            if not body:
                continue

            for element in body.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']):
                text_for_check = element.get_text(" ", strip=True).upper()
                
                # Filter out common Gutenberg boilerplate
                if any(kw in text_for_check for kw in ["PROJECT GUTENBERG", "*** START OF", "*** END OF"]):
                    continue
                if any(cls in element.get('class', []) for cls in ['pg-boilerplate', 'toc', 'pgheader']):
                    continue
                
                # Process as chapter or paragraph
                if element.name.startswith('h'):
                    chapter_title = element.get_text(separator=' ', strip=True)
                    if chapter_title and len(chapter_title) > 2:
                        # Use a tuple to distinguish chapter markers from paragraphs
                        content_parts.append(('CHAPTER_BREAK', chapter_title))
                elif element.name == 'p':
                    paragraph_text = element.get_text(separator=' ', strip=True)
                    if paragraph_text and len(paragraph_text) > 10:
                        content_parts.append(('PARAGRAPH', paragraph_text))
                        
        return content_parts

    def process_single_epub(self, epub_path: Path):
        """Main workflow for processing a single EPUB file."""
        logging.info(f"Processing: {epub_path.name}")
        
        try:
            book = epub.read_epub(epub_path)
            metadata = self_extract_metadata(book, epub_path.name)

            # --- Elegant Exclusion Check ---
            # Create a representative filename and check against the exclusion set early.
            check_filename = f"{self._sanitize_filename(metadata['title'])} - {self._sanitize_filename(metadata['author'])}.txt"
            if check_filename in self.excluded_books:
                logging.warning(f"Skipping excluded book: {check_filename}")
                return

            # --- Extraction and Cleaning ---
            raw_content_parts = self._extract_content(book)
            if not raw_content_parts:
                logging.error(f"No content found for {epub_path.name}")
                return

            # --- Structure the Content ---
            structured_content = []
            current_chapter = []

            for part_type, text in raw_content_parts:
                if part_type == 'CHAPTER_BREAK':
                    if current_chapter:
                        structured_content.append(current_chapter)
                    # Start a new chapter. The chapter title itself is not added to the paragraph list.
                    current_chapter = [] 
                elif part_type == 'PARAGRAPH':
                    cleaned_paragraph = self._clean_text(text)
                    if cleaned_paragraph:
                        current_chapter.append(cleaned_paragraph)
            
            # Append the last chapter if it exists
            if current_chapter:
                structured_content.append(current_chapter)

            # --- Final Assembly and Output ---
            final_data = {
                "metadata": metadata,
                "content": structured_content
            }
            
            # Add this book's metadata to our master list for the final CSV
            self.all_metadata.append(metadata)

            # Save the final JSON object for this book
            json_output_filename = self._sanitize_filename(metadata['title']) + '.json'
            json_output_path = self.output_dir / json_output_filename
            
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, ensure_ascii=False, indent=4)
            
            logging.info(f"Successfully created: {json_output_path}")

        except Exception as e:
            logging.error(f"Failed to process {epub_path.name}. Error: {e}", exc_info=True)

    def run(self):
        """Processes all EPUB files in the input directory."""
        logging.info(f"Starting processing from: {self.input_dir}")
        
        epub_files = list(self.input_dir.glob('*.epub'))
        if not epub_files:
            logging.warning("No .epub files found in the input directory.")
            return

        for epub_path in epub_files:
            self.process_single_epub(epub_path)
        
        # --- Create a final summary metadata CSV ---
        if self.all_metadata:
            csv_path = self.output_dir / "_master_metadata.csv"
            logging.info(f"Writing summary metadata to: {csv_path}")
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.all_metadata[0].keys())
                writer.writeheader()
                writer.writerows(self.all_metadata)
        
        logging.info("--- Processing Complete ---")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Define your input and output folders here
    INPUT_EPUB_FOLDER = "input"
    OUTPUT_JSON_FOLDER = "processed_output"

    processor = EpubProcessor(
        input_folder=INPUT_EPUB_FOLDER,
        output_folder=OUTPUT_JSON_FOLDER,
        excluded_books=EXCLUDED_BOOKS
    )
    processor.run()