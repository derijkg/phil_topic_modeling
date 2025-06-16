# extracts p from epubs, and cleans, adding PARAGRAPH_BREAK
import os
import re
import json
import pandas as pd

excluded_texts = ['Leviathan - Thomas Hobbes.txt', # old lang
                  'Nathan the Wise; a dramatic poem in five acts - Gotthold Ephraim Lessing.txt', # poem
                  'The Communist Manifesto - Karl Marx, Friedrich Engels.txt', # short para
                  'The Fable of the Bees; Or, Private Vices, Public Benefits - Bernard Mandeville.txt', # story
                  "Hegel's Philosophy of Mind - Georg Wilhelm Friedrich Hegel.txt", #NOT ORIGINAL
                  'An Inquiry Into the Nature and Causes of the Wealth of Nations - Adam Smith, M. Garnier.txt', # economic
                  'Primitive culture, vol. 1 (of 2) - Edward B. Tylor.txt', # anthropology
                  'Primitive culture, vol. 2 (of 2) - Edward B. Tylor.txt',
                  'Second Treatise of Government - John Locke.txt', #old langUAGE
                  'The Writings of Thomas Paine — Volume 4 (1794-1796)_ The Age of Reason - Thomas Paine.txt', # pol econ
                  'Three Dialogues Between Hylas and Philonous in Opposition to Sceptics and Atheists - George Berkeley.txt', #DIALOGUE
                  'Thus Spake Zarathustra_ A Book for All and None - Friedrich Wilhelm Nietzsche.txt', # dialogue / parable
                  'On the Origin of Species By Means of Natural Selection _ Or, the Preservation of Favoured Races in the Struggle for Life - Charles Darwin.txt', #very different topics
                  'An Inquiry into the Nature and Causes of the Wealth of Nations - Adam Smith.txt',
                  'The Descent of Man, and Selection in Relation to Sex - Charles Darwin.txt',
                  'Auguste Comte and Positivism - John Stuart Mill.txt',
                  'Dialogues Concerning Natural Religion - David Hume.txt'

                  ]

# re patterns
# breaks
content_pattern = re.compile(r'CONTENT_START(.*?)CONTENT_END', re.DOTALL)
paragraph_break_pattern = re.compile(r'PARAGRAPH_BREAK')
chapter_break_pattern = re.compile(r'CHAPTER_BREAK')

# clean
authors_note_pattern = re.compile(r' ?—( Author’s Note .) ?', re.IGNORECASE)
num_pattern = re.compile(r' ?\(\d+?\) ?')
reference_pattern = re.compile(r' ?\[.+?\] ?', re.DOTALL)
section_symbol_pattern = re.compile(r' ?§ ?\d* ?\.? ?')
starting_number_pattern = re.compile(r'^\d+\.?\s*')
tab_pattern = re.compile(r'( *\t  *)')
numeral_start_pattern = re.compile(r'( ?[IVXCM]+\. ?)') # ???
num_parenth_pattern = re.compile(r' ?\(\d+?:?\d+?\) ?')
arrow_pattern = re.compile(r' ?[↑↓↩↪]+ ?')
connexion_pattern = re.compile(r'connexion', re.IGNORECASE)
asterisk_pattern = re.compile(r' ?\* ?')
end_num_pattern = re.compile(r'(?<=[\.?!]) \d+ ?([A-Z])?')
def end_num_replacer(match):
  """
  This function is used by re.sub. It checks if an uppercase letter
  was captured in group 1. If so, it returns a space plus that letter.
  If not, it returns an empty string.
  """
  letter = match.group(1)
  if letter:
    # A letter was found, e.g., 'T'. Return ' T'.
    return f' {letter}'
  else:
    # No letter was found (match was at the end of the string). Return nothing.
    return ''
### big_word_split_pattern = re.compile(r'(?<=.)[A-Z]{1}( )[A-Z]')

# finishers
space_comma_pattern = re.compile(r' ?, ?')
space_period_pattern = re.compile(r'( ?\. ?)') # combine punct + space
extra_space_pattern = re.compile(r' {2,}')
newline_pattern = re.compile(r' ?\n ?')





class Cleaner:
    def __init__(self, folder_path_in, folder_path_out, metadata='gutenberg_metadata.csv'):
        self.folder_path_in = folder_path_in
        self.folder_path_out = folder_path_out
        if not os.path.exists(self.folder_path_out):
            os.makedirs(self.folder_path_out)
        self.df = pd.read_csv(metadata)

    def clean(self, content):
        content = re.sub(authors_note_pattern, ' ', content) # remove author notes
        content = re.sub(num_pattern,' ',content) # (1265)\s
        content = re.sub(reference_pattern, ' ', content) # references
        content = re.sub(starting_number_pattern, '', content) # starting with number.
        content = re.sub(section_symbol_pattern, ' ', content) # section symbols
        content = re.sub(tab_pattern, ' ', content) # remove tabs
        content = re.sub(numeral_start_pattern,' ',content)
        content = re.sub(num_parenth_pattern,' ',content)
        content = re.sub(arrow_pattern, ' ', content) # arrows
        content = re.sub(connexion_pattern, 'connection', content)
        content = re.sub(space_comma_pattern, ', ', content) # space before comma
        content = end_num_pattern.sub(end_num_replacer, content)
        content = re.sub(asterisk_pattern, ' ', content)
        
        # finishing touch
        content = re.sub(newline_pattern, ' ', content)
        content = re.sub(space_period_pattern, '. ', content)
        content = re.sub(extra_space_pattern, ' ', content) # extra spaces
        #content = re.sub(big_word_split_pattern,'',content) # artefacts of html like T REATISE
        return content
    
    def sanitize_filename(self, name):
        """Removes or replaces characters that are invalid in filenames."""
        name = re.sub(r'[<>:"/\\|?*]', '_', name)
        name = re.sub(r'\s+', ' ', name).strip() # Consolidate whitespace
        name = name[:150] # Limit filename length to avoid issues
        return name

    def extract_content(self, text):
        match = re.search(content_pattern, text)
        if match:
            return match.group(1).strip()
        return ''

    def get_metadata(self, file_path):
        head, tail = os.path.split(file_path)
        title = tail.split(' - ')[0].strip()
        row = self.df[self.df['Original Title'].apply(self.sanitize_filename) == title]
        if not row.empty:
            metadata = row.iloc[0].to_dict()
            return metadata
        else:
            raise ValueError(f'No metadata found for {tail}')
        

    def split_chapters(self, content):
        chapters = re.split(chapter_break_pattern, content)
        chapter_list = []
        for idx, chapter in enumerate(chapters):
            if chapter.strip() == '':
                continue
            if chapter.count('\n') +1 >=8:
                chapter_list.append(chapter.strip())
        return chapter_list
   
    def split_paragraphs(self,content):
        content = re.split(paragraph_break_pattern, content)
        paragraph_list = []
        for paragraph in content:

            if paragraph.strip() == '':
                continue
            if paragraph.count('\n') +1 >= 5:
                paragraph = paragraph.strip()
                paragraph_list.append(paragraph)
        return paragraph_list

    # ind file
    def _process_file(self, file_path):
        head, tail = os.path.split(file_path)
        if tail in excluded_texts:
            return
        with open(file_path,'r', encoding='utf-8') as file:
            content = file.read()
        content = self.extract_content(content)
        if content == '':
            print(f'CONTENT NOT FOUND FOR {tail}')
            return
        
        metadata = self.get_metadata(file_path)
        result  = {
            "meta": metadata
        }
        per_chapter = self.split_chapters(content)
        chapters_end = []
        for c_idx, chapter in enumerate(per_chapter):
            per_paragraph = self.split_paragraphs(chapter)
            paragraphs_end = []
            for p_idx, paragraph in enumerate(per_paragraph):
                paragraph = self.clean(paragraph)
                paragraph = paragraph.strip()
                if paragraph == '':
                    continue
                else:
                    paragraphs_end.append(paragraph)
            chapters_end.append(paragraphs_end)
        result['content'] = chapters_end

        save_path = os.path.join(self.folder_path_out,tail)
        #with open(save_path,'w',encoding='utf-8') as f:
        #    f.write(content)
        print(f'Processed {tail}')
        return result
    
    # general logic
    def process(self, path):
        if os.path.isfile(path):
            result_all = self._process_file(path)
            out_file_name = os.path.join(self.folder_path_out, os.path.basename(path))
            out_file_name = out_file_name.replace('.txt', '.json')
            with open(out_file_name, 'w', encoding='utf-8') as f:
                json.dump(result_all, f, ensure_ascii=False, indent=4)
        else:
            result_all = []
            for file in os.listdir(path):
                file_path = os.path.join(path,file)
                result = self._process_file(file_path)
                result_all.append(result)
            with open(os.path.join(self.folder_path_out, 'all_processed.json'), 'w', encoding='utf-8') as f:
                json.dump(result_all, f, ensure_ascii=False, indent=4)
        return result_all

if __name__ == '__main__':
    folder_path_in = r'extracted_texts_start_end'
    folder_path_out = r'cleaned_texts'

    cleaner = Cleaner(folder_path_in=folder_path_in,folder_path_out=folder_path_out)
    #cleaner.process(r'data_processing\extracted_texts\A System of Logic, Ratiocinative and Inductive - John Stuart Mill.txt')
    cleaner.process(folder_path_in)