# extracts p from epubs, and cleans, adding PARAGRAPH_BREAK
import os
import re
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
                  'Theologico-Political Treatise — Part 2 - Benedictus de Spinoza.txt',
                  'Three Dialogues Between Hylas and Philonous in Opposition to Sceptics and Atheists - George Berkeley.txt', #DIALOGUE
                  'Thus Spake Zarathustra_ A Book for All and None - Friedrich Wilhelm Nietzsche.txt', # dialogue / parable
                  'On the Origin of Species By Means of Natural Selection _ Or, the Preservation of Favoured Races in the Struggle for Life - Charles Darwin.txt', #very different topics
                  'An Inquiry into the Nature and Causes of the Wealth of Nations - Adam Smith.txt',
                  'The Descent of Man, and Selection in Relation to Sex - Charles Darwin.txt',
                  'Auguste Comte and Positivism - John Stuart Mill.txt'

                  ]

#re patterns
reference_pattern = re.compile(r' ?\[.+?\] ?', re.DOTALL)
extra_space_pattern = re.compile(r' {2,}')
starting_number_pattern = re.compile(r'\n( ?\d+ ?\.? ?)')
section_symbol_pattern = re.compile(r' ?§ ?\d* ?\.? ?')
paragraph_break_pattern = re.compile(r'PARAGRAPH_BREAK')
content_pattern = re.compile(r'CONTENT_START(.*?)CONTENT_END', re.DOTALL)
tab_pattern = re.compile(r'\n( *\t  *)')
num_pattern = re.compile(r'\(\d+?\) ?')
big_word_split_pattern = re.compile(r'\b[A-Z]{1}( )[A-Z]')
numeral_start_pattern = re.compile(r'\n( ?[IVXCM]+\. ?)')
num_parenth_pattern = re.compile(r'\(\d+?:?\d+?\) ?')
arrow_pattern = re.compile(r' ?[↑↓↩↪]+ ?')

class Cleaner:
    def __init__(self, folder_path_in, folder_path_out):
        self.folder_path_in = folder_path_in
        self.folder_path_out = folder_path_out
        if not os.path.exists(self.folder_path_out):
            os.makedirs(self.folder_path_out)


    # cleaner functions
    def extract_content(self, text):
        match = re.search(content_pattern, text)
        if match:
            return match.group(1).strip()
        return ''

    def reduce_paragraphs(self,content):
        content = re.split(paragraph_break_pattern, content)
        selected_paragraphs = []
        for paragraph in content:
            if paragraph.count('\n') +1 >= 5:
                selected_paragraphs.append(paragraph)
        results = 'PARAGRAPH_BREAK'.join(selected_paragraphs)
        return results


    def clean(self, content):
        content = re.sub(big_word_split_pattern,'',content) # artefacts of html like T REATISE
        content = re.sub(extra_space_pattern, ' ', content) # extra spaces
        content = re.sub(num_pattern,'',content) # (1265)\s
        content = re.sub(reference_pattern, ' ', content) # references
        content = re.sub(starting_number_pattern, '', content) # starting with number.
        content = re.sub(section_symbol_pattern, '', content) # section symbols
        content = re.sub(tab_pattern, '', content) # remove tabs
        content = re.sub(numeral_start_pattern,'',content)
        content = re.sub(num_parenth_pattern,'',content)
        content = re.sub(arrow_pattern, ' ', content) # arrows

        return content

    # general logic
    def process(self, path):
        if os.path.isfile(path):
            self._process_file(path)
        else:
            for file in os.listdir(path):
                file_path = os.path.join(path,file)
                self._process_file(file_path)


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
        content = self.reduce_paragraphs(content)
        content = self.clean(content)

        save_path = os.path.join(self.folder_path_out,tail)
        with open(save_path,'w',encoding='utf-8') as f:
            f.write(content)
        print(f'Processed {tail}')


folder_path_in = r'extracted_texts_start_end'
folder_path_out = r'cleaned_texts'

cleaner = Cleaner(folder_path_in=folder_path_in,folder_path_out=folder_path_out)
#cleaner.process(r'data_processing\extracted_texts\A System of Logic, Ratiocinative and Inductive - John Stuart Mill.txt')
cleaner.process(folder_path_in)