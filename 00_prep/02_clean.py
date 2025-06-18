# extracts p from epubs, and cleans, adding PARAGRAPH_BREAK
import os
import re
import json
import pandas as pd
from collections import defaultdict
import numpy as np
import pprint

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

# paragraph remover
#authors_note_pattern = re.compile(r"^\d+?\*") actually belongs to james mill
editor_pattern = re.compile(r'— Ed\.', re.IGNORECASE)

# clean
newline_pattern = re.compile(r' ?\n ?')
authors_ast_pattern = re.compile(r' ?\d+?\* ?') # 2*, 45*
num_pattern = re.compile(r' ?\(\d+?\) ?')
reference_pattern = re.compile(r' ?\[.+?\] ?', re.DOTALL)
section_symbol_pattern = re.compile(r'^§+ ?\d* ?\.? ?') # start of paragraph section + num
starting_number_pattern = re.compile(r'^\d+\.?\s*')
tab_pattern = re.compile(r'( *\t  *)')
num_parenth_pattern = re.compile(r' ?\(\d+?:?\d+?\) ?')
#numeral_start_pattern = re.compile(r'^\(?[IVXCM]+\)?\. ?') # ???
numeral_start_pattern = re.compile(
    r'^\s*M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\.\s*',
    re.IGNORECASE
)
arrow_pattern = re.compile(r' ?[↑↓↩↪]+ ?')
connexion_pattern = re.compile(r'connexion', re.IGNORECASE)
asterisk_pattern = re.compile(r' ?\* ?')
# spaced_ellipsis_pattern = re.compile(r'(?: \.){2,}')
end_num_pattern = re.compile(r'\s*\d+\s*$')
'''
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
'''
### big_word_split_pattern = re.compile(r'(?<=.)[A-Z]{1}( )[A-Z]')

# finishers
space_comma_pattern = re.compile(r' ,')
space_period_pattern = re.compile(r' \.') # combine punct + space
extra_space_pattern = re.compile(r' {2,}')

class Cleaner:
    def __init__(self, folder_path_in, folder_path_out, metadata='gutenberg_metadata.csv'):
        self.folder_path_in = folder_path_in
        self.folder_path_out = folder_path_out
        if not os.path.exists(self.folder_path_out):
            os.makedirs(self.folder_path_out)
        self.df = pd.read_csv(metadata)
        self.removal_stats = defaultdict(int)

    def _apply_regex_and_track(self, content, pattern_name, pattern, replacement):
        """Applies a regex substitution and tracks the number of matches removed."""
        matches = pattern.findall(content)
        self.removal_stats[pattern_name] += len(matches)

        content = pattern.sub(replacement, content)
        return content

    def clean(self, content):
        patterns_to_apply = [
            ('newlines', newline_pattern, ' '),
            ('author_ast', authors_ast_pattern, ' '),
            ('parenthesized_numbers', num_pattern, ' '),
            ('bracketed_references', reference_pattern, ' '),
            ('paragraph_start_number', starting_number_pattern, ''),
            ('section_symbols', section_symbol_pattern, ' '),
            ('tabs', tab_pattern, ' '),
            ('parenthesized_colond_numbers', num_parenth_pattern, ' '),
            ('roman_numeral_start', numeral_start_pattern, ' '),
            ('arrows', arrow_pattern, ' '),
            ('connexion_to_connection', connexion_pattern, 'connection'),
            ('end_of_sentence_number', end_num_pattern, ' '),
            ('asterisks', asterisk_pattern, ' '),
            ('space_before_comma', space_comma_pattern, ','),
            ('space_before_period', space_period_pattern, '.'),
        ]

        for name, pattern, repl in patterns_to_apply:
            content = self._apply_regex_and_track(content, name, pattern, repl)
            # maintenence
            content = re.sub(extra_space_pattern, ' ', content)
            content = content.strip()

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
        else:
            raise ValueError(f'content indicators not found')

    def get_metadata(self, file_path):
        head, tail = os.path.split(file_path)
        title = tail.split(' - ')[0].strip()
        row = self.df[self.df['Original Title'].apply(self.sanitize_filename) == title]
        if not row.empty:
            row_series = row.iloc[0]
            cleaned_series = row_series.replace({np.nan: None})
            metadata = cleaned_series.to_dict()
            return metadata
        else:
            raise ValueError(f'No metadata found for {tail}')
        

    def split_chapters(self, content):
        chapters = re.split(chapter_break_pattern, content)
        chapter_list = []
        for idx, chapter in enumerate(chapters):
            chapter = chapter.strip()
            if chapter == '':
                continue
            if chapter.count('\n') +1 >=10:
                chapter_list.append(chapter.strip())
        return chapter_list
   
    def split_paragraphs(self,content):
        content = re.split(paragraph_break_pattern, content)
        paragraph_list = []
        for paragraph in content:
            paragraph = paragraph.strip()
            if paragraph == '':
                continue
            #if re.match(authors_note_pattern, paragraph): # author notes are indicated by this pattern in 'analysis of the phenomena of the human mind by james mill
            #    print('SKIPPED AUT NOTE')
            #    continue
            if re.search(editor_pattern, paragraph):
                print('SKIPPED ED NOTE')
                continue
            if paragraph.count('\n') +1 >= 5:
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
            raise ValueError(f'CONTENT NOT FOUND FOR {tail}')
        
        metadata = self.get_metadata(file_path)
        result  = {
            "meta": metadata
        }
        chapters_end = []
        per_chapter = self.split_chapters(content)
        for c_idx, chapter in enumerate(per_chapter):
            chapter = chapter.strip()
            if chapter == '':
                print('SKIPPED EMPTY CHAPTER')
                continue
            per_paragraph = self.split_paragraphs(chapter)
            paragraphs_end = []
            for p_idx, paragraph in enumerate(per_paragraph):
                if paragraph == '':
                    print('SKIPPED EMPTY PARAGRAPH')
                    continue
                paragraph = self.clean(paragraph)
                paragraph = paragraph.strip()
                if paragraph == '':
                    print('SKIPPED EMPTY PARAGRAPH')
                    continue
                else:
                    paragraphs_end.append(paragraph)
            chapters_end.append(paragraphs_end)
        for chapter in chapters_end:                  # remove empty chapters after cleaning
            if chapter == []:
                chapters_end.remove(chapter)
        result['content'] = chapters_end
        print(f'Processed {tail}')
        return result
    
    # general logic
    def process(self, path):
        if os.path.isfile(path):
            result_all = self._process_file(path)
            out_file_name = os.path.join(self.folder_path_out, os.path.basename(path))
            out_file_name = out_file_name.replace('.txt', '.json')
            self.report_stats()
            if result_all is not None:
                return result_all, out_file_name
            else:
                raise ValueError('File is excluded')

        else:
            result_all = []
            for file in os.listdir(path):
                file_path = os.path.join(path,file)
                result = self._process_file(file_path)
                if result is not None:
                    result_all.append(result)
            out_file_name = os.path.join(self.folder_path_out, 'all_processed.json')
            self.report_stats()
            return result_all, out_file_name

    def report_stats(self):
        """Prints a formatted report of num of matches by each regex pattern."""
        print("\n" + "="*50)
        print("      Regex Pattern Match Statistics")
        print("="*50)

        if not self.removal_stats:
            print("No statistics recorded.")
            return

        # Sort stats by characters removed, descending
        sorted_stats = sorted(self.removal_stats.items(), key=lambda item: item[1], reverse=True)

        print(f"{'Pattern Name':<35} | {'Matches'}")
        print("-" * 50)
        total_removed = 0
        for name, count in sorted_stats:
            print(f"{name:<35} | {count:,}")
            total_removed += count
        print("-" * 50)
        print(f"{'Total matches':<35} | {total_removed:,}")
        print("="*50 + "\n")

if __name__ == '__main__':
    folder_path_in = r'extracted_texts_start_end'
    folder_path_out = r'cleaned_texts'
    
    # add start, end
    start = 'CONTENT_START'
    end = 'CONTENT_END'
    content_map = {'A System of Logic, Ratiocinative and Inductive - John Stuart Mill.txt': (862,
                                                                            2242508),
    'A Theological-Political Treatise [Part III] - Benedictus de Spinoza.txt': (2691,
                                                                                94108),
    'A Theological-Political Treatise [Part IV] - Benedictus de Spinoza.txt': (4911,
                                                                                156965),
    'A Treatise Concerning the Principles of Human Knowledge - George Berkeley.txt': (2345,
                                                                                    216466),
    'A Treatise of Human Nature - David Hume.txt': (386, 1438779),
    'Aids to Reflection; and, The Confessions of an Inquiring Spirit - Samuel Taylor Coleridge.txt': (8456,
                                                                                                    919513),
    'An Enquiry Concerning Human Understanding - David Hume.txt': (642, 338910),
    'An Enquiry Concerning the Principles of Morals - David Hume.txt': (2576,
                                                                        267445),
    'An Essay Concerning Humane Understanding, Volume 1 _ MDCXC, Based on the 2nd Edition, Books 1 and 2 - John Locke.txt': (1166,
                                                                                                                            853983),
    'Analysis of the Phenomena of the Human Mind - James Mill.txt': (4755,
                                                                    1564367),
    'Beyond Good and Evil - Friedrich Wilhelm Nietzsche.txt': (1484, 413197),
    'Chance, Love, and Logic_ Philosophical Essays - Charles S. Peirce.txt': (6326,
                                                                            519509),
    'Common Sense - Thomas Paine.txt': (3654, 127328),
    'Dialogues Concerning Natural Religion - David Hume.txt': (428, 217862),
    'Essays - David Hume.txt': (13107, 214637),
    'Essays — First Series - Ralph Waldo Emerson.txt': (783, 418204),
    'Essays — Second Series - Ralph Waldo Emerson.txt': (449, 351573),
    'Ethics - Benedictus de Spinoza.txt': (640, 517291),
    'First Principles - Herbert Spencer.txt': (11653, 1036002),
    'Fundamental Principles of the Metaphysic of Morals - Immanuel Kant.txt': (614,
                                                                                181892),
    "Kant's Critique of Judgement - Immanuel Kant.txt": (54869, 747982),
    "Kant's Prolegomena to Any Future Metaphysics - Immanuel Kant.txt": (5380,
                                                                        300837),
    'Laocoon - Gotthold Ephraim Lessing.txt': (2801, 349076),
    'Letters on England - Voltaire.txt': (5454, 222027),
    'Nature - Ralph Waldo Emerson.txt': (260, 88563),
    'On Liberty - John Stuart Mill.txt': (25751, 308855),
    'The Analogy of Religion to the Constitution and Course of Nature _ To which are added two brief dissertations_ I. On personal identity. II. On the nat - Joseph Butler.txt': (80134,
                                                                                                                                                                                    631626),
    'The Birth of Tragedy; or, Hellenism and Pessimism - Friedrich Wilhelm Nietzsche.txt': (60316,
                                                                                            330000),
    'The Critique of Practical Reason - Immanuel Kant.txt': (604, 382690),
    'The Critique of Pure Reason - Immanuel Kant.txt': (551, 1279814),
    'The Essence of Christianity _ Translated from the second German edition - Ludwig Feuerbach.txt': (1236,
                                                                                                        882494),
    'The Genealogy of Morals _ The Complete Works, Volume Thirteen, edited by Dr. Oscar Levy. - Friedrich Wilhelm Nietzsche.txt': (2202,
                                                                                                                                    334295),
    'The Golden Bough_ A Study of Magic and Religion - James George Frazer.txt': (14167,
                                                                                2303539),
    'The Principles of Psychology, Volume 1 (of 2) - William James.txt': (906,
                                                                        1664624),
    'The Social Contract & Discourses - Jean-Jacques Rousseau.txt': (97653,
                                                                    725819),
    'The Subjection of Women - John Stuart Mill.txt': (700, 256118),
    'The Theory of Moral Sentiments _ Or, an Essay Towards an Analysis of the Principles by Which Men Naturally Judge Concerning the Conduct and Character, - Adam Smith.txt': (1131,
                                                                                                                                                                                688296),
    'The Will to Believe, and Other Essays in Popular Philosophy - William James.txt': (1346,
                                                                                        588633),
    'The Works of the Right Honourable Edmund Burke, Vol. 01 (of 12) - Edmund Burke.txt': (131214,
                                                                                            428437),
    'The Writings of Thomas Paine — Volume 2 (1779-1792)_ The Rights of Man - Thomas Paine.txt': (295572,
                                                                                                537005),
    'Theodicy _ Essays on the Goodness of God, the Freedom of Man and the Origin of Evil - Freiherr von Gottfried Wilhelm Leibniz.txt': (108026,
                                                                                                                                        1110283),
    'Theologico-Political Treatise — Part 1 - Benedictus de Spinoza.txt': (3408,
                                                                            177748),
    'Theologico-Political Treatise — Part 2 - Benedictus de Spinoza.txt': (2696,
                                                                            173236),
    'Utilitarianism - John Stuart Mill.txt': (647, 162359)} 
    if content_map == {}:
        raise ValueError('No map')
    elif len(content_map) == len(os.listdir(folder_path_in)):
        for file, (start_pos, end_pos) in content_map.items():
            file_path = os.path.join(folder_path_in,file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if start and end not in content:
                new_content = content[:start_pos] + start + content[start_pos:]
                new_content = new_content[:end_pos] + end + new_content[end_pos:]
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
            else: continue
    else: raise ValueError('not same len')

    # clean
    cleaner = Cleaner(folder_path_in=folder_path_in,folder_path_out=folder_path_out)
    results, out_file_name = cleaner.process(folder_path_in)
    
    # adding years
    title_year = {
        'A System of Logic, Ratiocinative and Inductive': 1843,
        'A Theological-Political Treatise [Part III]': 1670,
        'A Theological-Political Treatise [Part IV]': 1670,
        'A Treatise Concerning the Principles of Human Knowledge': 1710,
        'A Treatise of Human Nature': 1739,
        'Aids to Reflection; and, The Confessions of an Inquiring Spirit': 1835,
        'An Enquiry Concerning Human Understanding': 1748,
        'An Enquiry Concerning the Principles of Morals': 1751,
        'An Essay Concerning Humane Understanding, Volume 1 / MDCXC, Based on the 2nd Edition, Books 1 and 2': 1690,
        'Analysis of the Phenomena of the Human Mind': 1829,
        'Beyond Good and Evil': 1886,
        'Chance, Love, and Logic: Philosophical Essays': 1923,
        'Common Sense': 1776,
        'Dialogues Concerning Natural Religion': 1779,
        'Essays': '1741',
        'Essays — First Series': 1841,
        'Essays — Second Series': 1844,
        'Ethics': 1677,
        'First Principles': 1862,
        'Fundamental Principles of the Metaphysic of Morals': 1785,
        "Kant's Critique of Judgement": 1790,
        "Kant's Prolegomena to Any Future Metaphysics": 1783,
        'Laocoon': 1766,
        'Letters on England': 1733,
        'Nature': 1836,
        'On Liberty': 1859,
        'The Analogy of Religion to the Constitution and Course of Nature / To which are added two brief dissertations: I. On personal identity. II. On the nature of virtue.': 1736,
        'The Birth of Tragedy; or, Hellenism and Pessimism': '1872',
        'The Critique of Practical Reason': '1788',
        'The Critique of Pure Reason': '1781',
        'The Essence of Christianity / Translated from the second German edition': '1841',
        'The Genealogy of Morals / The Complete Works, Volume Thirteen, edited by Dr. Oscar Levy.': '1887',
        'The Golden Bough: A Study of Magic and Religion': '1890',
        'The Principles of Psychology, Volume 1 (of 2)': '1890',
        'The Social Contract & Discourses': '1762',
        'The Subjection of Women': '1869',
        'The Theory of Moral Sentiments / Or, an Essay Towards an Analysis of the Principles by Which Men Naturally Judge Concerning the Conduct and Character, First of Their Neighbours, and Afterwards of Themselves. to Which Is Added, a Dissertation on the Origin of Languages.': '1759',
        'The Will to Believe, and Other Essays in Popular Philosophy': '1896',
        'The Works of the Right Honourable Edmund Burke, Vol. 01 (of 12)': '1812',
        'The Writings of Thomas Paine — Volume 2 (1779-1792): The Rights of Man': '1784',
        'Theodicy / Essays on the Goodness of God, the Freedom of Man and the Origin of Evil': '1710',
        'Theologico-Political Treatise — Part 1': '1670',
        'Theologico-Political Treatise — Part 2': '1670',
        'Utilitarianism': '1863'
        }
    title_year = {k: int(v) for k, v in title_year.items()}

    for entry in results:
        meta = entry.get('meta')
        if meta:
            title = meta.get('Original Title')
            year = title_year.get(title)
            meta['Publication Year (Original)'] = year
            entry['meta'] = meta
        else:
            raise ValueError('META NOT FOUND')

    # write
    with open(out_file_name, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
