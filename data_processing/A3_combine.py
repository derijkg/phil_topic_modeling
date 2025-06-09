import re
import os
import json
excluded_texts = ['Leviathan - Thomas Hobbes.txt',
                  'Nathan the Wise; a dramatic poem in five acts - Gotthold Ephraim Lessing.txt',
                  'The Communist Manifesto - Karl Marx, Friedrich Engels.txt',
                  'The Fable of the Bees; Or, Private Vices, Public Benefits - Bernard Mandeville.txt',
                  "Hegel's Philosophy of Mind - Georg Wilhelm Friedrich Hegel.txt", #NOT ORIGINAL
                  'An Inquiry Into the Nature and Causes of the Wealth of Nations - Adam Smith, M. Garnier.txt',
                  'Primitive culture, vol. 1 (of 2) - Edward B. Tylor.txt',
                  'Primitive culture, vol. 2 (of 2) - Edward B. Tylor.txt',
                  'Second Treatise of Government - John Locke.txt', #old langUAGE
                  'The Writings of Thomas Paine — Volume 4 (1794-1796)_ The Age of Reason - Thomas Paine.txt',
                  'Theologico-Political Treatise — Part 2 - Benedictus de Spinoza.txt',
                  'Three Dialogues Between Hylas and Philonous in Opposition to Sceptics and Atheists - George Berkeley.txt', #DIALOGUE
                  'Thus Spake Zarathustra_ A Book for All and None - Friedrich Wilhelm Nietzsche.txt', # dialogue / parable
                  'On the Origin of Species By Means of Natural Selection _ Or, the Preservation of Favoured Races in the Struggle for Life - Charles Darwin.txt', #very different topics
                  'An Inquiry into the Nature and Causes of the Wealth of Nations - Adam Smith.txt',
                  'The Descent of Man, and Selection in Relation to Sex - Charles Darwin.txt',
                  'Auguste Comte and Positivism - John Stuart Mill.txt'

                  ]

newline_pattern = re.compile(r' ?\n ?')
paragraph_break_pattern = re.compile(r'PARAGRAPH_BREAK')
asterisk_pattern = re.compile(r' ?\* ?')
double_space_pattern = re.compile(r' +')



def process_all(path):
    if os.path.isfile(path):
        return process_file(path)
    else:
        result = []
        for file in os.listdir(path):
            if file in excluded_texts:
                continue
            else:
                result.append(process_file(os.path.join(path,file)))
        return result


def process_file(path):
    with open(path,'r',encoding='utf-8') as f:
        content = f.read()
    head , tail = os.path.split(path)
    result={
        'title': tail,
        'paragraphs': [],
        'idx': []
    }

    for idx, paragraph in enumerate(re.split(paragraph_break_pattern,content)):
        if paragraph.strip() == '':
            continue
        result['paragraphs'].append(process_para(paragraph))
        result['idx'].append(idx)
    return result


def process_para(content):
    content = content.strip('\n')
    content = re.sub(newline_pattern,' ',content)
    content = re.sub(asterisk_pattern,' ',content)
    content = re.sub(double_space_pattern,' ',content)
    return content





path = r'cleaned_texts_start_end'
#path = r'cleaned_texts\The Critique of Practical Reason - Immanuel Kant.txt'
result = process_all(path)

with open('data.json','w',encoding='utf-8') as f:
    json.dump(result,f)



