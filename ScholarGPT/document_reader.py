''' Dependencies - PyMuPDF,Pandas,Numpy
intsall using 
pip install pymuPDF
pip install pandas
pip install numpy '''


import fitz
import pandas as pd
import re
from unidecode import unidecode


def block_reader(file_path):
  doc = fitz.open(file_path)
  block_dict = {}
  page_num = 1
  for page in doc: 
      file_dict = page.get_text('dict') 
      block = file_dict['blocks'] 
      block_dict[page_num] = block 
      page_num += 1 
  return block_dict




def format_unpacker(block_dict):
  rows = []
  for page_num, blocks in block_dict.items():
    for block in blocks:
        if block['type'] == 0:
            for line in block['lines']:
                for span in line['spans']:
                    xmin, ymin, xmax, ymax = list(span['bbox'])
                    font_size = span['size']
                    text = unidecode(span['text'])
                    span_font = span['font']
                    is_upper = False
                    is_bold = False
                    if "bold" in span_font.lower():
                        is_bold = True
                    if re.sub("[\(\[].*?[\)\]]", "", text).isupper():
                        is_upper = True
                    if text.replace(" ","") !=  "":
                        rows.append((xmin, ymin, xmax, ymax, text, is_upper, is_bold, span_font, font_size))
  spanning = pd.DataFrame(rows, columns=['xmin','ymin','xmax','ymax', 'text', 'is_upper','is_bold','span_font', 'font_size'])
  return spanning
