''' Dependencies - PyMuPDF,Pandas,Numpy
intsall using 
pip install pymuPDF
pip install pandas
pip install numpy '''


import fitz


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
