import sys
import nbformat
from nbclient import NotebookClient

nb_path = 'examples/case_study.ipynb'
out_path = 'examples/case_study_executed.ipynb'

nb = nbformat.read(nb_path, as_version=4)
# Prepend a cell to ensure project root is on sys.path
init_code = """import sys
sys.path.insert(0, '.')
print('sys.path adjusted:', sys.path[0])
"""
nb['cells'].insert(0, nbformat.v4.new_code_cell(init_code))

client = NotebookClient(nb, timeout=600, kernel_name='python3')
client.execute()
nbformat.write(nb, out_path)
print('Executed notebook saved to', out_path)
