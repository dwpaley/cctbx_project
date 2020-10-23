from __future__ import division, print_function
import sys
import libtbx.load_env
from cctbx_website.regression.exercise import exercise

def run():
  exercise(script = "doc_model_map_manager.py",
           tmp_path = 'tmp_files_6')

if __name__ == '__main__':
  sys.exit(run())
