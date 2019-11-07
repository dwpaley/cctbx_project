from __future__ import absolute_import, division, print_function

# LIBTBX_SET_DISPATCHER_NAME mmtbx.rama_z
# LIBTBX_SET_DISPATCHER_NAME phenix.rama_z

from mmtbx.programs import rama_z
from iotbx.cli_parser import run_program

result = run_program(rama_z.Program)
