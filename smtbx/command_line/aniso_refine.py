from __future__ import absolute_import, division, print_function
# LIBTBX_SET_DISPATCHER_NAME smtbx.aniso_refine

import os, sys, argparse
from scitbx import lstbx
import scitbx.lstbx.normal_eqns_solving
from smtbx import refinement
from timeit import default_timer as current_time


allowed_input_file_extensions = ('.ins', '.res', '.cif')

class command_line_error(RuntimeError): pass

class number_of_arguments_error(command_line_error): pass

class energy_missing_error(RuntimeError): pass

def make_ins_fstring():
  return '''TITL MONOCOPLUS_30KEV_A OLEX2: imported from CIF
CELL 0.41328 11.7916 15.5197 16.596 88.652 81.195 88.489
ZERR 2 0.0005 0.0007 0.0008 0.001 0.001 0.001
LATT 1
SFAC C H Co F O P Se
UNIT 62 150 12 1 2 10 16

TEMP 23
WGHT 0.1
FVAR 1

{}
{}
{}
{}
{}
{}


C  1     0.47460  0.19890  0.53260  11.00000  .01 
O  5     0.44490  0.18280  0.47220  11.00000  .01 
P1  6     0.89992  0.31742  0.56069  11.00000  .01 
P2  6     0.74588  0.33278  0.92425  11.00000  .01 
P3  6     0.46853  0.52079  0.70287  11.00000  .01 
P4  6     0.28192  0.21304  0.87753  11.00000  .01 
P5  6     0.70667  0.00709  0.76057  11.00000  .01 
Se1  7     0.72379  0.41415  0.72947  11.00000  .01 
Se2  7     0.81676  0.21645  0.75325  11.00000  .01 
Se3  7     0.48372  0.37075  0.84973  11.00000  .01 
Se4  7     0.46182  0.11467  0.71655  11.00000  .01 
Se5  7     0.70515  0.15585  0.59986  11.00000  .01 
Se6  7     0.57019  0.17442  0.87613  11.00000  .01 
Se7  7     0.37241  0.31241  0.68768  11.00000  .01 
Se8  7     0.61039  0.35720  0.57449  11.00000  .01 
hklf 4

END
'''
def make_ins_fstring_monoP_se():
  return '''TITL MONOP_SE OLEX2: imported from CIF
CELL 0.41328 11.7954 23.4662 23.0432 90 95.308 90
ZERR 4 0.0006 0.0011 0.0011 0 0.001 0
LATT 1
SYMM -X,0.5+Y,0.5-Z
SFAC C H Co P Se
UNIT 168 408 24 24 32

TEMP -173
WGHT 0.1
FVAR 1

{}
{}
{}
{}
{}
{}
{}
{}

Co1   3     0.33394  0.58458  0.69843  11.00000  .01 
Co2   3     0.40616  0.60309  0.82382  11.00000  .01 
Co3   3     0.19897  0.53775  0.78893  11.00000  .01 
Co4   3     0.11039  0.63457  0.71464  11.00000  .01 
Co5   3     0.31916  0.69970  0.74937  11.00000  .01 
Co6   3     0.18140  0.65293  0.83896  11.00000  .01 
P1    4     0.41270  0.55378  0.62443  11.00000  .01 
P2    4     0.55423  0.58821  0.88316  11.00000  .01 
P3    4     0.13548  0.45475  0.80673  11.00000  .01 
P4    4     -.03514  0.65012  0.65423  11.00000  .01 
P5    4     0.39186  0.78014  0.72920  11.00000  .01 
P6    4     0.09542  0.68617  0.90950  11.00000  .01
HKLF 4

END
'''

def make_parser():
  parser = argparse.ArgumentParser(
      description='''Refinement of inelastic scattering factors. Contact: Daniel
Paley (dwp2111@columbia.edu)''')
  parser.add_argument(
      'ref_structure',
      type=str,
      help='''A structure in .cif or .ins format determined far from any
absorption edge. Coordinates, displacements, and occupancies will be fixed
at their values in this file.'''
      )
  parser.add_argument(
      'reflections',
      type=str,
      help='''A .hkl file containing intensities (ShelX HKLF 4 format).'''
      )
  parser.add_argument(
      'anom_atom',
      type=str,
      help='''The atom type for which f' and f" will be refined. All atoms of
this type will be refined independently.'''
      )
  group = parser.add_mutually_exclusive_group()
  group.add_argument(
    '-e', '--energy',
    type=float,
    default=None,
    help='''Beam energy in eV.'''
    )
  group.add_argument(
    '-E', '--energy-in-fname',
    type=str,
    default=None,
    help='''First digit and length of energy if given in hkl filename. Format
start,length.'''
    )
  parser.add_argument(
    '-t', '--table',
    action='store_true',
    help='Output in condensed format suitable for further processing')
  parser.add_argument(
    '-c', '--max-cycles',
    type=int,
    default=100,
    help='Stop refinement as soon as the given number of cycles have been '
         'performed')
  parser.add_argument(
    '-o', '--outfile',
    type=str,
    default=None,
    help='Write output to filename OUTFILE (default: print to stdout)')
  parser.add_argument(
    '-O', '--overwrite',
    action='store_true',
    help='No error if OUTFILE exists')
  parser.add_argument(
    '-d', '--stop-deriv',
    type=float,
    default=1e-7,
    help='Stop refinement as soon as the largest absolute value of objective '
         'derivatives is below the given threshold.')
  parser.add_argument(
    '-s', '--stop-shift',
    type=float,
    default=1e-7,
    help='Stop refinement as soon as the Euclidean norm of the vector '
         'of parameter shifts is below the given threshold.')
  parser.add_argument(
    '-p', '--proj',
    type=int,
    default=None,
    help='For testing: specify the projections of f\' and f" tensors to output.'
         ' Current choices are 1: monoCO, Co atoms; 2: monoCOplus, Co atoms')
  return parser

def run(args):

  # adjust file names
  in_root, in_ext = os.path.splitext(args.ref_structure)

  # Check that input files exist
  for filename in (args.ref_structure, args.reflections):
    if not os.path.isfile(filename):
      raise command_line_error("No such file %s" % filename)

  # Check output file
  if args.outfile and os.path.isfile(args.outfile) and not args.overwrite:
    raise command_line_error("Output file {} exists.".format(args.outfile))

  # Load input model and reflections
  if in_ext == '.cif':
    xm = refinement.model.from_cif(model=args.ref_structure)
    xm.add_reflections_with_polarization(args.reflections)
  else:
    raise NotImplementedError("Only .cif input supported")

  # Look for beam energy
  if args.energy:
    energy = args.energy
    wvl = 12398 / energy
  elif args.energy_in_fname:
    estart,elength = args.energy_in_fname.split(',')
    estart = int(estart)
    elength = int(elength)
    energy = float(args.reflections[estart:estart+elength])
    wvl = 12398 / energy
  else:
    energy = None
    sys.stderr.write('''WARNING: Using beam energy from reference model. \
Inelastic form factors for \n non-refined atoms may be inaccurate.\n''')
    wvl = xm.wavelength




  # Load default anomalous scattering factors if wavelength is available
  if wvl:
    xm.xray_structure.set_inelastic_form_factors(wvl, 'sasaki')
  else:
    raise energy_missing_error()

  # At last...
  anom_sc_list=[]
  from cctbx import adp_restraints
  xm.restraints_manager.isotropic_fp_proxies = \
      adp_restraints.shared_isotropic_fp_proxy()
  xm.restraints_manager.isotropic_fdp_proxies = \
      adp_restraints.shared_isotropic_fdp_proxy()
  for i, sc in enumerate(xm.xray_structure.scatterers()):
    sc.flags.set_grad_site(False)
    sc.flags.set_grad_u_iso(False)
    sc.flags.set_grad_u_aniso(False)
    sc.flags.set_grad_occupancy(False)
    if sc.element_symbol().lower() == args.anom_atom.lower():
      sc.convert_to_fp_aniso(xm.xray_structure.unit_cell())
      sc.convert_to_fdp_aniso(xm.xray_structure.unit_cell())
      sc.flags.set_use_fp_fdp_aniso(True)
      sc.flags.set_grad_fp_aniso(True)
      sc.flags.set_grad_fdp_aniso(True)
      anom_sc_list.append(sc)
      xm.restraints_manager.isotropic_fp_proxies.append(
          adp_restraints.isotropic_fp_proxy(i_seqs=(i,), weight=100))
      xm.restraints_manager.isotropic_fdp_proxies.append(
          adp_restraints.isotropic_fdp_proxy(i_seqs=(i,), weight=100))

  ls = xm.least_squares()
  steps = lstbx.normal_eqns_solving.naive_iterations(
    non_linear_ls=ls,
    n_max_iterations=args.max_cycles,
    gradient_threshold=args.stop_deriv,
    step_threshold=args.stop_shift)
  from sys import stderr
  sys.stderr.write(str(ls.r1_factor()))
  sys.stderr.write("\n")

  def fp_proj(sc, vec, uc):
    import numpy as np
    from math import sqrt
    u = np.array(vec)
    g_elem = uc.metrical_matrix()
    g = np.array([
        [g_elem[0], g_elem[3], g_elem[4]],
        [g_elem[3], g_elem[1], g_elem[5]],
        [g_elem[4], g_elem[5], g_elem[2]]])
    g_star = np.linalg.inv(g)
    h = np.dot(u, g)
    h_len = sqrt(np.dot(h, np.dot(g_star, h)))
    h = h/h_len

    fp_star_elem = sc.fp_star
    fp_star = np.array([
        [fp_star_elem[0], fp_star_elem[3], fp_star_elem[4]],
        [fp_star_elem[3], fp_star_elem[1], fp_star_elem[5]],
        [fp_star_elem[4], fp_star_elem[5], fp_star_elem[2]]])

    return np.dot(h, np.dot(fp_star, h))

  def fdp_proj(sc, vec, uc):
    import numpy as np
    from math import sqrt
    u = np.array(vec)
    g_elem = uc.metrical_matrix()
    g = np.array([
        [g_elem[0], g_elem[3], g_elem[4]],
        [g_elem[3], g_elem[1], g_elem[5]],
        [g_elem[4], g_elem[5], g_elem[2]]])
    g_star = np.linalg.inv(g)
    h = np.dot(u, g)
    h_len = sqrt(np.dot(h, np.dot(g_star, h)))
    h = h/h_len

    fdp_star_elem = sc.fdp_star
    fdp_star = np.array([
        [fdp_star_elem[0], fdp_star_elem[3], fdp_star_elem[4]],
        [fdp_star_elem[3], fdp_star_elem[1], fdp_star_elem[5]],
        [fdp_star_elem[4], fdp_star_elem[5], fdp_star_elem[2]]])

    return np.dot(h, np.dot(fdp_star, h))

  def print_fp_cart(sc, uc):
    import numpy as np
    from cctbx import adptbx
    fp_cart_elem = adptbx.u_star_as_u_cart(uc, sc.fp_star)
    fp_cart = np.array([
        [fp_cart_elem[0], fp_cart_elem[3], fp_cart_elem[4]],
        [fp_cart_elem[3], fp_cart_elem[1], fp_cart_elem[5]],
        [fp_cart_elem[4], fp_cart_elem[5], fp_cart_elem[2]]])
    print(fp_cart)

  def print_fdp_cart(sc, uc):
    import numpy as np
    from cctbx import adptbx
    fdp_cart_elem = adptbx.u_star_as_u_cart(uc, sc.fdp_star)
    fdp_cart = np.array([
        [fdp_cart_elem[0], fdp_cart_elem[3], fdp_cart_elem[4]],
        [fdp_cart_elem[3], fdp_cart_elem[1], fdp_cart_elem[5]],
        [fdp_cart_elem[4], fdp_cart_elem[5], fdp_cart_elem[2]]])
    print(fdp_cart)





  # Prepare output
  result = ''

  from cctbx import adptbx
  uc = xm.xray_structure.unit_cell()
#  The following takes the anomalous atoms and writes them in ShelXL format but
#  with their f' and f" tensors (suitably scaled) taking the place of ADPs.
#  The output lines can be copied into a .res file for viewing.
#
  fp_list = []
  for sc in anom_sc_list:
    fp_cif = adptbx.u_star_as_u_cif(uc, sc.fp_star)
    fp_s = [x * -.01 for x in fp_cif]
    #print("{} 5 {:.5f} {:.5f} {:.5f} 11 {:.5f} {:.5f} {:.5f} =\n {:.5f} {:.5f} {:.5f} ".format(
    #  sc.label, sc.site[0], sc.site[1], sc.site[2], fp_s[0], fp_s[1], fp_s[2], fp_s[5], fp_s[4], fp_s[3]))
    fp_list.append("{} 5 {:.5f} {:.5f} {:.5f} 11 {:.5f} {:.5f} {:.5f} =\n {:.5f} {:.5f} {:.5f} ".format(
      sc.label, sc.site[0], sc.site[1], sc.site[2], fp_s[0], fp_s[1], fp_s[2], fp_s[5], fp_s[4], fp_s[3]))
  with open('{}.ins'.format(energy), 'w') as f:
    f.write(make_ins_fstring_monoP_se().format(*fp_list))

  for sc in anom_sc_list:
    fdp_cif = adptbx.u_star_as_u_cif(uc, sc.fdp_star)
    fdp_s = [x * .02 for x in fdp_cif]
    #print("{} 3 {:.5f} {:.5f} {:.5f} 11 {:.5f} {:.5f} {:.5f} =\n {:.5f} {:.5f} {:.5f} ".format(
    #  sc.label, sc.site[0], sc.site[1], sc.site[2], fdp_s[0], fdp_s[1], fdp_s[2], fdp_s[5], fdp_s[4], fdp_s[3]))


  # Directions (in crystal coordinates) on which f' and f" tensors will be
  # projected for output. Only for testing. These all give a projection on a
  # line through the center of the Co6 cluster.
  ProjectionUvwDict = {
  #For monoCO, Co
  1:  [
      ( .124, .150, .142),
      ( .014,-.152, .157),
      (-.269, .058, .046),
      ( .014,-.152, .157),
      (-.269, .058, .046),
      ( .124, .150, .142)],

  #for monoCOplus, Co
  2:  [
      ( .139, .073, .199),
      ( .304, .053,-.149),
      ( .115,-.248, .033),
      ( .304, .053,-.149),
      ( .115,-.248, .033),
      ( .139, .073, .199)],

  #monoP, Se
  3:  [
      ( .259,-.205, .005),
      ( .207, .156, .163),
      (-.018, .097,-.225),
      ( .444, .048,-.054),
      (-.018, .097,-.225),
      ( .444, .048,-.054),
      ( .259,-.205, .005),
      ( .207, .156, .163)]
  }

  if args.table:
    vecs = ProjectionUvwDict[args.proj]
    if energy: result += "{:.1f} ".format(energy)
    else: result += "{} ".format(args.reflections)
    result += "{:.5f} ".format(ls.r1_factor()[0])
    result += "1.00000 "
    for i_sc in range(len(anom_sc_list)):
      result += "{:.3f} ".format(fp_proj(anom_sc_list[i_sc], vecs[i_sc], uc))
    for i_sc in range(len(anom_sc_list)):
      result += "{:.3f} ".format(fdp_proj(anom_sc_list[i_sc], vecs[i_sc], uc))
    result += '\n'

  else:
    result += "\n### REFINE ANOMALOUS SCATTERING FACTORS ###\n"
    result += "Reflections: {}\n\n".format(args.reflections)
    for sc in anom_sc_list:
      result += "{}:\n\tfp: {:.3f}\n\tfdp: {:.3f}\n".format(sc.label, sc.fp, sc.fdp)

  # Write to file or stdout
  if args.outfile:
    with open(args.outfile, 'w') as f:
      f.write(result)
  else:
    print(result)
    #import numpy as np
    #with np.printoptions(precision=2, suppress=True):
    #  for sc in anom_sc_list:
    #    print_fp_cart(sc, uc)
    #  print()
    #  for sc in anom_sc_list:
    #    print_fdp_cart(sc, uc)




if __name__ == '__main__':
  from timeit import default_timer as current_time
  t0 = current_time()
  parser = make_parser()
  args = parser.parse_args()
  try:
    run(args)
  except number_of_arguments_error:
    parser.print_usage()
    sys.exit(1)
  except command_line_error as err:
    print("\nERROR: %s\n" % err, file=sys.stderr)
    parser.print_help()
    sys.exit(1)
  except energy_missing_error as err:
    print('Must provide beam energy on the command line or in the reference '
        'structure.')
    sys.exit(1)
  t1 = current_time()
  if not args.table:
    print("Total time: %.3f s" % (t1 - t0))
