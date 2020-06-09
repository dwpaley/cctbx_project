from __future__ import absolute_import, division, print_function
import iotbx.pdb
from libtbx.test_utils import approx_equal
from cctbx.sgtbx import space_group_info
from cctbx.development import random_structure
import cctbx.maptbx.box
from libtbx import group_args
import iotbx.pdb
from iotbx.map_manager import map_manager
from iotbx.map_model_manager import map_model_manager
import mmtbx.model
from scitbx.array_family import flex

def get_random_structure_and_map(
   use_static_structure=False,
   random_seed=171413,
  ):

  if use_static_structure:
    mmm=map_model_manager()
    mmm.generate_map()
    return group_args(model = mmm.model(), mm = mmm.map_manager())
  import random
  random.seed(random_seed)
  i=random.randint(1,714717)
  flex.set_random_seed(i)

  xrs = random_structure.xray_structure(
    space_group_info = space_group_info(19),
    volume_per_atom  = 25.,
    elements         = ('C', 'N', 'O', 'H')*10,
    min_distance     = 1.5)
  fc = xrs.structure_factors(d_min=2).f_calc()
  fft_map = fc.fft_map(resolution_factor=0.25)
  fft_map.apply_volume_scaling()
  ph = iotbx.pdb.input(
    source_info=None, lines=xrs.as_pdb_file()).construct_hierarchy()
  ph.atoms().set_xyz(xrs.sites_cart())
  map_data = fft_map.real_map_unpadded()
  mm = map_manager(
    unit_cell_grid             = map_data.accessor().all(),
    unit_cell_crystal_symmetry = fc.crystal_symmetry(),
    origin_shift_grid_units    = (0,0,0),
    map_data                   = map_data)
  model = mmtbx.model.manager(
    model_input=None, pdb_hierarchy=ph, crystal_symmetry=fc.crystal_symmetry())
  return group_args(model = model, mm = mm)

def exercise_around_model():
  mam = get_random_structure_and_map(use_static_structure=True)

  map_data_orig   = mam.mm.map_data().deep_copy()
  sites_frac_orig = mam.model.get_sites_frac().deep_copy()
  sites_cart_orig = mam.model.get_sites_cart().deep_copy()
  cs_orig         = mam.model.crystal_symmetry()

  box = cctbx.maptbx.box.around_model(
    map_manager = mam.mm,
    model       = mam.model.deep_copy(),
    cushion     = 10,
    wrapping    = True)
  new_mm1 = box.map_manager
  new_mm2 = box.apply_to_map(map_manager=mam.mm.deep_copy())
  assert approx_equal(new_mm1.map_data(), new_mm2.map_data())

  new_model1 = box.model
  new_model2 = box.apply_to_model(model=mam.model.deep_copy())
  assert new_model1.crystal_symmetry().is_similar_symmetry(
         new_model2.crystal_symmetry())
  assert new_model1.crystal_symmetry().is_similar_symmetry(
         box.crystal_symmetry)

  assert approx_equal(new_model1.get_sites_cart()[0],(19.705233333333336, 15.631525, 13.5040625))
  # make sure things did change
  assert new_mm2.map_data().size() != map_data_orig.size()

  # make sure things are changed in-place and are therefore different from start
  assert box.map_manager.map_data().size() != map_data_orig.size()
  assert box.model.get_sites_frac() != sites_frac_orig
  assert box.model.get_sites_cart() !=  sites_cart_orig
  assert (not cs_orig.is_similar_symmetry(box.model.crystal_symmetry()))

  # make sure box, model and map_manager remember original crystal symmetry
  assert cs_orig.is_similar_symmetry(box.original_crystal_symmetry)
  assert cs_orig.is_similar_symmetry(
    box.map_manager.original_unit_cell_crystal_symmetry)

  assert approx_equal (box.model.get_shift_manager().shift_cart,
     [5.229233333333334, 5.061524999999999, 5.162062499999999])

  assert box.model.get_shift_manager(
      ).shifted_crystal_symmetry.is_similar_symmetry(
     box.model.crystal_symmetry())
  assert box.model.get_shift_manager(
      ).original_crystal_symmetry.is_similar_symmetry(cs_orig)
  assert (not box.model.get_shift_manager(
      ).shifted_crystal_symmetry.is_similar_symmetry(cs_orig))

  assert approx_equal(
     box.model._figure_out_hierarchy_to_output(do_not_shift_back=False
       ).atoms().extract_xyz()[0],
        (14.476, 10.57, 8.342))

  # make sure we can stack shifts
  sel=box.model.selection("resseq 219:219")
  m_small=box.model.select(selection=sel)
  # Just until deep_copy() deep-copies shift_manager...
  m_small._shift_manager=m_small.get_shift_manager().deep_copy()

  assert approx_equal(box.model.get_shift_manager().shift_cart,
     m_small.get_shift_manager().shift_cart)
  assert not (box.model.get_shift_manager() is m_small.get_shift_manager())


  # Now box again:
  small_box = cctbx.maptbx.box.around_model(
    map_manager = mam.mm,
    model       = m_small,
    cushion     = 5,
    wrapping    = True)

  # Make sure nothing was zeroed out in this map (wrapping=True)
  assert new_mm1.map_data().as_1d().count(0)==0

  # Now without wrapping...
  box = cctbx.maptbx.box.around_model(
    map_manager = mam.mm,
    model       = mam.model.deep_copy(),
    cushion     = 10,
    wrapping    = False)

  # make sure things are changed in-place and are therefore different from start
  assert box.map_manager.map_data().size() != map_data_orig.size()
  assert box.model.get_sites_frac() != sites_frac_orig
  assert box.model.get_sites_cart() !=  sites_cart_orig
  assert (not cs_orig.is_similar_symmetry(box.model.crystal_symmetry()))

  # make sure box, model and map_manager remember original crystal symmetry
  assert cs_orig.is_similar_symmetry(box.original_crystal_symmetry)
  assert cs_orig.is_similar_symmetry(
    box.map_manager.original_unit_cell_crystal_symmetry)

  assert box.map_manager.map_data().as_1d().count(0)==81264

  # Now specify bounds directly
  box = cctbx.maptbx.box.with_bounds(
    map_manager = mam.mm.deep_copy(),
    lower_bounds= (-7, -7, -7),
    upper_bounds= (37, 47, 39),
    wrapping    = False)

  new_model=box.apply_to_model(mam.model.deep_copy())
  # make sure things are changed in-place and are therefore different from start
  assert box.map_manager.map_data().size() != map_data_orig.size()
  assert new_model.get_sites_frac() != sites_frac_orig
  assert new_model.get_sites_cart() !=  sites_cart_orig
  assert (not cs_orig.is_similar_symmetry(new_model.crystal_symmetry()))

  # make sure box, model and map_manager remember original crystal symmetry
  assert cs_orig.is_similar_symmetry(box.original_crystal_symmetry)
  assert cs_orig.is_similar_symmetry(
    box.map_manager.original_unit_cell_crystal_symmetry)

  assert box.map_manager.map_data().as_1d().count(0)==81264

  # Now specify bounds directly and init with model
  box = cctbx.maptbx.box.with_bounds(
    map_manager = mam.mm.deep_copy(),
    lower_bounds= (-7, -7, -7),
    upper_bounds= (37, 47, 39),
    wrapping    = False,
    model = mam.model.deep_copy())

  new_model=box.model
  # make sure things are changed in-place and are therefore different from start
  assert box.map_manager.map_data().size() != map_data_orig.size()
  assert new_model.get_sites_frac() != sites_frac_orig
  assert new_model.get_sites_cart() !=  sites_cart_orig
  assert (not cs_orig.is_similar_symmetry(new_model.crystal_symmetry()))

  # make sure box, model and map_manager remember original crystal symmetry
  assert cs_orig.is_similar_symmetry(box.original_crystal_symmetry)
  assert cs_orig.is_similar_symmetry(
    box.map_manager.original_unit_cell_crystal_symmetry)

  assert box.map_manager.map_data().as_1d().count(0)==81264

  #
  # IF you are about to change this - THINK TWICE!
  #
  import inspect
  r = inspect.getargspec(cctbx.maptbx.box.around_model.__init__)
  assert r.args == ['self', 'map_manager', 'model', 'cushion','wrapping'], r.args
  r = inspect.getargspec(cctbx.maptbx.box.with_bounds.__init__)
  assert r.args == ['self', 'map_manager', 'lower_bounds', 'upper_bounds','wrapping','model'], r.args

if (__name__ == "__main__"):
  exercise_around_model()