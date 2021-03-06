from __future__ import absolute_import, division, print_function
from xfel.merging.application.worker import worker
from dxtbx.imageset import ImageSetFactory
from dials.array_family import flex
import os

from dials.command_line.stills_process import Processor
class integrate_only_processor(Processor):
  def __init__(self, params):
    self.params = params

class integrate(worker):
  """
  Calls the stills process version of dials.integrate
  """
  def __init__(self, params, mpi_helper=None, mpi_logger=None):
    super(integrate, self).__init__(params=params, mpi_helper=mpi_helper, mpi_logger=mpi_logger)

  def __repr__(self):
    return 'Integrate reflections'

  def run(self, experiments, reflections):
    from dials.util import log
    self.logger.log_step_time("INTEGRATE")

    logfile = os.path.splitext(self.logger.rank_log_file_path)[0] + "_integrate.log"
    log.config(logfile=logfile)
    processor = integrate_only_processor(self.params)

    # Re-generate the image sets using their format classes so we can read the raw data
    # Integrate the experiments one at a time to not use up memory
    all_integrated = flex.reflection_table()
    for expt_id, expt in enumerate(experiments):
      expt.imageset = ImageSetFactory.make_imageset(expt.imageset.paths(), single_file_indices=expt.imageset.indices())
      refls = reflections.select(reflections['exp_id'] == expt.identifier)
      idents = refls.experiment_identifiers()
      del idents[expt_id]
      idents[0] = expt.identifier
      refls['id'] = flex.int(len(refls), 0)

      integrated = processor.integrate(experiments[expt_id:expt_id+1], refls)

      idents = integrated.experiment_identifiers()
      del idents[0]
      idents[expt_id] = expt.identifier
      integrated['id'] = flex.int(len(integrated), expt_id)
      all_integrated.extend(integrated)
      expt.imageset.clear_cache()

    return experiments, reflections

if __name__ == '__main__':
  from xfel.merging.application.worker import exercise_worker
  exercise_worker(integrate)
