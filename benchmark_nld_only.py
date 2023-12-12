import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Must happen before importing tf.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU is faster than GPU.

from absl import app  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=unused-import
import tensorflow as tf  # pylint: disable=unused-import

from tf_coder.value_search import colab_interface
from tf_coder.value_search import value_search_settings as settings_module
from tf_coder.benchmarks import all_benchmarks


def get_settings():
  """Specifies settings for TF-Coder. Edit this function!"""
  # How long to search for a solution, in seconds.
  time_limit = 300

  # How many solutions to find before stopping. If more than 1, the entire
  # search will slow down.
  number_of_solutions = 1

  # Whether solutions must use all inputs, at least one input, or no such
  # requirement. Choose one of "all inputs", "one input", "no restriction".
  solution_requirement = 'all inputs'
  assert solution_requirement in ['all inputs', 'one input', 'no restriction']

  return settings_module.from_dict({
      'timeout': time_limit,
      'only_minimal_solutions': False,
      'max_solutions': number_of_solutions,
      'require_all_inputs_used': solution_requirement == 'all inputs',
      'require_one_input_used': solution_requirement == 'one input',
  })


def run_tf_coder(inputs, output, constants, description, settings):
  """Runs TF-Coder on a problem, using the given settings."""
  # Results will be printed to standard output.
  result = colab_interface.run_value_search_from_colab(
      inputs, output, constants, description, settings)

  return result[1]


def main(unused_argv):
    # Load the models.
    colab_interface.warm_up()

    # Acquire the settings
    settings = get_settings()

    # Acquire the benchmarks
    benchmarks = [b for b in all_benchmarks.all_benchmarks() if b.description != ""]

    search_times = []
    offset = 0
    for i, bm in enumerate(benchmarks[offset:]):
        print("Benchmark #", offset + i, "/", len(benchmarks), ": ", bm.target_program)
        time = run_tf_coder(bm.examples[0].inputs, bm.examples[0].output, bm.constants, bm.description, settings)
        search_times.append(time)
        print("Found in ", time, "s!")
        print(search_times)

    print(search_times)



if __name__ == '__main__':
  app.run(main)
