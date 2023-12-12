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
  benchmarks = all_benchmarks.all_benchmarks()

  search_times = [3.253859306999999, 1.2145669139999988, 1.7304103680000011, 1.746582857, 2.149095567, 1.1278407329999993, 99.424533016, 304.050046639, 1.4726406010000233, 2.8289209619999838, 4.684921004999978, 1.9714681400000131, 2.067901212000038, 3.782480743000008, 1.7121255790000305, 1.5168292259999703, 1.1965666930000225, 1.8414282399999706, 1.5644627229999628, 39.362683729000025, 8.454730765999955, 36.76426960200007, 1.8659395999999333, 26.601995644999988, 1.7760869120000962, 300.585933471, 1.9128611399999045, 7.6626471240000456, 1.4998477159999766, 52.767152159000034, 2.266996234999965, 4.854138300000045, 3.957452675000013, 122.83867675900012, 2.5429120940000303, 28.20196477100012, 28.387348124000027, 300.06734223700005, 1.604979749999984, 167.5854413709999, 69.18496835099995, 2.557794559000058, 1.6871340899999723, 300.6700129029998, 138.4407379019999, 1.6507742730000246, 1.5574526300001708, 7.383190665999791, 3.2645593679999365, 24.097928454999874, 11.167181396999695, 1.271145911000076, 5.7601903100003256, 3.817169767999985, 1.9629541089998384, 10.048821293999936, 1.1710925570000654, 2.127285778999976, 1.5304904130002797, 300.6135883910001, 2.656434540000191, 1.5137469920000513, 4.178197791000002, 2.384390453999913, 25.032877164000183, 14.643682460000036, 1.440247994999936, 2.2678873869999734, 245.5946136929997]
  offset = 69
  for i, bm in enumerate(benchmarks[offset:]):
    print("Benchmark #", offset + i, "/", len(benchmarks), ": ", bm.target_program)
    time = run_tf_coder(bm.examples[0].inputs, bm.examples[0].output, bm.constants, bm.description, settings)
    search_times.append(time)
    print("Found in ", time, "s!")
    print(search_times)

  print(search_times)

if __name__ == '__main__':
  app.run(main)
