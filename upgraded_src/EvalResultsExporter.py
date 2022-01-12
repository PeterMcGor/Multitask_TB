import tensorflow as tf
import json
import numpy as np


class EvalResultsExporter(tf.estimator.Exporter):
    """Passed into an EvalSpec for saving the result of the final evaluation
  step locally or in Google Cloud Storage.
  """

    def __init__(self, name):
        assert name, '"name" argument is required.'
        self._name = name

    @property
    def name(self):
        return self._name

    def export(self, estimator, export_path, checkpoint_path, eval_result, is_the_final_export):

        if not is_the_final_export:
            return None

        tf.compat.v1.logging.info(('EvalResultsExporter (name: %s) '
                         'running after final evaluation.') % self._name)
        tf.compat.v1.logging.info('export_path: %s' % export_path)
        tf.compat.v1.logging.info('eval_result: %s' % eval_result)

        eval_results = {}

        for key, value in eval_result.items():
            if isinstance(value, np.float32):
                print('KEY',key,'VAL',value)
                eval_results[key] = value.item()

        tf.io.gfile.mkdir(export_path)

        print('EVAL_RESULTS', eval_result)
        with tf.io.gfile.GFile('%s/eval_results.json' % export_path, 'w') as f:
            f.write(json.dumps(eval_results))
