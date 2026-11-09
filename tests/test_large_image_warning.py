import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from libgs.data.dataset import base
from pipeline import data

class LargeImageWarningTests(unittest.TestCase):
 def setUp(self):
  self.value=base.WARNED.value
 def tearDown(self):
  with base.WARNED.get_lock():base.WARNED.value=self.value
 def test_main_process_resizes_and_warns_once(self):
  with base.WARNED.get_lock():base.WARNED.value=0
  ds=SimpleNamespace(image_size=(2689,2016),resolution=-1,resolution_scale=1.)
  with patch.object(base,'get_worker_info',return_value=None),contextlib.redirect_stdout(io.StringIO()) as stream:
   self.assertEqual(base.Dataset.resolve_size(ds,SimpleNamespace(width=2689,height=2016)),(1600,1199))
   self.assertEqual(base.Dataset.resolve_size(ds,SimpleNamespace(width=2689,height=2016)),(1600,1199))
  self.assertEqual(stream.getvalue().count('[WARNING]'),1)
 def test_worker_calls_original_warning(self):
  with patch.object(base,'get_worker_info',return_value=SimpleNamespace(id=0)),patch.object(data,'_libgs_original_resolution_warning') as original:
   data._safe_resolution_warning();original.assert_called_once_with()
 def test_small_and_explicit_resolution_unchanged(self):
  with patch.object(base,'warning',side_effect=AssertionError('unexpected warning')):
   self.assertEqual(base.Dataset.resolve_size(SimpleNamespace(image_size=(1345,1008),resolution=-1,resolution_scale=1.),SimpleNamespace(width=1345,height=1008)),(1345,1008))
   self.assertEqual(base.Dataset.resolve_size(SimpleNamespace(image_size=(2689,2016),resolution=2,resolution_scale=1.),SimpleNamespace(width=2689,height=2016)),(1344,1008))
if __name__=='__main__':unittest.main()
