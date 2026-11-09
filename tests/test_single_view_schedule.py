import unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from pipeline.frame_plan import FramePlanController

class SingleViewScheduleTests(unittest.TestCase):
    def fixture(self, regional=False, two=False, admission=False):
        views={f"cam{i:02}.png":NS(path=Path(f"cam{i:02}.png")) for i in range(1,21)}
        fake=NS(transaction_camera_views=lambda: views, config=NS(birth=NS(regional_postfit_admission=admission, regional_color_birth=regional),full_update=NS(regional_repair=two,repair_selection="all_points",audit_camera_sequence=False)), datamodule=NS(config=NS(training_view_seed=37)),current_frame=210,full_update_active=two)
        return fake
    def sample(self,m,step):
        return FramePlanController.frame_plan_training_views(m,step,1)
    def test_one_view_cycle_no_global_rng_and_no_test(self):
        m=self.fixture();state=torch.random.get_rng_state();names=[self.sample(m,s)[0].path.name for s in range(1,21)]
        self.assertEqual(len(set(names)),20);self.assertNotIn("cam00.png",names);self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        self.assertEqual(names,[self.sample(m,s)[0].path.name for s in range(1,21)])
    def test_auxiliary_views_are_not_optimizer_views(self):
        m=self.fixture(regional=True);result=self.sample(m,200);self.assertEqual(len(result),1);self.assertEqual(len(m._plan_step_views),4)
        m=self.fixture(two=True);result=self.sample(m,200);self.assertEqual(len(result),1);self.assertEqual(len(m._plan_step_views),1)
    def test_reserved_postfit_admission_uses_one_training_view(self):
        m=self.fixture(admission=True)
        self.assertEqual(len(self.sample(m,200)),1)
        proposed={v.path.name for v in m._plan_step_views}
        self.assertEqual(len(proposed),4)
        m._plan_admission_camera_names=[name for name in m.transaction_camera_views() if name not in proposed][:4]
        for step in (201,250,300):
            self.assertEqual(len(self.sample(m,step)),1)
            self.assertTrue({v.path.name for v in m._plan_step_views}.isdisjoint(m._plan_admission_camera_names))

if __name__ == "__main__": unittest.main()
