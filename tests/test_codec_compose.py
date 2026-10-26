"""Decoder-equivalent CPU composition without recursive codec work."""
import unittest
from unittest.mock import patch
import torch
from tests.test_codec import packet_fixture
from tests import test_spatial_motion as spatial_tests
from pipeline.model import rigid_tree_codec as codec


class SurvivorCompositionTests(unittest.TestCase):
    def test_nonspatial_matches_wire_with_survivors_and_owner_edits(self):
        prior,args=packet_fixture(16)
        rows=torch.tensor([0,2,4,6,8,10,12,14])
        owner=prior.owner[rows].clone();owner[0]=0
        parents=prior.parents
        rt,rr=args[6:8];gt,gr=args[8:10]
        metadata=(1.234567890123,.000123456789,.00123456789)
        steps=dict(gs_translation_step_extent=.0000123456789,gs_rotation_step_rad=.000987654321)
        inputs=(prior,1,metadata[0],rows,owner,parents,rt,rr,gt[rows],gr[rows],
            torch.empty(0,dtype=torch.long),torch.empty(0,3),torch.empty(0,4),*metadata[1:])
        decoded=codec.decode_motion(codec.encode_motion(*inputs,**steps),prior)
        xyz,q=codec.compose_decoded_survivors(prior,metadata[0],rows,owner,parents,
            rt,rr,gt[rows],gr[rows],*metadata[1:],**steps)
        self.assertTrue(torch.equal(xyz,decoded.xyz));self.assertTrue(torch.equal(q,decoded.rotation))

    def test_spatial_relative_offsets_match_wire_and_never_recurse(self):
        for relative in (False,True):
            with self.subTest(relative=relative):
                model=spatial_tests.SpatialGatedMotionTests().model(repetitions=2)
                prior=model.codec_previous_state
                # Nonunit raw quaternions detect unwanted normalization.
                prior.rotation=prior.rotation*1.00123
                rows=torch.arange(len(prior.xyz))[1:]
                owner=prior.owner[rows];parents=prior.parents;opened=prior.spatial_open
                g=torch.Generator().manual_seed(37)
                rt=torch.randint(-20,21,(len(parents),3),generator=g,dtype=torch.int16);rt[0]=0
                rr=torch.randint(-20,21,rt.shape,generator=g,dtype=torch.int16);rr[0]=0
                gt=torch.randint(-30,31,(len(rows),3),generator=g,dtype=torch.int16)
                gr=torch.randint(-30,31,gt.shape,generator=g,dtype=torch.int16)
                extent=1.234567890123;step=.000123456789;rstep=.00123456789
                options=dict(gs_translation_step_extent=.0000123456789,gs_rotation_step_rad=.000987654321,
                    spatial_open=opened,spatial_relative=relative)
                xyz,q=codec.compose_decoded_survivors(prior,extent,rows,owner,parents,rt,rr,gt,gr,step,rstep,**options)
                sources=torch.tensor([0,len(rows)-1]);birth_xyz=xyz[sources]+.0001
                birth_q=torch.nn.functional.normalize(q[sources],dim=-1)+.001
                inputs=(prior,1,extent,rows,owner,parents,rt,rr,gt,gr,sources,birth_xyz,birth_q,step,rstep)
                with patch.object(codec,'decode_motion',side_effect=AssertionError('encoder must not decode')):
                    packet=codec.encode_motion(*inputs,birth_pose_steps=(.00001,.001),compression='auto',**options)
                decoded=codec.decode_motion(packet,prior)
                self.assertTrue(torch.equal(xyz,decoded.xyz[:len(rows)]))
                self.assertTrue(torch.equal(q,decoded.rotation[:len(rows)]))
                expected_xyz=xyz[sources]+torch.round((birth_xyz-xyz[sources])/.00001)*.00001
                predictor=torch.nn.functional.normalize(q[sources],dim=-1)
                expected_q=predictor+torch.round((birth_q-predictor)/.001)*.001
                self.assertTrue(torch.equal(expected_xyz,decoded.xyz[len(rows):]))
                self.assertTrue(torch.equal(expected_q,decoded.rotation[len(rows):]))

    def test_motion_convenience_method_does_not_mutate_or_encode(self):
        model=spatial_tests.SpatialGatedMotionTests().model(repetitions=2)
        before=model.codec_previous_state.digest()
        with patch.object(codec,'encode_motion',side_effect=AssertionError('pure composition')):
            xyz,q=model.decoded_survivor_pose()
        self.assertEqual(before,model.codec_previous_state.digest())
        rt,rr,gt,gr=model._values(hard=True)
        empty=torch.empty(0,dtype=torch.long)
        packet=codec.encode_motion(model.frame_reference_state,model.frame_index,model.extent,
            model.previous_rows,model.owner,model.parents,rt,rr,gt,gr,empty,torch.empty(0,3),torch.empty(0,4),
            model.config.motion_xyz_step_extent,model.config.motion_rotation_step_rad,
            gs_translation_step_extent=model.config.gs_xyz_step_extent,gs_rotation_step_rad=model.config.gs_rotation_step_rad,
            spatial_open=model.opened,spatial_relative=model.relative_regions)
        decoded=codec.decode_motion(packet,model.frame_reference_state)
        self.assertTrue(torch.equal(xyz,decoded.xyz));self.assertTrue(torch.equal(q,decoded.rotation))

if __name__=='__main__':unittest.main()
