import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required")
class STMambaNetTests(unittest.TestCase):
    def test_stmambanet_accepts_bench_epoch_shapes(self):
        from wyu_bci_bench.frontier_models import make_stmambanet

        model = make_stmambanet(channels=41, classes=3)
        logits = model(torch.randn(2, 41, 480))
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertTrue(torch.isfinite(logits).all())

    def test_stmambanet_is_differentiable(self):
        from wyu_bci_bench.frontier_models import STMambaNet

        model = STMambaNet(channels=8, classes=4, embedding_dim=12, pool_size=5)
        eeg = torch.randn(3, 8, 101, requires_grad=True)
        loss = model(eeg).square().mean()
        loss.backward()
        self.assertIsNotNone(eeg.grad)
        self.assertTrue(torch.isfinite(eeg.grad).all())

    def test_stmambanet_rejects_invalid_inputs(self):
        from wyu_bci_bench.frontier_models import STMambaNet

        with self.assertRaisesRegex(ValueError, "six positive odd"):
            STMambaNet(channels=8, classes=2, kernel_sizes=(3, 5))
        model = STMambaNet(channels=8, classes=2)
        with self.assertRaisesRegex(ValueError, "expected input"):
            model(torch.randn(2, 7, 100))

    def test_eeg_simpleconv_shape_and_subject_head(self):
        from wyu_bci_bench.frontier_models import make_eeg_simpleconv

        model = make_eeg_simpleconv(channels=41, classes=3, features=8,
                                    conv_blocks=3, subjects=5)
        logits, subject_logits = model(torch.randn(4, 41, 480))
        self.assertEqual(tuple(logits.shape), (4, 3))
        self.assertEqual(tuple(subject_logits.shape), (4, 5))

    def test_eeg_simpleconv_author_resampling_contract(self):
        from wyu_bci_bench.frontier_models import EEGSimpleConv

        model = EEGSimpleConv(3, 2, features=4, conv_blocks=1, sfreq=250, resampling=125)
        x = torch.randn(2, 3, 250)
        self.assertEqual(tuple(model(x).shape), (2, 2))
        with self.assertRaisesRegex(ValueError, "specified together"):
            EEGSimpleConv(3, 2, sfreq=250)

    def test_eeg_mftnet_shape_and_gradients(self):
        from wyu_bci_bench.frontier_models import make_eeg_mftnet

        model = make_eeg_mftnet(channels=8, samples=256, classes=4,
                                f1=2, depth_multiplier=2, f2=8)
        eeg = torch.randn(2, 8, 256, requires_grad=True)
        logits = model(eeg)
        self.assertEqual(tuple(logits.shape), (2, 4))
        logits.square().mean().backward()
        self.assertIsNotNone(eeg.grad)
        self.assertTrue(torch.isfinite(eeg.grad).all())

    def test_eegmamba_patch_embedding_contract(self):
        from wyu_bci_bench.frontier_models import EEGMambaPatchEmbedding

        embedding = EEGMambaPatchEmbedding()
        patches = embedding(torch.randn(2, 22, 4, 200))
        self.assertEqual(tuple(patches.shape), (2, 22, 4, 200))
        masked = embedding(torch.randn(2, 22, 4, 200), torch.ones(2, 22, 4))
        self.assertEqual(tuple(masked.shape), (2, 22, 4, 200))

    def test_eegmamba_optional_kernel_failure_is_explicit(self):
        from wyu_bci_bench.frontier_models import make_eegmamba

        try:
            import mamba_ssm  # noqa: F401
        except ImportError:
            with self.assertRaisesRegex(ImportError, "optional Mamba-SSM CUDA package"):
                make_eegmamba(classes=4)

    def test_eegm2_unet_structure_and_odd_input_length(self):
        from wyu_bci_bench.frontier_models import EEGM2

        model = EEGM2(channels=6, output_channels=6, scale_factor=4,
                      state_block_factory=lambda _: torch.nn.Identity())
        eeg = torch.randn(2, 6, 193, requires_grad=True)
        reconstructed = model(eeg)
        self.assertEqual(tuple(reconstructed.shape), (2, 6, 193))
        reconstructed.abs().mean().backward()
        self.assertTrue(torch.isfinite(eeg.grad).all())

    def test_eegm2_representation_and_dual_domain_loss(self):
        from wyu_bci_bench.frontier_models import (
            eegm2_representation_summary, eegm2_temporal_spectral_loss,
        )

        signal = torch.randn(3, 5, 64)
        self.assertEqual(tuple(eegm2_representation_summary(signal).shape), (3, 45))
        loss = eegm2_temporal_spectral_loss(signal, signal * 0.9)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0)

    def test_additional_frontier_architecture_shapes(self):
        from wyu_bci_bench.frontier_advanced import (
            AGTCNet, CFSPMNet, DualExpertTransformer, EEGTriNetPlusPlus, GADCN,
            BFMMSDA, GeometryEEGNet, GraphCSPNet,
            MIMamba, PTSM, PhaseSpaceSPDNet,
            SPDTokenTransformer, SubjectAlignedEncoder,
            TensorCSPNet, maml_episode_loss,
        )

        x = torch.randn(2, 4, 64)
        cfspm = CFSPMNet(4, 3, embedding=8, depth=1, f1=2,
                          depth_multiplier=2, pool1=2, pool2=2)
        models = [
            MIMamba(4, 3, width=8), cfspm,
            DualExpertTransformer(4, 3, width=8), PTSM(4, 3, width=8),
            EEGTriNetPlusPlus(4, 3, width=8, patch_size=32, patch_stride=16,
                              lstm_hidden=4, transformer_layers=1),
            SPDTokenTransformer(4, 3, width=8, patch_size=16, patch_stride=8),
            AGTCNet(4, 3, width=8), GeometryEEGNet(4, 3, rank=3),
            PhaseSpaceSPDNet(4, 3, rank=3), GADCN(4, 3, rank=3, width=8),
        ]
        for model in models:
            logits = model(x)
            self.assertEqual(tuple(logits.shape), (2, 3), type(model).__name__)
            self.assertTrue(torch.isfinite(logits).all(), type(model).__name__)

        aligned = SubjectAlignedEncoder(4, 3, subjects=5, width=8, adapter_rank=2)
        aligned_logits = aligned(x, subject_ids=torch.tensor([1, 3]))
        self.assertEqual(tuple(aligned_logits.shape), (2, 3))
        self.assertEqual(aligned.adapter_left.numel() + aligned.adapter_right.numel(), 2 * 4 * 2 * 5)
        with self.assertRaisesRegex(ValueError, "subject_ids are required"):
            aligned(x)

        source_features = torch.tensor([[1., 0.], [0.9, 0.1], [0., 1.], [0.1, 0.9]])
        source_labels = torch.tensor([0, 0, 1, 1])
        source_domains = torch.tensor([0, 1, 0, 1])
        target_features = torch.tensor([[0.95, 0.05], [0.05, 0.95]])
        selected = cfspm.select_pseudo_labels(source_features, source_labels, source_domains,
                                              target_features, confidence_threshold=0.5)
        self.assertEqual(selected["labels"].tolist(), [0, 1])
        self.assertTrue(selected["mask"].all())

        domain_adapter = BFMMSDA(torch.nn.Identity(), 2, 2, top_k=1)
        selected_domains = domain_adapter.select_sources(
            torch.tensor([[0., 0.], [0.1, 0.], [5., 5.], [5.1, 5.]]),
            torch.tensor([0, 0, 1, 1]), torch.tensor([[1., 0.], [0.95, 0.05]]),
        )
        self.assertEqual(selected_domains["domains"].numel(), 1)
        self.assertEqual(int(selected_domains["domains"][0]), 0)
        self.assertTrue(torch.allclose(selected_domains["weights"].sum(), torch.tensor(1.)))

        raw_covariance = torch.randn(1, 2, 9, 22, 12)
        covariance = raw_covariance @ raw_covariance.transpose(-1, -2) / 12 + 1e-3 * torch.eye(22)
        tensor_model = TensorCSPNet()
        tensor_logits = tensor_model(covariance)
        self.assertEqual(tuple(tensor_logits.shape), (1, 4))
        self.assertTrue(torch.isfinite(tensor_logits).all())

        graph_raw = torch.randn(2, 2, 22, 12)
        graph_covariance = graph_raw @ graph_raw.transpose(-1, -2) / 12 + 1e-3 * torch.eye(22)
        graph_model = GraphCSPNet(segments=2, graph_matrix=torch.eye(2))
        graph_logits = graph_model(graph_covariance)
        self.assertEqual(tuple(graph_logits.shape), (2, 4))
        self.assertTrue(torch.isfinite(graph_logits).all())

        trinet = EEGTriNetPlusPlus(4, 3, width=8, patch_size=32, patch_stride=16,
                                   lstm_hidden=4, transformer_layers=1)
        support_x, query_x = torch.randn(4, 4, 64), torch.randn(2, 4, 64)
        episode = maml_episode_loss(trinet, support_x, torch.tensor([0, 0, 1, 1]),
                                    query_x, torch.tensor([0, 1]), steps=1)
        episode.backward()
        self.assertTrue(torch.isfinite(episode))
        self.assertIsNotNone(trinet.head.weight.grad)

    def test_bfm_msda_uses_foundation_selection_and_dual_alignment(self):
        from wyu_bci_bench.frontier_advanced import BFMMSDA

        model=BFMMSDA(torch.nn.Identity(),3,2,top_k=1,warm_up=1)
        source=torch.randn(8,3,requires_grad=True)
        target=torch.randn(4,3,requires_grad=True)
        domains=torch.tensor([0,0,0,0,1,1,1,1])
        source_bfm=torch.tensor([[1.,0.],[0.9,0.1],[1.,0.],[0.8,0.2],
                                 [0.,1.],[0.1,0.9],[0.,1.],[0.2,0.8]])
        target_bfm=torch.tensor([[0.9,0.1],[0.95,0.05]])
        selection=model.select_sources(source_bfm,domains,target_bfm)
        self.assertEqual(selection["domains"].tolist(),[0])
        result=model.adaptation_loss(source,torch.tensor([0,0,1,1,0,0,1,1]),
                                     domains,target,source_bfm,target_bfm,epoch=2)
        self.assertTrue(result["alignment_active"])
        self.assertTrue(torch.isfinite(result["cs_loss"]))
        self.assertTrue(torch.isfinite(result["ccs_loss"]))
        result["loss"].backward()
        self.assertTrue(torch.isfinite(source.grad).all())
        self.assertTrue(torch.isfinite(target.grad).all())

    def test_frontier_models_have_finite_input_gradients(self):
        from wyu_bci_bench.frontier_advanced import (
            AGTCNet, CFSPMNet, GADCN, MIMamba, SPDTokenTransformer,
        )

        factories = (
            lambda: MIMamba(4, 3, width=8, state_size=4, kernel_size=7),
            lambda: CFSPMNet(4, 3, embedding=8, depth=1, f1=2,
                             depth_multiplier=2, pool1=2, pool2=2),
            lambda: SPDTokenTransformer(4, 3, width=8, patch_size=16, patch_stride=8),
            lambda: AGTCNet(4, 3, width=8),
            lambda: GADCN(4, 3, rank=3, width=8),
        )
        for factory in factories:
            model = factory()
            x = torch.randn(2, 4, 64, requires_grad=True)
            loss = model(x).square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss), type(model).__name__)
            self.assertIsNotNone(x.grad, type(model).__name__)
            self.assertTrue(torch.isfinite(x.grad).all(), type(model).__name__)

    def test_spd_token_transformer_pads_short_epochs(self):
        from wyu_bci_bench.frontier_advanced import SPDTokenTransformer

        model = SPDTokenTransformer(3, 2, width=8, patch_size=16, patch_stride=8)
        logits = model(torch.randn(2, 3, 7))
        self.assertEqual(tuple(logits.shape), (2, 2))
        with self.assertRaisesRegex(ValueError, "expected raw EEG"):
            model(torch.randn(2, 2, 32))

    def test_phase_spdnet_uses_phase_covariance_and_bimap_head(self):
        from wyu_bci_bench.frontier_advanced import PhaseSpaceSPDNet

        model=PhaseSpaceSPDNet(3,2,delay=2,order=2,rank=4)
        x=torch.randn(5,3,64,requires_grad=True)
        logits,features=model(x,return_features=True)
        self.assertEqual(tuple(logits.shape),(5,2))
        self.assertEqual(tuple(features.shape),(5,10))
        self.assertEqual(tuple(model.bimap.shape),(9,4))
        logits.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        covariance=model.phase_covariance(x.detach())
        self.assertEqual(tuple(covariance.shape),(5,9,9))

    def test_eegmamba_2024_is_distinct_multitask_model(self):
        from wyu_bci_bench import EEGMamba2024

        model=EEGMamba2024(4,3,tasks=2,width=8,experts=3,top_k=2,
                           state_size=3,temporal_kernel=7)
        x=torch.randn(2,4,24,requires_grad=True)
        logits,gates,universal_weight=model(x,torch.tensor([0,1]),return_gates=True)
        self.assertEqual(tuple(logits.shape),(2,3))
        self.assertEqual(tuple(gates.shape),(2,3))
        self.assertTrue(torch.allclose(gates.sum(-1),torch.ones(2)))
        self.assertTrue(torch.allclose(universal_weight,1-gates.max(-1,keepdim=True).values))
        logits.mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_gadcn_exposes_dct_family_and_congruence_depth(self):
        from wyu_bci_bench.frontier_advanced import GADCN

        x=torch.randn(2,5,80)
        depths={}
        for variant,expected in (("dct",1),("dldct",2),("ddct_unet",3)):
            model=GADCN(5,3,rank=4,width=8,variant=variant)
            logits=model(x)
            self.assertEqual(tuple(logits.shape),(2,3))
            self.assertTrue(torch.isfinite(logits).all())
            depths[variant]=len(model.encoder)
        self.assertEqual(depths,{"dct":1,"dldct":2,"ddct_unet":3})

    def test_spd_token_transformer_geometries_and_multiband_covariances(self):
        from wyu_bci_bench.frontier_advanced import SPDTokenTransformer

        raw=torch.randn(3,4,48,requires_grad=True)
        for geometry in ("bwspd","logeuclidean","euclidean"):
            model=SPDTokenTransformer(4,3,width=8,patch_size=16,patch_stride=8,
                                      embedding=geometry,bn_embed=True,depth=1,
                                      heads=2,feedforward=16)
            logits=model(raw)
            self.assertEqual(tuple(logits.shape),(3,3))
            self.assertTrue(torch.isfinite(logits).all())
        windows=raw.detach().unfold(-1,16,8).transpose(1,2)
        windows=windows-windows.mean(-1,keepdim=True)
        covariance=windows@windows.transpose(-1,-2)/15+1e-5*torch.eye(4)
        model=SPDTokenTransformer(4,3,width=8,embedding="bwspd",depth=1,
                                  heads=2,feedforward=16,max_tokens=8)
        result=model(covariance,return_tokens=True)
        self.assertEqual(tuple(result["tokens"].shape),(3,5,8))

    def test_ptsm_factorized_masks_and_multitask_objective(self):
        from wyu_bci_bench.frontier_advanced import PTSM

        model = PTSM(4, 3, width=8, subject_classes=5)
        x = torch.randn(6, 4, 64, requires_grad=True)
        outputs = model(x, return_features=True)
        self.assertEqual(tuple(outputs["logits"].shape), (6, 3))
        self.assertEqual(tuple(outputs["spatial_masks"][0].shape), (6, 4, 1))
        self.assertEqual(len(outputs["temporal_masks"]), 3)
        losses = model.training_loss(x, torch.tensor([0, 1, 2, 0, 1, 2]),
                                     torch.tensor([0, 1, 2, 3, 4, 0]))
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_frontier_losses_are_finite_and_differentiable(self):
        from wyu_bci_bench.frontier_advanced import (
            cs_divergence_loss, geometric_consistency_loss,
        )

        source, target = torch.randn(8, 5, requires_grad=True), torch.randn(7, 5)
        signal, view = torch.randn(3, 4, 32, requires_grad=True), torch.randn(3, 4, 32)
        loss = cs_divergence_loss(source, target) + geometric_consistency_loss(signal, view)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(source.grad).all())
        self.assertTrue(torch.isfinite(signal.grad).all())


if __name__ == "__main__":
    unittest.main()
