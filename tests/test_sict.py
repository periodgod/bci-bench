import unittest

import torch

from wyu_bci_bench.sict import SICTAdapter, SICTStream


class SICTTests(unittest.TestCase):
    def test_zero_initialized_adapter_has_zero_residual(self):
        torch.manual_seed(4)
        model = SICTAdapter().eval()
        q = torch.randn(91)
        h = torch.randn(5, 91)
        torch.testing.assert_close(model.forward_step(q, h), torch.zeros(3), rtol=0, atol=0)

    def test_step_uses_only_supplied_prefix_and_replays_exactly(self):
        torch.manual_seed(7)
        model = SICTAdapter().eval()
        with torch.no_grad():
            model.output.weight.normal_()
            model.output.bias.normal_()
        q = torch.randn(91)
        prefix = torch.randn(8, 91)
        base = model.forward_step(q, prefix)
        extended_history = torch.cat((prefix, torch.randn(6, 91)), dim=0)
        same_query = model.forward_step(q, extended_history[:len(prefix)])
        torch.testing.assert_close(base, same_query, rtol=0, atol=0)
        torch.testing.assert_close(base, model.forward_step(q, prefix), rtol=0, atol=0)

    def test_stream_reset_and_session_isolation(self):
        torch.manual_seed(11)
        model = SICTAdapter().eval()
        with torch.no_grad():
            model.output.weight.normal_()
        stream = SICTStream(model)
        first = torch.randn(91)
        second = torch.randn(91)
        stream.predict_one(first)
        got = stream.predict_one(second)
        stream.reset_state()
        reset = stream.predict_one(second)
        fresh = model.forward_step(second)
        torch.testing.assert_close(reset, fresh, rtol=0, atol=0)
        self.assertFalse(torch.equal(got, reset))

    def test_memory_is_bounded_and_ablation_contracts_hold(self):
        torch.manual_seed(13)
        model = SICTAdapter(memory=3).eval()
        with torch.no_grad():
            model.output.weight.normal_()
        q = torch.randn(91)
        history = torch.randn(7, 91)
        full = model.forward_step(q, history)
        truncated = model.forward_step(q, history[-3:])
        torch.testing.assert_close(full, truncated, rtol=0, atol=0)
        torch.testing.assert_close(model.forward_step(q, history, ablation="no_memory"),
                                   model.forward_step(q), rtol=0, atol=0)
        uniform = model.forward_step(q, history[-3:], ablation="uniform")
        self.assertTrue(torch.isfinite(uniform).all())

    def test_adapter_parameters_receive_gradients(self):
        torch.manual_seed(17)
        model = SICTAdapter().train()
        q, history = torch.randn(91), torch.randn(12, 91)
        with torch.no_grad():
            model.output.weight.normal_()
        loss = model.forward_step(q, history).square().sum()
        loss.backward()
        self.assertIsNotNone(model.blocks[0].query.weight.grad)
        self.assertGreater(float(model.blocks[0].query.weight.grad.abs().sum()), 0.)

    def test_m2_hidden_api_preserves_logits(self):
        from wyu_bci_bench import nets
        import numpy as np

        rng = np.random.default_rng(27)
        features = rng.normal(size=(12, 1394)).astype(np.float32)
        labels = np.tile(np.arange(3), 4)
        model = nets.make_model(features, labels, "power_first_4bin").eval()
        raw = torch.randn(3, 41, 480)
        math = torch.randn(3, 1394)
        with torch.inference_mode():
            expected = model(raw, math)
            actual, hidden = model.forward_with_hidden(raw, math)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(hidden.shape, (3, 24))

    def test_sequence_is_causal_and_session_scoped(self):
        torch.manual_seed(23)
        model = SICTAdapter().eval()
        with torch.no_grad():
            model.output.weight.normal_()
        x = torch.randn(12, 91)
        streams = torch.tensor([0] * 6 + [1] * 6)
        base = model.forward_sequence(x, streams)
        changed = x.clone()
        changed[4:] += 10
        altered = model.forward_sequence(changed, streams)
        torch.testing.assert_close(base[:4], altered[:4], rtol=0, atol=0)
        # The second session has no access to the first session's tokens.
        isolated = model.forward_sequence(x[6:], torch.zeros(6, dtype=torch.long))
        torch.testing.assert_close(base[6:], isolated, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(model.forward_sequence(x, streams, ablation="no_memory"),
                                   torch.zeros(12, 3), rtol=0, atol=0)

    def test_sequence_matches_stepwise_streaming(self):
        torch.manual_seed(31)
        model = SICTAdapter().eval()
        with torch.no_grad():
            model.output.weight.normal_()
            model.output.bias.normal_()
        x = torch.randn(12, 91)
        streams = torch.tensor([0] * 7 + [1] * 5)
        with torch.inference_mode():
            batched = model.forward_sequence(x, streams)
            stepped = []
            for start, stop in ((0, 7), (7, 12)):
                state = SICTStream(model)
                stepped.extend(state.predict_one(q) for q in x[start:stop])
        torch.testing.assert_close(batched, torch.stack(stepped), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
