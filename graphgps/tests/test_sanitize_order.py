import unittest


class TestSanitizeOrder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import may fail if optional deps (performer_pytorch, mamba_ssm) are not installed.
        try:
            from graphgps.layer.gps_layer import GPSLayer  # noqa: F401
            cls.GPSLayer = GPSLayer
        except Exception as e:  # pragma: no cover
            cls.GPSLayer = None
            cls._import_exc = e

    def setUp(self):
        if self.GPSLayer is None:
            self.skipTest(f"Optional dependency missing for importing GPSLayer: {self._import_exc}")

    def _layer(self):
        # Avoid running GPSLayer.__init__ (would construct heavy modules).
        return self.GPSLayer.__new__(self.GPSLayer)

    def assertIsPermutation(self, perm, n):
        import torch

        self.assertEqual(perm.dtype, torch.long)
        self.assertEqual(perm.numel(), n)
        self.assertTrue(bool(((perm >= 0) & (perm < n)).all()))
        # uniqueness
        self.assertEqual(int(torch.unique(perm).numel()), n)

    def test_sanitize_node_order_valid(self):
        import torch

        layer = self._layer()
        n = 6
        node_order = torch.tensor([2, 1, 0, 3, 5, 4], dtype=torch.long)
        out = layer._sanitize_node_order(node_order, n, device=node_order.device)
        self.assertTrue(torch.equal(out, node_order))

    def test_sanitize_node_order_invalid_values_and_duplicates(self):
        import torch

        layer = self._layer()
        n = 5
        node_order = torch.tensor([0, 1, 1, -1, 7], dtype=torch.long)
        out = layer._sanitize_node_order(node_order, n, device=node_order.device)
        self.assertIsPermutation(out, n)

    def test_sanitize_node_order_wrong_length(self):
        import torch

        layer = self._layer()
        n = 4
        node_order = torch.tensor([3, 2], dtype=torch.long)
        out = layer._sanitize_node_order(node_order, n, device=node_order.device)
        self.assertIsPermutation(out, n)

    def test_sanitize_edge_order_invalid(self):
        import torch

        layer = self._layer()
        m = 4
        edge_order = torch.tensor([0, 0, 10, -2], dtype=torch.long)
        out = layer._sanitize_edge_order(edge_order, m, device=edge_order.device)
        self.assertEqual(out.numel(), m)
        self.assertTrue(bool(((out >= 0) & (out < m)).all()))
        self.assertEqual(int(torch.unique(out).numel()), m)

    def test_cuda_path_does_not_assert(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        layer = self._layer()
        n = 8
        node_order = torch.tensor([0, 1, 1, -1, 100], dtype=torch.long, device="cuda")
        out = layer._sanitize_node_order(node_order, n, device=node_order.device)
        self.assertIsPermutation(out, n)


if __name__ == "__main__":
    unittest.main()

