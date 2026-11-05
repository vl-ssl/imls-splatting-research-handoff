import unittest

from utils.mesh_domain import aligned_mesh_domain


class MeshDomainParityTests(unittest.TestCase):
    def test_no_growth_preserves_original(self):
        self.assertEqual(aligned_mesh_domain(256, 2.1, 2.1), (256, 2.1))

    def test_growth_keeps_even_lattice_aligned(self):
        old_resolution, old_scale = 256, 2.1
        new_resolution, new_scale = aligned_mesh_domain(
            old_resolution, old_scale, 2.305882353
        )
        self.assertEqual(new_resolution, 282)
        old_spacing = old_scale / (old_resolution - 1)
        self.assertAlmostEqual(new_scale / (new_resolution - 1), old_spacing)
        # Lower corner moves by an integer number of old voxels.
        self.assertAlmostEqual(
            (new_scale - old_scale) / (2 * old_spacing), 13.0
        )

    def test_odd_original_parity_is_preserved(self):
        resolution, scale = aligned_mesh_domain(81, 2.0, 2.11)
        self.assertEqual(resolution % 2, 1)
        self.assertGreaterEqual(scale, 2.11)

    def test_rejects_invalid_domain(self):
        with self.assertRaises(ValueError):
            aligned_mesh_domain(1, 2.1, 2.3)


if __name__ == '__main__':
    unittest.main()
