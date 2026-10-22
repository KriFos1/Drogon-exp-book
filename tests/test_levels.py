import unittest

import numpy as np

from drogon_levels import cell_corners, facies_probabilities, grandy_volume


def box_geometry(nx, ny, nz, top=1000.0, thickness=10.0):
    """A vertical-pillar box grid of 100 m x 100 m cells."""
    x, y = np.meshgrid(np.arange(nx + 1) * 100.0, np.arange(ny + 1) * 100.0)
    coord = np.zeros((ny + 1, nx + 1, 2, 3))
    coord[..., 0, 0] = coord[..., 1, 0] = x
    coord[..., 0, 1] = coord[..., 1, 1] = y
    coord[..., 0, 2], coord[..., 1, 2] = top, top + nz * thickness
    zcorn = np.zeros((nz, 2, ny, 2, nx, 2))
    for k in range(nz):
        zcorn[k, 0] = top + k * thickness
        zcorn[k, 1] = top + (k + 1) * thickness
    return coord, zcorn


class LevelHelpersTest(unittest.TestCase):
    def test_box_cells_have_their_geometric_volume(self):
        cells = cell_corners(*box_geometry(2, 3, 2))
        self.assertEqual(cells.shape, (12, 8, 3))
        np.testing.assert_allclose([grandy_volume(cell) for cell in cells], 100.0 * 100.0 * 10.0)

    def test_facies_probabilities_follow_the_book_truncation_groups(self):
        satnum = np.array([1, 3, 12, 4])
        active = np.array([True, True, True, False])
        fields = facies_probabilities(satnum, active)
        total = sum(fields[f"p{value}"] for value in range(1, 13))
        np.testing.assert_allclose(total[active], 1.0)
        self.assertEqual(fields["p1"][0], 0.8)
        self.assertEqual(fields["p2"][0], 0.05)
        self.assertEqual(fields["p3"][0], 0.0)
        self.assertTrue(all(fields[f"p{value}"][3] == -1.0 for value in range(1, 13)))


if __name__ == "__main__":
    unittest.main()
