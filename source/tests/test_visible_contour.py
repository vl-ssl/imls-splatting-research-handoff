import unittest
import torch
from utils.visible_contour import symmetric_contour_distance, visible_contour_loss


class TestVisibleContour(unittest.TestCase):
    def test_disjoint_contours_have_correct_gradient(self):
        pred = torch.tensor([[0., 0.], [0., .2]], requires_grad=True)
        target = torch.tensor([[.5, 0.], [.5, .2]])
        loss = symmetric_contour_distance(pred, target)
        loss.backward()
        self.assertTrue(bool((pred.grad[:, 0] < 0).all()))
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_equal_contours_zero(self):
        p = torch.tensor([[0., 0.], [.1, .2]], requires_grad=True)
        self.assertLess(float(symmetric_contour_distance(p, p.detach())), 1e-7)

    def test_visibility_excludes_hidden_triangle(self):
        clip = torch.tensor([[[-.5, -.5, 0., 1.], [.5, -.5, 0., 1.],
                              [0., .5, 0., 1.], [0., 0., 1., 1.]]], requires_grad=True)
        faces = torch.tensor([[0, 1, 2], [1, 2, 3]])
        raster = torch.zeros(1, 8, 8, 4)
        raster[0, 2:6, 2:6, 3] = 1
        alpha = torch.zeros(1, 8, 8)
        alpha[:, 2:6, 3:7] = 1
        loss, n, _ = visible_contour_loss(clip, faces, raster, alpha)
        loss.backward()
        self.assertEqual(n, 3)
        self.assertEqual(float(clip.grad[0, 3].abs().sum()), 0.)

    def test_empty_render_is_finite(self):
        clip = torch.zeros(1, 3, 4, requires_grad=True)
        loss, n, _ = visible_contour_loss(clip, torch.tensor([[0, 1, 2]]),
                                         torch.zeros(1, 8, 8, 4), torch.ones(1, 8, 8))
        self.assertEqual(n, 0)
        self.assertEqual(float(loss), 0.)
        loss.backward()


if __name__ == '__main__':
    unittest.main()
