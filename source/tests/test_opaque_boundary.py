"""Falsify edge selection and the direction of the new contour gradient."""
import unittest

import torch

from utils.opaque_boundary import boundary_terms, pixel_coordinates, prepare_target, visible_segments


def rectangle(device='cpu'):
    height, width = 32, 48
    points = torch.tensor([[8.5, 7.5], [29.5, 7.5], [29.5, 24.5], [8.5, 24.5]], device=device)
    xy = (points + .5) * 2 / points.new_tensor([width, height]) - 1
    clip = torch.cat((xy, torch.zeros_like(xy[:, :1]), torch.ones_like(xy[:, :1])), dim=1)[None]
    faces = torch.tensor([[0,1,2],[0,2,3]], dtype=torch.int32, device=device)
    y, x = torch.meshgrid(torch.arange(height,device=device), torch.arange(width,device=device), indexing='ij')
    foreground = (x >= 9) & (x <= 29) & (y >= 8) & (y <= 24)
    face_one = ((x-8.5)*17 >= (y-7.5)*21)
    raster = torch.zeros((1,height,width,4),device=device)
    raster[0,:,:,3] = torch.where(foreground, torch.where(face_one, 1, 2), 0)
    target = ((x >= 11) & (x <= 31) & (y >= 8) & (y <= 24)).float()
    return clip,faces,raster,foreground,target


class BoundaryTests(unittest.TestCase):
    def test_shared_diagonal_is_not_a_silhouette(self):
        clip,faces,rast,_,_ = rectangle()
        edges=visible_segments(clip,faces,rast)
        self.assertGreater(len(edges), 10)
        q=pixel_coordinates(edges.mean(dim=1),32,48)
        on_outer=torch.stack([(q[:,0]-8.5).abs(),(q[:,0]-29.5).abs(),
                              (q[:,1]-7.5).abs(),(q[:,1]-24.5).abs()]).amin(dim=0)
        self.assertLess(float(on_outer.max()), 1e-4)

    def test_projected_gradient_points_toward_shifted_ground_truth(self):
        clip,faces,rast,foreground,target=rectangle()
        target=prepare_target(target)
        shift=torch.tensor(0., requires_grad=True)
        def loss(dx):
            shifted=clip+dx*clip.new_tensor([2/48,0,0,0])
            mesh=dict(verts_clip=shifted,mesh_faces=faces,rast=rast,alpha=foreground.float()[...,None])
            return boundary_terms(mesh,target,use_region=False)[1]
        loss(shift).backward()
        self.assertLess(float(shift.grad), 0.)
        with torch.no_grad():
            finite=float((loss(shift+.002)-loss(shift-.002))/.004)
        self.assertAlmostEqual(float(shift.grad),finite,delta=.003)

    def test_empty_geometry_and_nonforeground_masks(self):
        clip,faces,rast,foreground,target=rectangle()
        self.assertIsNone(prepare_target(torch.ones_like(target)))
        self.assertIsNone(prepare_target(torch.zeros_like(target)))
        mesh=dict(verts_clip=clip,mesh_faces=faces,rast=rast.zero_(),alpha=torch.zeros_like(foreground)[...,None].float())
        region,edge,count=boundary_terms(mesh,prepare_target(target))
        self.assertGreater(float(region),0.)
        self.assertEqual((float(edge),count),(0.,0))

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA rasterizer')
    def test_real_rasterizer_pixel_origin_and_gradient(self):
        import nvdiffrast.torch as dr
        clip,faces,_,_,target=rectangle('cuda')
        clip.requires_grad_()
        rast,_=dr.rasterize(dr.RasterizeCudaContext(),clip.contiguous(),faces.contiguous(),resolution=[32,48])
        a=(rast[...,3:] > 0).float()
        a=dr.antialias(a.contiguous(),rast,clip,faces)
        mesh=dict(verts_clip=clip,mesh_faces=faces,rast=rast,alpha=a[0])
        region,edges,count=boundary_terms(mesh,prepare_target(target))
        self.assertGreater(count,10)
        (region+edges).backward()
        self.assertTrue(torch.isfinite(clip.grad).all())
        self.assertLess(float(clip.grad[...,0].sum()),0.)


if __name__=='__main__':
    unittest.main()
