# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse
import atexit
import time
from pose_sender_kimm import PoseSender


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  parser.add_argument('--publish_pose', action='store_true', help='Send poses to the ROS2 UDP bridge')
  parser.add_argument('--udp_host', default='127.0.0.1')
  parser.add_argument('--udp_port', type=int, default=5005)
  parser.add_argument('--frame_id', default='camera_color_optical_frame')
  parser.add_argument('--max_frames', type=int, default=0, help='0: all frames; positive: limit for debugging')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  # requiring CAD model
  # Load once; the mesh is also used for rendering during every tracking step.
  mesh = trimesh.load(args.mesh_file)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  #TODO
  scorer = ScorePredictor()

  #TODO
  refiner = PoseRefinePredictor()

  glctx = dr.RasterizeCudaContext()

  est = FoundationPose(
    model_pts=mesh.vertices,
    model_normals=mesh.vertex_normals,
    mesh=mesh, scorer=scorer,
    refiner=refiner,
    debug_dir=debug_dir,
    debug=debug, glctx=glctx
  )

  logging.info("estimator initialization done")

  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
  sender = PoseSender(args.udp_host, args.udp_port, args.frame_id) if args.publish_pose else None
  if sender is not None:
    atexit.register(sender.close)

  for i in range(len(reader.color_files)):
    if args.max_frames > 0 and i >= args.max_frames:
      break
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    # Dataset has no capture timestamps here: use replay frame-read wall time.
    timestamp_ns = time.time_ns()
    if i==0:

      # Without a saved mask, use an instance segmentation model such as Mask R-CNN.
      mask = reader.get_mask(0).astype(bool)

      # very first step pose
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
      if est.pose_last is None:
        raise RuntimeError('Registration failed: check target mask and valid depth')

      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)

    else:
      # after first step
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

    # Both register and track_one return the original mesh-to-camera 4x4 pose.
    if sender is not None:
      sender.send(pose, reader.id_strs[i], timestamp_ns)

    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      cv2.imshow('1', vis[...,::-1])
      cv2.waitKey(1)

    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)
