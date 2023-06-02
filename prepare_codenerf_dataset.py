import argparse
import os
import sys
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

import numpy as np
import cv2
import pyrender
import open3d as o3d
import trimesh
import gibson2

from models.pano3d.dataloader import IGSceneDataset
from utils.igibson_utils import IGScene
from utils.mesh_utils import MeshIO
from utils.image_utils import save_image


def get_pyrender_renderer(width, height, K, trimesh_mesh):
    cam = pyrender.IntrinsicsCamera(
        fx=K[0, 0], fy=K[1, 1],
        cx=K[0, 2], cy=K[1, 2]
    )

    ambient_l = np.array([0.35, 0.35, 0.35, 1.0])

    if isinstance(trimesh_mesh, trimesh.scene.scene.Scene):
        scene = pyrender.Scene.from_trimesh_scene(
            trimesh_mesh, ambient_light=ambient_l
        )
        print(mesh_path, 'returns scene')
    else:
        pyrender_mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)

        scene = pyrender.Scene(
            ambient_light=ambient_l
        )
        scene.add(pyrender_mesh)

    cam_node = scene.add(cam)

    r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

    return {'renderer': r, 'scene': scene, 'cam_node': cam_node}


def pyrender_mesh(renderer, T_world_cam):
    # use OpenGL convention (East-Up-South) instead of East-Down-North
    rotation_correction = np.asarray([
        [1, 0, 0], [0, -1, 0], [0, 0, -1]
    ])
    render_flags = pyrender.constants.RenderFlags.RGBA # | pyrender.constants.RenderFlags.ALL_WIREFRAME

    T_world_cam = T_world_cam.copy()
    T_world_cam[:3, :3] = T_world_cam[:3, :3] @ rotation_correction

    renderer['scene'].set_pose(renderer['cam_node'], T_world_cam)

    lights = [
        pyrender.DirectionalLight(color=np.ones(3), intensity=5.0),
        pyrender.SpotLight(
            color=np.ones(3), intensity=5.0,
            innerConeAngle=np.pi/16*0.1, outerConeAngle=np.pi/6*0.1
        ),
        pyrender.PointLight(color=np.ones(3), intensity=2.0)
    ]

    nodes = []
    for light in lights:
        nodes.append(renderer['scene'].add(light, pose=T_world_cam))

    rendered_color, rendered_depth = renderer['renderer'].render(
        renderer['scene'], flags=render_flags
    )

    for node in nodes:
        renderer['scene'].remove_node(node)

    return rendered_color, rendered_depth


def get_perspective_camera(equ_h, equ_w, FOV, THETA, PHI, height, width):
    #
    # THETA is left/right angle, PHI is up/down angle, both in degree
    #
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * THETA)
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * PHI)
    R_equ_persp = R2 @ R1

    f = 0.5 * width * 1 / np.tan(0.5 * FOV)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([
	[f, 0, cx],
	[0, f, cy],
	[0, 0,  1],
    ], np.float32)

    return R_equ_persp, K


def test_gibson():
    import pybullet as p
    from xml.etree import ElementTree as ET
    from shapely.geometry import Polygon, Point, MultiPoint
    import shapely

    from gibson2.simulator import Simulator
    from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
    from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
    from gibson2.utils.assets_utils import get_ig_scene_path, get_cubicasa_scene_path, get_3dfront_scene_path
    from gibson2.utils.mesh_util import quat2rotmat, xyzw2wxyz

    from configs.data_config import IG56CLASSES
    from utils.relation_utils import RelationOptimization
    from utils.render_utils import seg2obj, render_camera, is_obj_valid, hdr_texture, hdr_texture2, background_texture
    from utils.igibson_utils import IGScene, hash_split
    from utils.image_utils import ImageIO
    from utils.layout_utils import scene_layout_from_mesh, room_layout_from_scene_layout, \
        manhattan_pix_layout_from_room_layout, cuboid_world_layout_from_room_layout, \
        manhattan_world_layout_from_room_layout, horizon_layout_gt_from_scene_data
    from utils.transform_utils import bdb3d_corners, IGTransform
    from utils.basic_utils import write_json, read_pkl, write_pkl

    dataset_path = gibson2.ig_dataset_path
    perspective = False

    parser = argparse.ArgumentParser(
        description='Render RGB panorama from iGibson scenes.')
    parser.add_argument('--scene', dest='scene_name',
                        type=str, default=None,
                        help='The name of the scene to load')
    parser.add_argument('--source', dest='scene_source',
                        type=str, default='IG',
                        help='The name of the source dataset, among [IG,CUBICASA,THREEDFRONT]')
    parser.add_argument('--output', type=str, default='data/igibson',
                        help='The path of the output folder')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for generating camera pose')
    parser.add_argument('--width', type=int, default=512,
                        help='Height of output image')
    parser.add_argument('--height', type=int, default=512,
                        help='Height of output image')
    parser.add_argument('--processes', type=int, default=0,
                        help='Number of threads')
    parser.add_argument('--renders', type=int, default=10,
                        help='Number of renders per room')
    parser.add_argument('--cam_height', type=float, default=[1.6], nargs='+',
                        help='Height of camera in meters (provide two numbers to specify range)')
    parser.add_argument('--cam_pitch', type=float, default=[0.], nargs='+',
                        help='Pitch of camera in degrees (provide two numbers to specify range)')
    parser.add_argument('--random_yaw', default=False, action='store_true',
                        help='Randomize camera yaw')
    parser.add_argument('--vertical_fov', type=float, default=None,
                        help='Fov for perspective camera in degrees')
    parser.add_argument('--render_type', type=str, default=['rgb', 'seg', 'sem', 'depth'], nargs='+',
                        help='Types of renders (rgb/normal/seg/sem/depth/3d)')
    parser.add_argument('--strict', default=False, action='store_true',
                        help='Raise exception if render fails')
    parser.add_argument('--super_sample', type=int, default=2,
                        help='Set to greater than 1 to use super_sample')
    parser.add_argument('--no_physim', default=False, action='store_true',
                        help='Do physical simulation before rendering')
    parser.add_argument('--train', type=float, default=0.7,
                        help='Ratio of train split')
    parser.add_argument('--horizon_lo', default=False, action='store_true',
                        help='Generate Horizon format layout GT from manhattan layout')
    parser.add_argument('--json', default=False, action='store_true',
                        help='Save camera info as json too')
    parser.add_argument('--cuboid_lo', default=False, action='store_true',
                        help='Generate cuboid world frame layout from manhattan layout')
    parser.add_argument('--world_lo', default=False, action='store_true',
                        help='Generate manhatton world frame layout')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='ID of GPU used for rendering')
    parser.add_argument('--split', default=False, action='store_true',
                        help='Split train/test dataset without rendering')
    parser.add_argument('--random_obj', default=None, action='store_true',
                        help='Use the 10 objects randomization for each scene')
    parser.add_argument('--resume', default=False, action='store_true',
                        help='Resume from existing renders')
    parser.add_argument('--expand_dis', type=float, default=0.1,
                        help='Distance of bdb3d expansion when generating collision and touch relation '
                             'between objects, walls, floor and ceiling')
    parser.add_argument('--crop_width', default=None, type=int,
                        help='Width of image cropped of ground truth 2d bounding box')
    parser.add_argument('--relation', default=False, action='store_true',
                        help='Generate relationships')
    args = parser.parse_args()
    args_dict = args.__dict__.copy()
    args_dict = {
        **args_dict,
        'random_obj': 0,
        'scene_name': 'Beechwood_0_int',
        'vertical_fov': 90,
    }
    args = argparse.Namespace(**args_dict)

    scene_name, scene_source = args.scene_name, args.scene_source
    scene_dir = get_ig_scene_path(args.scene_name)
    camera = Path(f'data/igibson/{args.scene_name}/000000/data.pkl')

    light_modulation_map_filename = os.path.join(
        scene_dir, 'layout', 'floor_lighttype_0.png'
    )
    scene = InteractiveIndoorScene(
        args.scene_name,
        texture_randomization=False,
        object_randomization=args.random_obj is not None,
        object_randomization_idx=args.random_obj,
        scene_source=args.scene_source
    )
    settings = MeshRendererSettings(
        env_texture_filename=hdr_texture,
        env_texture_filename2=hdr_texture2,
        env_texture_filename3=background_texture,
        light_modulation_map_filename=light_modulation_map_filename,
        enable_shadow=True, msaa=False, enable_pbr=True
    )
    scene_layout = scene_layout_from_mesh(args)
    if not scene_layout:
        raise Exception('Layout not valid!')

    vertical_fov = args.vertical_fov
    render_width = args.height
    output_width = args.height * 2

    output_height = args.height
    render_height = args.height * args.super_sample
    render_width *= args.super_sample
    s = Simulator(mode='headless', image_width=render_width, image_height=render_height,
                  vertical_fov=vertical_fov, device_idx=args.gpu_id, rendering_settings=settings)

    # convert floor_trav pickle file to protocol-4 to avoid error
    floor_trav_path = os.path.join(scene_dir, 'layout', 'floor_trav_0.p')
    if os.path.exists(floor_trav_path):
        try:
            read_pkl(floor_trav_path)
        except ValueError:
            print(f"floor_trav pickle file {floor_trav_path} is not compatible, converting...")
            floor_trav = read_pkl(floor_trav_path, protocol=5)
            write_pkl(floor_trav, floor_trav_path)

    # import scene and run physical simulation
    try:
        s.import_ig_scene(scene)
    except Exception as err:
        s.disconnect()
        raise err
    if not args.no_physim:
        for i in range(200):
            s.step()

    # get scene object info
    is_fixed = {} # if the link type is fixed or floating
    urdf_files = {} # temp URDF files of each sub object
    obj_ids = list(scene.objects_by_id.keys())
    i_obj = 0
    obj_groups = {} # main object and the sub objects of object groups
    while i_obj < len(obj_ids):
        obj_id = obj_ids[i_obj]
        urdf_object = scene.objects_by_id[obj_id]
        for i_subobj, (fixed, urdf_file) in enumerate(zip(urdf_object.is_fixed, urdf_object.urdf_paths)):
            is_fixed[obj_id + i_subobj] = fixed
            urdf_files[obj_id + i_subobj] = urdf_file
        obj_group = urdf_object.body_ids.copy()
        if len(obj_group) > 1:
            # treat the object with the greatest mass as main object
            mass_list = []
            for i_subobj in obj_group:
                obj_tree = ET.parse(urdf_files[i_subobj])
                mass_list.append(float(obj_tree.find("link").find("inertial").find('mass').attrib['value']))
            main_object = obj_group[np.argmax(mass_list)]
            obj_group.remove(main_object)
            obj_groups[main_object] = obj_group
        i_obj += len(urdf_object.body_ids)

    # get object params
    objs = {}
    for obj_id in range(len(scene.objects_by_id)):
        # get object info
        obj = scene.objects_by_id[obj_id]
        if getattr(obj, 'bounding_box', None) is None:
            continue
        obj_dict = {
            'classname': obj.category,
            'label': IG56CLASSES.index(obj.category),
            'model_path': os.path.join(*obj.model_path.split('/')[-2:]),
            'is_fixed': is_fixed[obj_id],
        }

        # get object bdb3d
        if is_fixed[obj_id]:
            orn = p.getLinkState(obj_id, 0)[-1]
            aabb = p.getAABB(obj_id, 0)
        else:
            _, orn = p.getBasePositionAndOrientation(obj_id)
            aabb = p.getAABB(obj_id, -1)

        # use axis aligned bounding box center of first link as bounding box center
        centroid = np.mean(aabb, axis=0)
        basis = quat2rotmat(xyzw2wxyz(orn))[:3, :3]
        obj_dict['bdb3d'] = {
            'centroid': centroid.astype(np.float32),
            'basis': basis.astype(np.float32),
            'size': obj.bounding_box.astype(np.float32)
        }
        objs[obj_id] = obj_dict

    # get object layout
    object_layout = []
    for obj in objs.values():
        corners = bdb3d_corners(obj['bdb3d'])
        corners2d = corners[(0, 1, 3, 2), :2]
        obj2d = Polygon(corners2d)
        object_layout.append(obj2d)
    object_layout = shapely.ops.cascaded_union(object_layout)
    # plot_layout(object_layout)

    # render random camera, get and save GT
    np.random.seed(args.seed)
    camera_paths = []
    i_camera = 0
    while i_camera < args.renders:
        # randomize camera position
        _, (px, py, pz) = scene.get_random_point()
        if len(args.cam_height) == 1:
            pz = args.cam_height[0]
        else:
            pz = np.random.random() * (args.cam_height[1] - args.cam_height[0]) + args.cam_height[0]
        camera_pos = np.array([px, py, pz], dtype=np.float32)

        # generate room layout by camera position
        camera_name = i_camera + args.random_obj * args.renders if args.random_obj is not None else i_camera
        camera = {
            'pos': camera_pos,
            'height': output_height,
            'width': output_width
        }
        data = {
            'name': f"{camera_name:05d}",
            'scene': scene_name,
            'room': scene.get_room_instance_by_point(camera_pos[:2]),
            'camera': camera
        }
        skip_info = f"Skipped camera {data['name']} of {data['scene']}: "
        if data['room'] is None:
            print(skip_info + "room is 'None'")
            continue
        room_layout = room_layout_from_scene_layout(camera, scene_layout)
        if room_layout is None:
            print(skip_info + "room layout generation failed")
            continue

        # randomize camera target
        if args.random_yaw:
            if perspective:
                yaw = np.random.random() * 2 * np.pi
            else:
                yaw = np.random.randint(4) * np.pi / 2
        else:
            yaw = np.pi / 2  # default align to positive direction of axis x
        if len(args.cam_pitch) == 1:
            pitch = args.cam_pitch[0]
        else:
            pitch = np.random.random() * (args.cam_pitch[1] - args.cam_pitch[0]) + args.cam_pitch[0]
        pitch = np.deg2rad(pitch)
        camera_target = np.array([px + np.sin(yaw), py + np.cos(yaw), pz + np.tan(pitch)], dtype=np.float32)
        camera["target"] = camera_target
        camera["up"] = np.array([0, 0, 1], dtype=np.float32)
        if perspective:
            camera['K'] = s.renderer.get_intrinsics().astype(np.float32) / 2

        # generate camera layout and check if the camaera is valid
        layout = {'manhattan_pix': manhattan_pix_layout_from_room_layout(camera, room_layout)}
        data['layout'] = layout
        if layout['manhattan_pix'] is None:
            print(skip_info + "manhattan pixel layout generation failed")
            continue
        if args.cuboid_lo:
            layout['cuboid_world'] = cuboid_world_layout_from_room_layout(room_layout)
        if args.world_lo:
            layout['manhattan_world'] = manhattan_world_layout_from_room_layout(room_layout)
        if args.horizon_lo:
            layout['horizon'] = horizon_layout_gt_from_scene_data(data)

        # filter out camera by object layout
        camera_point = Point(*camera['pos'][:2])
        if any(obj.contains(camera_point) for obj in object_layout):
            print(skip_info + "inside or above/below obj")
            continue
        nearest_point, _ = shapely.ops.nearest_points(object_layout.boundary, camera_point)
        distance_obj = camera_point.distance(nearest_point)
        # if distance_obj < 0.5:
        #     print(f"{skip_info}too close ({distance_obj:.3f} < 0.5) to object")
        #     continue

        # render
        render_results = render_camera(s.renderer, camera, args.render_type,
                                       perspective, obj_groups, scene.objects_by_id)

        # extract object params
        if 'seg' in args.render_type:
            data['objs'] = []
            ids = np.unique(render_results['seg']).astype(np.int).tolist()

            for obj_id in ids:
                if obj_id not in objs.keys():
                    continue
                obj_dict = objs[obj_id].copy()
                obj_dict['id'] = obj_id

                # get object bdb2d
                obj_dict.update(seg2obj(render_results['seg'], obj_id))
                if not is_obj_valid(obj_dict):
                    continue

                # rotate camera to recenter bdb3d
                recentered_trans = IGTransform.level_look_at(data, obj_dict['bdb3d']['centroid'])
                corners = recentered_trans.world2campix(bdb3d_corners(obj_dict['bdb3d']))
                full_convex = MultiPoint(corners).convex_hull
                # pyplot.plot(*full_convex.exterior.xy)
                # pyplot.axis('equal')
                # pyplot.show()

                # filter out objects by ratio of visible part
                contour = obj_dict['contour']
                contour_points = np.stack([contour['x'], contour['y']]).T
                visible_convex = MultiPoint(contour_points).convex_hull
                if visible_convex.area / full_convex.area < 0.2:
                    continue

                data['objs'].append(obj_dict)

            if not data['objs']:
                print(f"{skip_info}no object in the frame")
                continue

        # construction IGScene
        ig_scene = IGScene(data)

        # generate relation
        if args.relation:
            relation_optimization = RelationOptimization(expand_dis=args.expand_dis)
            relation_optimization.generate_relation(ig_scene)


def get_scaled_pose(obj, object_path):
    T_world_cam = obj['T_w_persp']

    T_world_obj = np.eye(4)
    T_world_obj[:3, :3] = obj['bdb3d']['basis']
    T_world_obj[:3, -1] = obj['bdb3d']['centroid']
    T_obj_world = np.linalg.inv(T_world_obj)

    # In igibson object front is -Y, in SRN it's +Y
    rvec = np.array([0., 0., np.pi])
    R = cv2.Rodrigues(rvec)[0]
    T = np.eye(4)
    T[:3, :3] = R
    T_obj_world = T @ T_obj_world

    T_obj_cam = T_obj_world @ T_world_cam

    # make it fit in unit cube
    # (based on https://github.com/vsitzmann/shapenet_renderer/issues/1#issuecomment-821186408)
    # scale = 1.0 / np.max(obj['bdb3d']['size']) / np.linalg.norm(obj['bdb3d']['size'])
    scale = 1.0 / np.max(obj['bdb3d']['size'])
    T_obj_cam[:3, 3] *= scale

    # obj pcd in object (canonical) coordinate frame (centered at object)
    obj_mesh = deepcopy(obj['merged_mesh'])
    obj_mesh = obj_mesh.apply_transform(T_obj_world)
    obj_mesh = obj_mesh.apply_scale(scale)
    obj_mesh.export(str(object_path / 'mesh_scaled.obj'))

    return T_obj_cam


def get_camera_pose(eye, center, up):
    """adapted from trescope.blender.blender_front3d.setCamera
    z-axis points forward (unlike the orignal/OpenGL convention)
    """
    eye = np.array(list(eye))
    center = np.array(list(center))
    north = np.array(list(up))
    direction = center - eye
    forward = direction / np.linalg.norm(direction)
    right = np.cross(-north, forward)
    up = np.cross(forward, right)
    rotation = np.vstack([right, up, forward]).T
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, -1] = eye

    return matrix


def generate_test_poses(T_obj_cam):
    xyz = T_obj_cam[:3, 3]
    r = np.linalg.norm(xyz)
    phi = np.arccos(xyz[2] / r)

    def spherical_to_cartesian(r, theta, phi):
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return [x, y, z]

    poses = []
    thetas = np.linspace(0, 2 * np.pi, num=49, endpoint=False)
    for theta in thetas:
        xyz = spherical_to_cartesian(r, theta, phi)
        poses.append(get_camera_pose(xyz, [0, 0, 0], [0, 0, 1]))

    for i in range(4):
        phi += np.deg2rad(7)
        thetas = np.linspace(0, 2 * np.pi, num=50, endpoint=False)
        for theta in thetas:
            xyz = spherical_to_cartesian(r, theta, phi)
            poses.append(get_camera_pose(xyz, [0, 0, 0], [0, 0, 1]))

    return poses


def save_srn_data(obj, dataset_path):
    scene_name = obj['scene_name'] + '_' + str(obj['scene_idx'])
    obj_name = '_'.join(obj['model_path'].split('/')) + '_' + str(obj['id'])

    object_path = (
        Path(dataset_path).parent / 'srn_chairs/chairs_test' / f'{scene_name}_{obj_name}'
    )
    object_path.mkdir(parents=True, exist_ok=True)

    K = obj['K']
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    h, w = obj['rgb'].shape[:2]
    with open(object_path / 'intrinsics.txt', 'w') as f:
        f.write(
            f'{fx} {cx} {cy} 0.000000\n'
            f'0. 0. 0.\n'
            f'1.\n'
            f'{h} {w}'
        )

    intrinsics_path = object_path / 'intrinsics'
    intrinsics_path.mkdir(exist_ok=True)
    np.savetxt(intrinsics_path / '000000.txt', K.reshape((1, -1)))

    pose_path = object_path / 'pose'
    pose_path.mkdir(exist_ok=True)
    T_obj_cam = get_scaled_pose(obj, object_path)
    np.savetxt(pose_path / '000000.txt', T_obj_cam.reshape((1, -1)))

    rgb = obj['rgb']
    mask = obj['seg']
    # rgb = (rgb.astype(float) * 0.6).astype(np.uint8)
    rgb[mask < 200] = 255
    rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGRA)
    rgba[..., -1] = 255 # (mask * 255).astype(np.uint8)

    rgb_path = object_path / 'rgb'
    rgb_path.mkdir(exist_ok=True)
    cv2.imwrite(str(rgb_path / '000000.png'), rgba)

    test_poses = generate_test_poses(T_obj_cam)
    rgba_dummy = np.ones_like(rgba) * 255
    for i, pose in enumerate(test_poses, 1):
        np.savetxt(pose_path / f'{i:06d}.txt', pose.reshape((1, -1)))
        np.savetxt(intrinsics_path / f'{i:06d}.txt', K.reshape((1, -1)))
        cv2.imwrite(str(rgb_path / f'{i:06d}.png'), rgba_dummy)

    o3d.io.write_triangle_mesh(
        str(object_path / 'pose.ply'),
        o3d.geometry.TriangleMesh.create_coordinate_frame()
            .scale(0.25, [0.]*3).transform(T_obj_cam)
    )

    # test_poses_mesh = o3d.geometry.TriangleMesh()
    # for pose in test_poses:
    #     test_poses_mesh += (
    #         o3d.geometry.TriangleMesh.create_coordinate_frame()
    #         .scale(0.25, [0.]*3).transform(pose)
    #     )
    # o3d.io.write_triangle_mesh(
    #     str(object_path / 'test_poses.ply'), test_poses_mesh
    # )


def get_train_objects(dataset):
    train_dir = '/home/gruvi-3dv/workspace/datasets/ShapeNet-SRN/srn_chairs/chairs_train/chairs_2.0_train'
    return [i.name for i in Path(train_dir) .iterdir() if i.is_dir()]


def get_test_objects(dataset):
    test_dir = '/home/gruvi-3dv/workspace/datasets/ShapeNet-SRN/srn_chairs/chairs_test'
    return [i.name for i in Path(test_dir) .iterdir() if i.is_dir()]


if __name__ == '__main__':
    dataset_path = str((Path(__file__).parent / 'data/igibson').resolve())

    dataset = IGSceneDataset({'data': {'split': dataset_path}})
    crop_types = ('rgb', 'seg')
    train_objects = get_train_objects(dataset)
    train_objects = {i: True for i in train_objects}

    test_objects = get_test_objects(dataset)
    test_objects = {i: True for i in test_objects}

    for split in dataset.split:
        scene = IGScene.from_pickle(split)
        scene.crop_images(
            perspective=True, short_width=128, crop_types=crop_types
        )

        Twc = scene['camera']['cam3d2world']
        o3d.io.write_triangle_mesh(
            '/tmp/equ_cam.ply',
            o3d.geometry.TriangleMesh.create_coordinate_frame()
                .scale(0.25, [0.]*3).transform(Twc)
        )

        for obj_idx in range(len(scene['objs'])): # [0, 1]: #
            obj = scene['objs'][obj_idx]
            bdb3d = obj['bdb3d']
            obj_name = '_'.join(obj['model_path'].split('/'))

            if (
                scene['objs'][obj_idx]['classname'] != 'chair'
                # or obj['model_path'].split('/')[-1] in train_objects
                or obj['model_path'].split('/')[-1] not in test_objects
            ):
                continue

            print('Processing', obj_name)

            origin = np.zeros(3, dtype=np.float32)
            centroid = scene.transform.obj2frame(origin, bdb3d)

            obj_path = (
                Path('data/data/ig_dataset/objects') /
                Path(scene['objs'][obj_idx]['model_path']) /
                'shape' / 'visual'
            )
            obj_list = list(obj_path.glob('*.obj'))
            merged_mesh = MeshIO.from_file(obj_list).load().merge()
            v = np.array(merged_mesh.vertices, dtype=np.float32)
            min_v = np.min(v, axis=0)
            max_v = np.max(v, axis=0)
            obj_bbox = max_v - min_v
            obj_centroid = np.mean([max_v, min_v], axis=0)
            # scale = 1. / float(max(obj_bbox))

            shift_mat = np.eye(4)
            shift_mat[:3, -1] = -obj_centroid

            scale_mat = np.eye(4)
            for i in range(3):
                scale_mat[i, i] = bdb3d['size'][i] / obj_bbox[i]

            T_world_obj = np.eye(4)
            T_world_obj[:3, :3] = bdb3d['basis']
            T_world_obj[:3, -1] = bdb3d['centroid'] # centroid #

            merged_mesh.apply_transform(T_world_obj @ scale_mat @ shift_mat)

            for key in crop_types:
                crop = obj[key]
                save_image(crop, Path('/tmp') / f'{obj_name}_{key}_render.png')

            bfov = obj['bfov']
            R_equ_persp, K = get_perspective_camera(
                *(scene.image_io['rgb'].shape[:2]),
                bfov['x_fov'], bfov['lon'], -bfov['lat'],
                *(obj['rgb'].shape[:2])
            )

            T_w_equ = scene['camera']['cam3d2world']
            T_w_persp = T_w_equ.copy()
            T_w_persp[:3, :3] = T_w_equ[:3, :3] @ R_equ_persp

            obj['K'] = K
            obj['T_w_persp'] = T_w_persp
            obj['obj_centroid'] = obj_centroid
            obj['merged_mesh'] = merged_mesh
            obj['scene_name'] = scene.data['scene']
            obj['scene_idx'] = scene.data['name']

            save_srn_data(obj, dataset_path)

            height, width = obj['rgb'].shape[:2]
            renderer = get_pyrender_renderer(width, height, K, merged_mesh)
            img, depth = pyrender_mesh(renderer, T_w_persp)
            mask = depth > 1e-3

            crop = obj['rgb']
            crop[depth > 1e-3, 0] = 212

            cv2.imwrite(f'/tmp/{obj_name}_masked.png', crop[..., ::-1])
            cv2.imwrite(f'/tmp/{obj_name}_rerendered.png', img)

            merged_mesh.export(f'/tmp/{obj_name}.ply')

            o3d.io.write_triangle_mesh(
                f'/tmp/{obj_name}_pose.ply',
                o3d.geometry.TriangleMesh.create_coordinate_frame()
                    .scale(0.25, [0.]*3).transform(T_world_obj)
            )

            # o3d.io.write_triangle_mesh(
            #     f'/tmp/{obj_name}_cam.ply',
            #     o3d.geometry.TriangleMesh.create_coordinate_frame()
            #         .scale(0.25, [0.]*3).transform(T_w_persp)
            # )
