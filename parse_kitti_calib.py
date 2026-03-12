from pathlib import Path
import numpy as np
import re


def parse_kitti_calib(path: Path):
    data = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            v = v.strip()
            if not v:
                data[k] = np.array([])
                continue
            try:
                nums = np.array([float(x) for x in v.split()], dtype=float)
                data[k] = nums
            except ValueError:
                data[k] = v
    return data


def rt_to_T(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=float)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def reshape_if_present(calib, key, shape):
    if key not in calib:
        return None
    arr = calib[key]
    if arr.size != int(np.prod(shape)):
        return None
    return arr.reshape(shape)


def print_kitti_field_meanings():
    print("\n================ 字段含义说明（calib_cam_to_cam.txt） ================")
    print("S_xx      : 1x2 矫正前图像 xx 的大小（宽, 高）")
    print("K_xx      : 3x3 矫正前摄像机 xx 的校准矩阵（内参）")
    print("D_xx      : 1x5 矫正前摄像机 xx 的畸变参数（k1, k2, k3, p1, p2）")
    print("R_xx      : 3x3 外参旋转矩阵（从 cam0 到 cam_xx）")
    print("T_xx      : 3x1 外参平移向量（从 cam0 到 cam_xx）")
    print("S_rect_xx : 1x2 矫正后图像 xx 的大小")
    print("R_rect_xx : 3x3 纠正旋转矩阵（使图像平面共面）")
    print("P_rect_xx : 3x4 矫正后投影矩阵")
    print("=====================================================================")


def print_camera_calib(cam_cam):
    cam_ids = sorted(
        {
            m.group(1)
            for k in cam_cam.keys()
            for m in [re.match(r"^K_(\d{2})$", k)]
            if m
        }
    )

    print("\n================ 每个相机标定参数 ================")
    for cid in cam_ids:
        print(f"\n---------- 相机 {cid} ----------")
        for key, shape, desc in [
            (f"S_{cid}", (2,), "S_xx（矫正前大小）"),
            (f"K_{cid}", (3, 3), "K_xx（内参矩阵）"),
            (f"D_{cid}", (5,), "D_xx（畸变参数）"),
            (f"R_{cid}", (3, 3), "R_xx（cam0->cam_xx 旋转）"),
            (f"T_{cid}", (3,), "T_xx（cam0->cam_xx 平移）"),
            (f"S_rect_{cid}", (2,), "S_rect_xx（矫正后大小）"),
            (f"R_rect_{cid}", (3, 3), "R_rect_xx（纠正旋转）"),
            (f"P_rect_{cid}", (3, 4), "P_rect_xx（矫正后投影）"),
        ]:
            arr = reshape_if_present(cam_cam, key, shape)
            print(desc + ":")
            if arr is None:
                print("N/A")
            else:
                if shape == (5,):
                    print(arr.reshape(1, 5))
                elif shape == (3,):
                    print(arr.reshape(3, 1))
                else:
                    print(arr)
    return cam_ids


def compute_lidar_to_each_camera(velo_cam, cam_cam, cam_ids):
    """
    已知:
    T_v_c0: Velodyne -> cam0 (来自 calib_velo_to_cam)
    T_c0_cx: cam0 -> cam_xx (来自 R_xx, T_xx)
    T_rect_xx: cam_xx -> rectified_cam_xx (来自 R_rect_xx)

    计算:
    T_v_cx      = T_c0_cx @ T_v_c0
    T_v_cx_rect = T_rect_xx @ T_v_cx
    """
    R_v_c0 = velo_cam["R"].reshape(3, 3)
    t_v_c0 = velo_cam["T"].reshape(3)
    T_v_c0 = rt_to_T(R_v_c0, t_v_c0)

    print("\n================ Velodyne -> 各相机（未矫正 / 矫正后） ================")
    print("基准: Velodyne -> cam0")
    print("R_v_c0 =\n", R_v_c0)
    print("t_v_c0 =\n", t_v_c0.reshape(3, 1))

    for cid in cam_ids:
        R_c0_cx = reshape_if_present(cam_cam, f"R_{cid}", (3, 3))
        t_c0_cx = reshape_if_present(cam_cam, f"T_{cid}", (3,))
        R_rect_xx = reshape_if_present(cam_cam, f"R_rect_{cid}", (3, 3))

        if R_c0_cx is None or t_c0_cx is None:
            print(f"\n相机 {cid}: 缺少 R_{cid} 或 T_{cid}，跳过。")
            continue

        T_c0_cx = rt_to_T(R_c0_cx, t_c0_cx)
        T_v_cx = T_c0_cx @ T_v_c0

        print(f"\n---------- Velodyne -> cam{cid}（未矫正） ----------")
        print("R_v_cx =")
        print(T_v_cx[:3, :3])
        print("t_v_cx =")
        print(T_v_cx[:3, 3].reshape(3, 1))

        if R_rect_xx is None:
            print(f"相机 {cid}: 缺少 R_rect_{cid}，无法计算 LiDAR -> rectified_cam{cid}")
            continue

        T_rect_xx = np.eye(4, dtype=float)
        T_rect_xx[:3, :3] = R_rect_xx
        T_v_cx_rect = T_rect_xx @ T_v_cx

        print(f"---------- Velodyne -> rectified_cam{cid}（矫正后） ----------")
        print("R_v_cx_rect =")
        print(T_v_cx_rect[:3, :3])
        print("t_v_cx_rect =")
        print(T_v_cx_rect[:3, 3].reshape(3, 1))


def main(calib_dir="."):
    calib_dir = Path(calib_dir)

    velo_cam = parse_kitti_calib(calib_dir / "calib_velo_to_cam.txt")
    imu_velo = parse_kitti_calib(calib_dir / "calib_imu_to_velo.txt")
    cam_cam = parse_kitti_calib(calib_dir / "calib_cam_to_cam.txt")

    # LiDAR -> cam0
    R_vc0 = velo_cam["R"].reshape(3, 3)
    t_vc0 = velo_cam["T"].reshape(3)
    print("=============== LiDAR -> Camera0 外参 ===============")
    print("R_vc0 =\n", R_vc0)
    print("t_vc0 =\n", t_vc0.reshape(3, 1))

    # LiDAR -> IMU（由 IMU->LiDAR 取逆）
    T_iv = rt_to_T(imu_velo["R"].reshape(3, 3), imu_velo["T"].reshape(3))
    T_vi = invert_T(T_iv)
    print("\n=============== LiDAR -> IMU 外参 ===============")
    print("R_vi =\n", T_vi[:3, :3])
    print("t_vi =\n", T_vi[:3, 3].reshape(3, 1))

    print_kitti_field_meanings()
    cam_ids = print_camera_calib(cam_cam)
    compute_lidar_to_each_camera(velo_cam, cam_cam, cam_ids)


if __name__ == "__main__":
    main('/data/datasets/kitti/2011_10_03')