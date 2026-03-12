import numpy as np


def read_calib(calib_path):
    data = {}
    with open(calib_path, "r") as f:
        for line in f.readlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            nums = np.array([float(x) for x in value.strip().split()])
            data[key] = nums
    return data


def projection_to_matrix(P):
    """12 → 3x4"""
    return P.reshape(3, 4)


def tr_to_matrix(Tr):
    """12 → 4x4"""
    Tr = Tr.reshape(3, 4)
    T = np.eye(4)
    T[:3, :] = Tr
    return T


def main(calib_path):

    calib = read_calib(calib_path)

    # lidar -> cam0
    Tr = tr_to_matrix(calib["Tr"])

    print("==== Velodyne -> Cam0 ====")
    print(Tr)
    print()

    cameras = {}

    for i in range(4):

        P = projection_to_matrix(calib[f"P{i}"])

        # intrinsic
        K = P[:, :3]

        # Tx
        fx = K[0, 0]
        Tx = P[0, 3] / fx

        # cam0 -> cam_i
        Ti = np.eye(4)
        Ti[0, 3] = Tx

        # lidar -> cam_i
        T_cam_velo = Ti @ Tr

        R = T_cam_velo[:3, :3]
        t = T_cam_velo[:3, 3]

        cameras[i] = {
            "K": K,
            "T_cam_velo": T_cam_velo,
            "R": R,
            "t": t,
        }

    for i in range(4):

        print("===================================")
        print(f"Camera {i}")

        print("\nIntrinsic K:")
        print(cameras[i]["K"])

        print("\nR (velo -> cam):")
        print(cameras[i]["R"])

        print("\nt (velo -> cam):")
        print(cameras[i]["t"])
        print()


if __name__ == "__main__":
    calib_file = "/data/datasets/kitti_odometry/dataset/sequences/00/calib.txt"
    main(calib_file)