import tf
import numpy as np

import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            FancyArrowPatch.draw(self, renderer)


def plot_3d_scatter(x,y,z,m):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Range')
    ax.set_ylabel('Swath')
    ax.set_zlabel('Depth')
    ax.set_xlim3d(0, np.max(x)+1)
    ax.set_ylim3d(np.min(y)-1, np.max(y)+1)
    ax.set_zlim3d(np.max(z)+1, 0)
    ax.scatter(x,y,z, marker=m)

def move_pose(pose, x, y, z, roll, pitch, yaw):
    trans = np.array([[x],[y],[z]])
    rot = tf.transformations.euler_matrix(roll, pitch, yaw)
    rot[:-1,-1:] = trans

    return pose.dot(rot)

def cart_to_polar(cart):
    return np.array([[np.arctan2(cart[1,0],cart[0,0])],
                    [np.sqrt(cart[0,0]**2 + cart[1,0]**2 + cart[2,0]**2)]])

def update_landmarks(landmarks, new_pose):
    relative_pose = np.linalg.inv(landmarks[0].Xa).dot(new_pose)
    for l in landmarks:
        l.cart_img2 = project_landmark(l.cart_img1, relative_pose)
        l.polar_img2 = cart_to_polar(l.cart_img2)
        l.rel_pose = relative_pose
        l.Xb = new_pose
        l.real_phi = np.arcsin(l.cart_img1[2,0]/l.polar_img1[1,0])
        l.update_phi(l.real_phi)

    return landmarks

def project_landmark(coords, relative_pose):
    rot = relative_pose[:-1,:-1]
    trans = relative_pose[:-1,-1:]

    return rot.transpose().dot(coords-trans)

def save_landmarks(landmarks, filename):
    with open(filename, "wb") as fp:
        pickle.dump(landmarks, fp)
    print("Landmarks saved as {}".format(filename))

def load_landmarks(filename):
    with open(filename, "rb") as fp:
        landmarks = pickle.load(fp)

    return landmarks

def polar_to_cart(polar):
    return np.array([[polar[1,0] * np.cos(polar[0,0]) * np.cos(polar[2,0])],
                     [polar[1,0] * np.sin(polar[0,0]) * np.cos(polar[2,0])],
                     [polar[1,0] * np.sin(polar[2,0])]])

def plot_svd(values):
    plt.figure()
    plt.plot(values)
    plt.title("Singular Values")

def compare_results(landmarks, state, relative_pose, phis,
                    singular_values, error_vector, phi_x, phi_y, phi_z):
    # Optimized pose
    trans = np.array([[state[0,0]],
                      [state[1,0]],
                      [state[2,0]]])
    yaw = state[3,0]
    pitch = state[4,0]
    roll = state[5,0]
    Xb_opt = tf.transformations.euler_matrix(roll, pitch, yaw)
    Xb_opt[:-1,-1:] = trans
    print("-- Optimized Pose --")
    print(Xb_opt)
    print("-- Real Pose --")
    print(landmarks[0].Xb)
    print("-- Pose Diff --")
    print(landmarks[0].Xb - Xb_opt)

    # Optimized relative pose
    print("-- Optimized Relative Pose --")
    print(relative_pose)
    print("-- Real Optimized Pose --")
    print(landmarks[0].rel_pose)
    print("-- Relative Pose Diff --")
    print(landmarks[0].rel_pose - relative_pose)

    # Optimized Landmarks
    real_landmarks = np.zeros((2*len(landmarks),1))
    real_phis = []
    i=0
    for l in landmarks:
        real_landmarks[i:i+2,:] = l.polar_img1
        real_phis.append(l.phi)
        i += 2
    print("-- Landmark Diff --")
    print(real_landmarks - state[6:,:])
    print("-- Phi Diff --")
    print(np.asarray(real_phis) - np.asarray(phis))

    # Plot Landmarks
    fig = plt.figure()
    plt.title("Landmarks")
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Range')
    ax.set_ylabel('Swath')
    ax.set_zlabel('Depth')
    ax.set_xlim3d(-1, 15)
    ax.set_ylim3d(10, -10)
    ax.set_zlim3d(10, -10)
    x = []
    y = []
    z = []
    # Ground truth
    for l in landmarks:
        x.append(l.cart_img1[0,0])
        y.append(l.cart_img1[1,0])
        z.append(l.cart_img1[2,0])
    ax.scatter(x,y,z, marker='o', color='g', s=72, label='true')
    x = []
    y = []
    z = []
    # Measurements
    for l in landmarks:
        cart = polar_to_cart(np.array([[l.polar_img2[0,0]],
                                       [l.polar_img2[1,0]],
                                       [0.0]]))
        x.append(cart[0,0])
        y.append(cart[1,0])
        z.append(0)
    ax.scatter(x,y,z, marker='*', color='y', s=72*2, label='measurements')
    x = []
    y = []
    z = []
    # Estimates
    for j,i in enumerate(range(6,len(state),2)):
        polar = np.array([[state[i,0]],
                          [state[i+1,0]],
                          [phis[j]]])
        cart = polar_to_cart(polar)
        x.append(cart[0,0])
        y.append(cart[1,0])
        z.append(cart[2,0])
    ax.scatter(x,y,z, marker='^', color='r', s=72, label='estimate')
    # Plot phi projections
    ax.scatter(phi_x, phi_y, phi_z, marker='.', color='b',
               label='projections', alpha=0.2)
    ax.legend()

    # Plot poses
    #Xa always at origin
    poses = []
    origin_x = Arrow3D([0,1],[0,0],[0,0], mutation_scale=5, lw=1,
                       arrowstyle="-|>", color='r')
    poses.append(origin_x)
    origin_y = Arrow3D([0,0],[0,1],[0,0], mutation_scale=5, lw=1,
                       arrowstyle="-|>", color='g')
    poses.append(origin_y)
    origin_z = Arrow3D([0,0],[0,0],[0,1], mutation_scale=5, lw=1,
                       arrowstyle="-|>", color='b')
    poses.append(origin_z)
    # Plot Xb using the computed relative transformation
    ex = relative_pose.dot(np.array([[2],[0],[0],[1]]))
    why = relative_pose.dot(np.array([[0],[2],[0],[1]]))
    zed = relative_pose.dot(np.array([[0],[0],[2],[1]]))
    t = relative_pose[:,-1]
    pose_x = Arrow3D([t[0],ex[0,0]],[t[1],ex[1,0]],[t[2],ex[2,0]], mutation_scale=5, lw=2,
                       arrowstyle="-|>", color='r')
    poses.append(pose_x)
    pose_y = Arrow3D([t[0],why[0,0]],[t[1],why[1,0]],[t[2],why[2,0]], mutation_scale=5, lw=2,
                       arrowstyle="-|>", color='g')
    poses.append(pose_y)
    pose_z = Arrow3D([t[0],zed[0,0]],[t[1],zed[1,0]],[t[2],zed[2,0]], mutation_scale=5, lw=2,
                       arrowstyle="-|>", color='b')
    poses.append(pose_z)
    for a in poses:
        ax.add_artist(a)

    # Plot Singular Values
    plot_svd(singular_values)

    # Plot iteration error
    plt.figure()
    plt.title("Magnitude of Delta per Iteration")
    plt.plot(error_vector)



