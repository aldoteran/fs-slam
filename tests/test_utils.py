import tf
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=250)
np.set_printoptions(suppress=True)

import pickle
import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from LandmarkDetector import Landmark

class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            FancyArrowPatch.draw(self, renderer)

def characterize_noise(landmarks, true_landmarks):
    theta_errors = []
    range_errors = []
    for i in range(len(landmarks)):
        diff = landmarks[i].polar_img2 - true_landmarks[i].polar_img2
        theta_errors.append(diff[0,0])
        range_errors.append(diff[1,0])
    theta_mean = np.sum(np.asarray(theta_errors))/len(theta_errors)
    range_mean = np.sum(np.asarray(range_errors))/len(range_errors)
    theta_var = 0
    range_var = 0
    for i in range(len(landmarks)):
        theta_var += (theta_errors[i] - theta_mean)**2
        range_var += (range_errors[i] - range_mean)**2
    theta_var /= len(theta_errors)
    range_var /= len(range_errors)

    print("Mean for angle theta [rad]: {}".format(theta_mean))
    print("Variance for angle theta [rad^2]: {}".format(theta_var))
    print("Mean for range [m]: {}".format(range_mean))
    print("Variance for range [m^2]: {}".format(range_var))

    return (theta_var, theta_mean, range_var, range_mean)

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

def init_plot(x_init, landmarks, phis, phi_proj_x,
              phi_proj_y, phi_proj_z, Xa, Xb, best_idx):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Range')
    ax.set_ylabel('Swath')
    ax.set_zlabel('Depth')
    ax.set_xlim3d(-1, 16+1)
    ax.set_ylim3d(11, -11)
    ax.set_zlim3d(8+1, -8-1)

    plot_scene(fig, ax, x_init, landmarks, phis, phi_proj_x,
               phi_proj_y, phi_proj_z, best_idx)

    return (fig, ax)

def plot_pose(fig, ax, relative_pose):

    poses = []
    ex = relative_pose.dot(np.array([[1],[0],[0],[1]]))
    why = relative_pose.dot(np.array([[0],[1],[0],[1]]))
    zed = relative_pose.dot(np.array([[0],[0],[1],[1]]))
    t = relative_pose[:,-1]
    pose_x = Arrow3D([t[0],ex[0,0]],[t[1],ex[1,0]],[t[2],ex[2,0]], mutation_scale=2, lw=2,
                       arrowstyle="-|>", color='r')
    poses.append(pose_x)
    pose_y = Arrow3D([t[0],why[0,0]],[t[1],why[1,0]],[t[2],why[2,0]], mutation_scale=2, lw=2,
                       arrowstyle="-|>", color='g')
    poses.append(pose_y)
    pose_z = Arrow3D([t[0],zed[0,0]],[t[1],zed[1,0]],[t[2],zed[2,0]], mutation_scale=2, lw=2,
                       arrowstyle="-|>", color='b')
    poses.append(pose_z)
    for a in poses:
        ax.add_artist(a)

    return (fig, ax)


def create_landmarks(N, Xb, Xb_noisy=None, Xa=None, T_Xb=None, origin=False):
    """
    Creates a list of N random landmarks with the first pose
    set in the origin.
    """
    landmarks = [Landmark(test=True) for i in range(N)]
    # Set N landmarks at random distance
    for l in landmarks:
        # Origin at trans 0, Z down, Y right, X forward
        if origin:
            l.Xa = Xa
        else:
            # l.Xa = tf.transformations.euler_matrix(np.pi, 0, 0)
            l.Xa = np.eye(4)
            l.cart_img1 = np.array([[np.round(random.uniform(8,16),3)],
                                    [np.round(random.uniform(-6,6),3)],
                                    [np.round(random.uniform(0,6),3)]])
            l.polar_img1 = cart_to_polar(l.cart_img1)
    landmarks = update_landmarks(landmarks, Xb, Xb_noisy)

    return landmarks

def move_true_origin(x, y, z, roll, pitch, yaw):
    # pose = tf.transformations.euler_matrix(np.pi,0,0)
    pose = np.eye(4)
    trans = np.array([[x],[y],[z]])
    rot = tf.transformations.euler_matrix(roll, pitch, yaw)
    rot[:-1,-1:] = trans

    return pose.dot(rot)

def move_noisy_origin(x, y, z, roll, pitch, yaw, pos_stddev, rot_stddev):
    # pose = tf.transformations.euler_matrix(np.pi,0,0)
    pose = np.eye(4)
    trans = np.array([[x + np.random.normal(0,pos_stddev)],
                      [y + np.random.normal(0,pos_stddev)],
                      [z + np.random.normal(0,pos_stddev)]])
    rot = tf.transformations.euler_matrix(roll + np.random.normal(0,rot_stddev),
                                          pitch + np.random.normal(0,rot_stddev),
                                          yaw + np.random.normal(0,rot_stddev))
    rot[:-1,-1:] = trans

    return pose.dot(rot)

def cart_to_polar(cart):
    return np.array([[np.arctan2(cart[1,0],cart[0,0])],
                     [np.sqrt(cart[0,0]**2 + cart[1,0]**2 + cart[2,0]**2)]])

def update_landmarks(landmarks, true_pose, noisy_pose=None):
    relative_pose = np.linalg.inv(landmarks[0].Xa).dot(true_pose)
    if type(noisy_pose) != None:
        noisy_relative_pose = np.linalg.inv(landmarks[0].Xa).dot(noisy_pose)
    for l in landmarks:
        l.cart_img2 = project_landmark(l.cart_img1, relative_pose)
        l.polar_img2 = cart_to_polar(l.cart_img2)
        if type(noisy_pose) != None:
            l.rel_pose = noisy_relative_pose
            l.Xb = noisy_pose
        else:
            l.rel_pose = relative_pose
            l.Xb = true_pose
        l.real_phi = np.arcsin(l.cart_img1[2,0]/l.polar_img1[1,0])
        l.update_phi(l.real_phi)

    return landmarks

def project_landmark(coords, relative_pose):
    rot = relative_pose[:-1,:-1]
    trans = relative_pose[:-1,-1:]

    return rot.transpose().dot(coords - trans)

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

def plot_scene(fig, ax, state, landmarks, phis, phi_x, phi_y, phi_z, best_idx):
    # Landmarks
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
    # Plot phi projections (cartesian)
    ax.scatter(phi_x, phi_y, phi_z, marker='.', color='b',
               label='phi projections', alpha=0.2)
    # Plot phi projections (polar)
    x = []
    y = []
    z = []
    for i in range(len(phi_x)):
        polar = cart_to_polar(np.array([[phi_x[i]],
                                        [phi_y[i]],
                                        [phi_z[i]]]))
        cart = polar_to_cart(np.array([[polar[0,0]],
                                       [polar[1,0]],
                                       [0.0]]))
        if i == best_idx:
            ax.scatter(cart[0,0],cart[1,0],cart[2,0], marker='.', color='r',
                       alpha=0.2, s=30)
            continue
        x.append(cart[0,0])
        y.append(cart[1,0])
        z.append(cart[2,0])
    ax.scatter(x,y,z, marker='.', color='k', label='polar projection', alpha=0.2)
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
    for a in poses:
        ax.add_artist(a)

    return (fig, ax)

def compare_results(landmarks, state, relative_pose, phis,
                    singular_values, error_vector, phi_x, phi_y, phi_z, best_idx):
    # Optimized relative pose
    print("-- Optimized Relative Pose --")
    print(np.round(relative_pose,4))
    print("-- Real Relative Pose --")
    print(np.round(landmarks[0].rel_pose,4))
    print("-- Relative Pose Diff --")
    print(np.round(landmarks[0].rel_pose - relative_pose,4))

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
    # Plot phi projections (cartesian)
    ax.scatter(phi_x, phi_y, phi_z, marker='.', color='b',
               label='phi projections', alpha=0.2)
    # Plot phi projections (polar)
    x = []
    y = []
    z = []
    for i in range(len(phi_x)):
        polar = cart_to_polar(np.array([[phi_x[i]],
                                        [phi_y[i]],
                                        [phi_z[i]]]))
        cart = polar_to_cart(np.array([[polar[0,0]],
                                       [polar[1,0]],
                                       [0.0]]))
        if i == best_idx:
            ax.scatter(cart[0,0],cart[1,0],cart[2,0], marker='.', color='r',
                       alpha=0.2, s=30)
            continue
        x.append(cart[0,0])
        y.append(cart[1,0])
        z.append(cart[2,0])
    ax.scatter(x,y,z, marker='.', color='k', label='polar projection', alpha=0.2)
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

def plot_search(landmarks, state, relative_pose, phis, rep_errors, phi_x, phi_y, phi_z, best_idx):
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
    # Plot phi projections (cartesian)
    ax.scatter(phi_x, phi_y, phi_z, marker='.', color='b',
               label='phi projections', alpha=0.2)
    # Plot phi projections (polar)
    x = []
    y = []
    z = []
    for i in range(len(phi_x)):
        polar = cart_to_polar(np.array([[phi_x[i]],
                                        [phi_y[i]],
                                        [phi_z[i]]]))
        cart = polar_to_cart(np.array([[polar[0,0]],
                                       [polar[1,0]],
                                       [0.0]]))
        if i == best_idx:
            ax.scatter(cart[0,0],cart[1,0],cart[2,0], marker='.', color='r',
                       alpha=0.2, s=30)
            continue
        x.append(cart[0,0])
        y.append(cart[1,0])
        z.append(cart[2,0])
    ax.scatter(x,y,z, marker='.', color='k', label='polar projection', alpha=0.2)
    ax.legend()

    # Plot poses
    #Xa always at origin
    ex = relative_pose.dot(np.array([[2],[0],[0],[1]]))
    why = relative_pose.dot(np.array([[0],[2],[0],[1]]))
    zed = relative_pose.dot(np.array([[0],[0],[2],[1]]))
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

    plt.show()

def plot_single_search(landmark, polar_state, relative_pose, best_phi,
                       old_error, rep_error, phi_x, phi_y, phi_z,
                       p_x, p_y, p_z, i, phi_range, best_idx):

    # Plot Landmarks
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title("Single Phi Search")
    ax.set_xlabel('Range')
    ax.set_ylabel('Swath')
    ax.set_zlabel('Depth')
    ax.set_xlim3d(-1, 15)
    ax.set_ylim3d(10, -10)
    ax.set_zlim3d(10, -10)

    # Ground truth
    cart = landmark.cart_img1
    ax.scatter(cart[0,0],cart[1,0],cart[2,0], marker='o', color='g', s=72,
               label='true (from Xa)')
    cart = landmark.cart_img2
    ax.scatter(cart[0,0],cart[1,0],cart[2,0], marker='^', color='g', s=72,
               label='true (from Xb)')

    # Measurement
    cart = polar_to_cart(np.array([[landmark.polar_img2[0,0]],
                                   [landmark.polar_img2[1,0]],
                                   [0.0]]))
    ax.scatter(cart[0,0],cart[1,0],cart[2,0], marker='*', color='y', s=72*2,
               label='measurements')

    # Estimate
    polar = np.array([[polar_state[0,0]],
                      [polar_state[1,0]],
                      [best_phi]])
    cart = polar_to_cart(polar)
    ax.scatter(cart[0,0],cart[1,0],cart[2,0], marker='o', color='r', s=72,
               label='estimate (from Xa)')
    polar = np.array([[landmark.polar_img2[0,0]],
                      [landmark.polar_img2[1,0]],
                      [best_phi]])
    cart = polar_to_cart(polar)
    ax.scatter(cart[0,0],cart[1,0],cart[2,0], marker='^', color='r', s=72,
               label='estimate (from Xb)')

    # Plot phi projections (cartesian)
    ax.scatter(phi_x, phi_y, phi_z, marker='.', color='b',
               label='phi projections', alpha=0.2)

    # Plot phi projections (polar)
    x = []
    y = []
    z = []
    for i in range(len(phi_x)):
        polar = cart_to_polar(np.array([[phi_x[i]],
                                        [phi_y[i]],
                                        [phi_z[i]]]))
        cart = polar_to_cart(np.array([[polar[0,0]],
                                       [polar[1,0]],
                                       [0.0]]))
        if i == best_idx:
            ax.scatter(cart[0,0],cart[1,0],cart[2,0], marker='.', color='r',
                       alpha=0.2, s=30)
            continue
        x.append(cart[0,0])
        y.append(cart[1,0])
        z.append(cart[2,0])
    ax.scatter(x,y,z, marker='.', color='k', label='polar projection', alpha=0.2)
    ax.legend()

    # Plot poses
    #Xa always at origin
    ex = relative_pose.dot(np.array([[1],[0],[0],[1]]))
    why = relative_pose.dot(np.array([[0],[1],[0],[1]]))
    zed = relative_pose.dot(np.array([[0],[0],[1],[1]]))
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

    # Plot reprojection error
    ax2 = fig.add_subplot(122)
    ax2.set_title("Reprojection Error")
    ax2.plot(phi_range,rep_error, 'k', label='error')
    ax2.plot([landmark.real_phi, landmark.real_phi],
             [min(rep_error),max(rep_error)], 'g-', label='real')
    ax2.scatter(best_phi, old_error, color='r', label='best')
    ax2.legend()

    plt.show()
