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
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation
import mpl_toolkits.mplot3d as mp3d

from LandmarkDetector import Landmark
from BundleAdjuster import BundleAdjuster

class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            FancyArrowPatch.draw(self, renderer)

class Annotation3D(Annotation):

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)

def annotate3D(ax, s, *args, **kwargs):
    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)

######################################
###### LANDMARK AND POSE TOOLS #######
######################################

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

def create_poses(x, y, z, roll, pitch, yaw, rot_stddev, pos_stddev):
    """
    Create a pose in homogeneous coordinates.

    Returns a noisy pose and a true pose
    """
    trans = np.array([[x],[y],[z]])
    true_pose = tf.transformations.euler_matrix(roll, pitch, yaw)
    true_pose[:-1,-1:] = trans

    trans = np.array([[x + np.random.normal(0,pos_stddev)],
                      [y + np.random.normal(0,pos_stddev)],
                      [z + np.random.normal(0,pos_stddev)]])
    noisy_pose = tf.transformations.euler_matrix(roll + np.random.normal(0,rot_stddev),
                                                 pitch + np.random.normal(0,rot_stddev),
                                                 yaw + np.random.normal(0,rot_stddev))
    noisy_pose[:-1,-1:] = trans

    return (true_pose, noisy_pose)

def create_random_poses(rot_stddev, pos_stddev):
    """
    Creates a random pose pose in homogeneous coordinates.

    Returns a noisy pose and a true pose
    """
    t = np.array([[np.round(random.uniform(-1, 1), 3)],
                  [np.round(random.uniform(-1, 1), 3)],
                  [np.round(random.uniform(-1, 1), 3)]])
    roll = np.round(random.uniform(-0.3, 0.3), 3)
    pitch = np.round(random.uniform(-0.3, 0.3), 3)
    yaw = np.round(random.uniform(-0.3, 0.3), 3)
    true_pose = tf.transformations.euler_matrix(roll, pitch, yaw)
    true_pose[:-1,-1:] = t

    trans = np.array([[t[0,0] + np.random.normal(0,pos_stddev)],
                      [t[1,0] + np.random.normal(0,pos_stddev)],
                      [t[2,0] + np.random.normal(0,pos_stddev)]])
    noisy_pose = tf.transformations.euler_matrix(roll + np.random.normal(0,rot_stddev),
                                                 pitch + np.random.normal(0,rot_stddev),
                                                 yaw + np.random.normal(0,rot_stddev))
    noisy_pose[:-1,-1:] = trans

    return (true_pose, noisy_pose)

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


#############################
##### PLOTTING TOOLS ########
#############################

THETA_MAX = 0.6108
THETA_MIN = -0.6108
MAX_RANGE = 17.0

def init_all_plots():
    fig, axs = plt.subplots(ncols=5, nrows=3)
    gs = axs[0,0].get_subplotspec().get_gridspec()
    for i in range(2):
        for j in range(3):
            axs[j,i].remove()
    ax_3d = fig.add_subplot(gs[0:,0:2], projection='3d')
    fig.tight_layout()

    return (fig, ax_3d, axs)

def init_3d_plot(fig, ax_3d, x_init, landmarks, phis, phi_proj_x,
              phi_proj_y, phi_proj_z, Xa, Xb, best_idx):

    ax_3d.set_title("Elevation Angle Search", {'fontsize': 18,
                                            'fontweight': 'bold',
                                            'verticalalignment': 'baseline',
                                            'horizontalalignment': 'center'})
    ax_3d.set_xlabel('Range')
    ax_3d.set_ylabel('Swath')
    ax_3d.set_zlabel('Depth')
    ax_3d.set_xlim3d(-1, 17+1)
    ax_3d.set_ylim3d(11, -11)
    ax_3d.set_zlim3d(8+1, -8-1)

    plot_scene(fig, ax_3d, x_init, landmarks, phis, phi_proj_x,
               phi_proj_y, phi_proj_z, best_idx)

    return (fig, ax_3d)

def update_pose(fig, ax, relative_pose, landmarks):

    # Relative pose
    rot = relative_pose[:-1,:-1]
    trans = relative_pose[:-1,-1:]

    # Plot Xb using the computed relative transformation
    poses = []
    ex = rot.dot(np.array([[2],[0],[0]])) + trans
    why = rot.dot(np.array([[0],[2],[0]])) + trans
    zed = rot.dot(np.array([[0],[0],[2]])) + trans
    t = trans[:,-1]
    pose_x = Arrow3D([t[0],ex[0,0]],[t[1],ex[1,0]],[t[2],ex[2,0]], mutation_scale=1, lw=3,
                       arrowstyle="-|>", color='r')
    poses.append(pose_x)
    pose_y = Arrow3D([t[0],why[0,0]],[t[1],why[1,0]],[t[2],why[2,0]], mutation_scale=1, lw=3,
                       arrowstyle="-|>", color='g')
    poses.append(pose_y)
    pose_z = Arrow3D([t[0],zed[0,0]],[t[1],zed[1,0]],[t[2],zed[2,0]], mutation_scale=1, lw=3,
                       arrowstyle="-|>", color='b')
    poses.append(pose_z)
    for a in poses:
        ax.add_artist(a)

    annotate3D(ax, s='X_1', xyz=[t[0],t[1],t[2]], fontsize=8, xytext=(3,-3),
               textcoords='offset points', ha='right', va='bottom')

    # Plot zero elevation plane of estimated pose
    vert_1 = trans
    vert_2 = rot.dot(np.array([[MAX_RANGE], [MAX_RANGE*np.sin(THETA_MAX)], [0]])) + trans
    vert_3 = rot.dot(np.array([[MAX_RANGE], [MAX_RANGE*np.sin(THETA_MIN)], [0]])) + trans
    vert = np.array([[vert_1[0,0], vert_1[1,0], vert_1[2,0]],
                     [vert_2[0,0], vert_2[1,0], vert_2[2,0]],
                     [vert_3[0,0], vert_3[1,0], vert_3[2,0]]])
    tri = mp3d.art3d.Poly3DCollection([vert], alpha=0.2, linewidth=1, edgecolor='r')
    alpha = 0.2
    tri.set_facecolor((1,0,0,alpha))
    ax.add_collection3d(tri)

    return (fig, ax)

def plot_rep_error(axs, row, col, rep_error, old_error, best_phi, real_phi, phi_range, idx):

    axs[row,col].plot(phi_range, rep_error, 'k', label='Projection Error')
    axs[row,col].plot([real_phi, real_phi], [min(rep_error), max(rep_error)], 'g--',
                      label='Real Elevation Angle')
    axs[row,col].scatter(best_phi, old_error, color='r',
                         label='Estimated Elevation Angle')
    axs[row,col].set_title("Landmark #{}".format(idx))
    if row == 2 and col == 3:
        axs[row,col].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                            shadow=True, ncol=3)

def plot_svd(values):
    plt.figure()
    plt.plot(values)
    plt.title("Singular Values")

def plot_scene(fig, ax, state, landmarks, phis, phi_x, phi_y, phi_z, best_idx):

    ax.set_xlabel('Range')
    ax.set_ylabel('Swath')
    ax.set_zlabel('Depth')
    ax.set_xlim3d(-1, 16+1)
    ax.set_ylim3d(11, -11)
    ax.set_zlim3d(8+1, -8-1)

    # Landmarks
    x = []
    y = []
    z = []
    # Ground truth
    for l in landmarks:
        x.append(l.cart_img1[0,0])
        y.append(l.cart_img1[1,0])
        z.append(l.cart_img1[2,0])
    ax.scatter(x,y,z, marker='o', color='g', s=72, alpha=0.8,
               label='True Landmark')
    # Estimates
    for j,i in enumerate(range(6,len(state),2)):
        polar = np.array([[state[i,0]],
                          [state[i+1,0]],
                          [phis[j]]])
        cart = polar_to_cart(polar)
        x.append(cart[0,0])
        y.append(cart[1,0])
        z.append(cart[2,0])
    ax.scatter(x,y,z, marker='^', color='r', s=72, label='Estimated Landmark')

    # Measurements
    x = []
    y = []
    z = []
    # from Xa
    for l in landmarks:
        cart = polar_to_cart(np.array([[l.polar_img1[0,0]],
                                       [l.polar_img1[1,0]],
                                       [0.0]]))
        x.append(cart[0,0])
        y.append(cart[1,0])
        z.append(0)
    ax.scatter(x,y,z, marker='*', color='y', s=72*2, alpha=0.8,
               label='Measurement [X_0]')

    # Plot phi projections (cartesian)
    ax.scatter(phi_x, phi_y, phi_z, marker='.', color='b',
               label='Search Arc [X_0]', alpha=0.2)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.01), shadow=True, ncol=2)

    # Plot poses
    #Xa always at origin
    poses = []
    origin_x = Arrow3D([0,1],[0,0],[0,0], mutation_scale=1, lw=1,
                       arrowstyle="-|>", color='r')
    poses.append(origin_x)
    origin_y = Arrow3D([0,0],[0,1],[0,0], mutation_scale=1, lw=1,
                       arrowstyle="-|>", color='g')
    poses.append(origin_y)
    origin_z = Arrow3D([0,0],[0,0],[0,1], mutation_scale=1, lw=1,
                       arrowstyle="-|>", color='b')
    poses.append(origin_z)
    for a in poses:
        ax.add_artist(a)

    # Plot zero elevation plane of initial pose (sonar image plane)
    vert = np.array([[0,0,0],
                     [MAX_RANGE, MAX_RANGE*np.sin(THETA_MAX), 0.0],
                     [MAX_RANGE, MAX_RANGE*np.sin(THETA_MIN), 0.0]])
    tri = mp3d.art3d.Poly3DCollection([vert], alpha=0.2, linewidth=1, edgecolor='b')
    alpha = 0.2
    tri.set_facecolor((0,0,1,alpha))
    ax.add_collection3d(tri)

    # Add text to pose
    # ax.text(0,0,0, 'X_0', (1,0,0))
    annotate3D(ax, s='X_0', xyz=[0,0,0], fontsize=8, xytext=(-3,3),
               textcoords='offset points', ha='right', va='bottom')

    return (fig, ax)

def compare_results(landmarks, state, relative_pose, phis,
                    singular_values, error_vector, phi_x, phi_y, phi_z,
                    best_idx, Xb_true):

    # Optimized relative pose
    print("-- Optimized Relative Pose --")
    print(np.round(relative_pose,4))
    print("-- Real Relative Pose --")
    print(np.round(np.linalg.inv(landmarks[0].Xa).dot(Xb_true),4))
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

    # Plot iteration error
    plt.figure()
    plt.title("Magnitude of Delta per Iteration")
    plt.plot(error_vector)

def plot_single_search(landmark, polar_state, relative_pose, best_phi,
                       old_error, rep_error, phi_x, phi_y, phi_z,
                       q_x, q_y, q_z, phi_range, best_idx):

    # Plot Landmarks
    fig = plt.figure()
    # Relative pose
    rot = relative_pose[:-1,:-1]
    trans = relative_pose[:-1,-1:]

    ##########
    # PLOT 1 #
    ##########

    ax = fig.add_subplot(121, projection='3d')
    ax.set_title("Elevation Angle Search", {'fontsize': 18,
                                            'fontweight': 'bold',
                                            'verticalalignment': 'baseline',
                                            'horizontalalignment': 'center'})
    ax.set_xlabel('Range')
    ax.set_ylabel('Swath')
    ax.set_zlabel('Depth')
    ax.set_xlim3d(-1, 18)
    ax.set_ylim3d(11, -11)
    ax.set_zlim3d(8, -8)

    # Ground truth
    cart = landmark.cart_img1
    ax.scatter(cart[0,0],cart[1,0],cart[2,0], marker='o', color='g', s=72,
               label='True Landmark')
    # Estimate
    polar = np.array([[polar_state[0,0]],
                      [polar_state[1,0]],
                      [best_phi]])
    cart = polar_to_cart(polar)
    ax.scatter(cart[0,0],cart[1,0],cart[2,0], marker='^', color='r', s=72,
               label='Estimated Landmark')

    # Measurement from Xa
    cart = polar_to_cart(np.array([[landmark.polar_img1[0,0]],
                                   [landmark.polar_img1[1,0]],
                                   [0.0]]))
    ax.scatter(cart[0,0],cart[1,0],cart[2,0], marker='*', color='y', s=72*2,
               alpha=0.8, label='Measurement [X_0]')
    # Measurement from Xb
    cart = polar_to_cart(np.array([[landmark.polar_img2[0,0]],
                                   [landmark.polar_img2[1,0]],
                                   [0.0]]))
    cart = rot.dot(cart) + trans
    ax.scatter(cart[0,0],cart[1,0],cart[2,0], marker='*', color='y', s=72*2,
               alpha=0.8, edgecolor='k', label='Measurement [X_1]')

    # Plot search arc wrt to Xa
    ax.scatter(phi_x, phi_y, phi_z, marker='.', color='b',
               label='Search Arc [X_0]', alpha=0.2)

    # Plot arc projections wrt Xb
    x = []
    y = []
    z = []
    for i in range(len(q_x)):
        polar = cart_to_polar(np.array([[q_x[i]],
                                        [q_y[i]],
                                        [q_z[i]]]))
        cart = polar_to_cart(np.array([[polar[0,0]],
                                       [polar[1,0]],
                                       [0.0]]))
        cart = rot.dot(cart) + trans
        if i == phi_range.tolist().index(best_phi):
            ax.scatter(cart[0,0],cart[1,0],cart[2,0], marker='.', color='r',
                       alpha=1.0, s=30)
            continue
        x.append(cart[0,0])
        y.append(cart[1,0])
        z.append(cart[2,0])
    ax.scatter(x,y,z, marker='.', color='k', label='Search Arc Projections [X_1]', alpha=0.2)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.01), shadow=True, ncol=2)

    # Plot poses
    #Xa always at origin
    poses = []
    origin_x = Arrow3D([0,0.5],[0,0],[0,0], mutation_scale=1, lw=1,
                       arrowstyle="-|>", color='r')
    poses.append(origin_x)
    origin_y = Arrow3D([0,0],[0,0.5],[0,0], mutation_scale=1, lw=1,
                       arrowstyle="-|>", color='g')
    poses.append(origin_y)
    origin_z = Arrow3D([0,0],[0,0],[0,0.5], mutation_scale=1, lw=1,
                       arrowstyle="-|>", color='b')
    poses.append(origin_z)

    # Plot Xb using the computed relative transformation
    # ex = rot.dot(np.array([[0.5],[0],[0]]) + trans)
    # why = rot.dot(np.array([[0],[0.5],[0]]) + trans)
    # zed = rot.dot(np.array([[0],[0],[0.5]]) + trans)
    # t = rot.dot(trans)[:,-1]
    ex = rot.dot(np.array([[0.5],[0],[0]])) + trans
    why = rot.dot(np.array([[0],[0.5],[0]])) + trans
    zed = rot.dot(np.array([[0],[0],[0.5]])) + trans
    t = relative_pose[:,-1]
    pose_x = Arrow3D([t[0],ex[0,0]],[t[1],ex[1,0]],[t[2],ex[2,0]], mutation_scale=1, lw=3,
                       arrowstyle="-|>", color='r')
    poses.append(pose_x)
    pose_y = Arrow3D([t[0],why[0,0]],[t[1],why[1,0]],[t[2],why[2,0]], mutation_scale=1, lw=3,
                       arrowstyle="-|>", color='g')
    poses.append(pose_y)
    pose_z = Arrow3D([t[0],zed[0,0]],[t[1],zed[1,0]],[t[2],zed[2,0]], mutation_scale=1, lw=3,
                       arrowstyle="-|>", color='b')
    poses.append(pose_z)
    for a in poses:
        ax.add_artist(a)

    # Annotate origin
    annotate3D(ax, s='X_0', xyz=[0,0,0], fontsize=8, xytext=(-3,3),
               textcoords='offset points', ha='right', va='bottom', style='oblique')
    # Annotate estimated pose
    annotate3D(ax, s='X_1', xyz=[t[0],t[1],t[2]], fontsize=8, xytext=(10,3),
               textcoords='offset points', ha='right', va='bottom', style='oblique')

    # Plot zero elevation plane of initial pose (sonar image plane)
    vert = np.array([[0,0,0],
                     [MAX_RANGE, MAX_RANGE*np.sin(THETA_MAX), 0.0],
                     [MAX_RANGE, MAX_RANGE*np.sin(THETA_MIN), 0.0]])
    tri = mp3d.art3d.Poly3DCollection([vert], alpha=0.2, linewidth=1, edgecolor='b')
    alpha = 0.2
    tri.set_facecolor((0,0,1,alpha))
    ax.add_collection3d(tri)

    # Plot zero elevation plane of estimated pose
    vert_1 = trans
    vert_2 = rot.dot(np.array([[MAX_RANGE], [MAX_RANGE*np.sin(THETA_MAX)], [0]])) + trans
    vert_3 = rot.dot(np.array([[MAX_RANGE], [MAX_RANGE*np.sin(THETA_MIN)], [0]])) + trans
    vert = np.array([[vert_1[0,0], vert_1[1,0], vert_1[2,0]],
                     [vert_2[0,0], vert_2[1,0], vert_2[2,0]],
                     [vert_3[0,0], vert_3[1,0], vert_3[2,0]]])
    tri = mp3d.art3d.Poly3DCollection([vert], alpha=0.2, linewidth=1, edgecolor='r')
    alpha = 0.2
    tri.set_facecolor((1,0,0,alpha))
    ax.add_collection3d(tri)

    ##########
    # PLOT 2 #
    ##########

    # Plot reprojection error
    ax2 = fig.add_subplot(122)
    ax2.set_title("Reprojection Error", {'fontsize': 18,
                                         'fontweight': 'bold',
                                         'verticalalignment': 'baseline',
                                         'horizontalalignment': 'center'})
    ax2.set_xlabel("Elevation Angle Phi")
    ax2.set_ylabel("Mahalanobis Distance")
    ax2.plot(phi_range,rep_error, 'k', label='Error')
    ax2.plot([landmark.real_phi, landmark.real_phi],
             [min(rep_error),max(rep_error)], 'g--', label='True angle [rad]')
    ax2.scatter(best_phi, old_error, color='r', label='Estimated angle [rad]')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               shadow=True, ncol=3)

    plt.show()

def plot_search(ax, landmarks, relative_pose, q_proj_x, q_proj_y, q_proj_z, phi_range, best_idx):

    # Relative pose
    rot = relative_pose[:-1,:-1]
    trans = relative_pose[:-1,-1:]

    # Measurement from Xb
    x = []
    y = []
    z = []
    for l in landmarks:
        cart = polar_to_cart(np.array([[l.polar_img2[0,0]],
                                       [l.polar_img2[1,0]],
                                       [0.0]]))
        cart = rot.dot(cart) + trans
        x.append(cart[0,0])
        y.append(cart[1,0])
        z.append(cart[2,0])
    ax.scatter(x, y, z, marker='*', color='y', s=72*2,
               alpha=0.8, edgecolor='k', label='Measurement [X_1]')

    # Plot arc projections wrt Xb
    x = []
    y = []
    z = []
    for i in range(len(q_proj_x)):
        polar = cart_to_polar(np.array([[q_proj_x[i]],
                                        [q_proj_y[i]],
                                        [q_proj_z[i]]]))
        cart = polar_to_cart(np.array([[polar[0,0]],
                                       [polar[1,0]],
                                       [0.0]]))
        cart = rot.dot(cart) + trans
        x.append(cart[0,0])
        y.append(cart[1,0])
        z.append(cart[2,0])
    ax.scatter(x,y,z, marker='.', color='k', label='Search Arc Projections [X_1]', alpha=0.2)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.01), shadow=True, ncol=2)

    plt.show()

##############################
#### RANDOM TESTING TOOLS ####
##############################

def setBoxColors(bp):
    plt.setp(bp['boxes'][0], color='blue', linewidth=2)
    plt.setp(bp['caps'][0], linewidth=2)
    plt.setp(bp['caps'][1], linewidth=2)
    plt.setp(bp['whiskers'][0], linewidth=2)
    plt.setp(bp['whiskers'][1], linewidth=2)
    plt.setp(bp['medians'][0], linewidth=2)

    plt.setp(bp['boxes'][1], color='red', linewidth=2)
    plt.setp(bp['caps'][2], linewidth=2)
    plt.setp(bp['caps'][3], linewidth=2)
    plt.setp(bp['whiskers'][2], linewidth=2)
    plt.setp(bp['whiskers'][3], linewidth=2)
    plt.setp(bp['medians'][1], linewidth=2)

def random_test(N, rot_stddev, pos_stddev):
    """
    Run the bundle adjustment framework N times with random poses and landmarks.
    Computes the error and plots the relevant data to evaluate the performance.
    """
    adjuster = BundleAdjuster(verbose=False, test=False, benchmark=True, iters=5)

    # Pose error
    x_err = []
    y_err = []
    z_err = []
    roll_err = []
    pitch_err = []
    yaw_err = []
    x_init = []
    y_init = []
    z_init = []
    roll_init = []
    pitch_init = []
    yaw_init = []
    # landmark error
    range_err = []
    bearing_err = []
    phi_err = []
    for i in range(N):
        Xb_true, Xb = create_random_poses(rot_stddev, pos_stddev)
        landmarks = create_landmarks(9, Xb_true, Xb)
        try:
            state, relative_pose, phis = adjuster.compute_constraint(landmarks,
                                                                     rot_stddev,
                                                                     pos_stddev)
        except TypeError:
            continue
        # Translation error
        x_err.append(np.linalg.norm(Xb_true[0,-1] - state[0,0]))
        y_err.append(np.linalg.norm(Xb_true[1,-1] - state[1,0]))
        z_err.append(np.linalg.norm(Xb_true[2,-1] - state[2,0]))
        # Rotation error
        roll, pitch, yaw = tf.transformations.euler_from_matrix(Xb_true)
        yaw_err.append(np.linalg.norm(yaw - state[3,0]))
        pitch_err.append(np.linalg.norm(pitch - state[4,0]))
        roll_err.append(np.linalg.norm(roll - state[5,0]))
        # Initial error
        roll_, pitch_, yaw_ = tf.transformations.euler_from_matrix(Xb)
        x_init.append(np.linalg.norm(Xb_true[0,-1] - Xb[0,-1]))
        y_init.append(np.linalg.norm(Xb_true[1,-1] - Xb[1,-1]))
        z_init.append(np.linalg.norm(Xb_true[2,-1] - Xb[2,-1]))
        yaw_init.append(np.linalg.norm(yaw - yaw_))
        pitch_init.append(np.linalg.norm(pitch - pitch_))
        roll_init.append(np.linalg.norm(roll - roll_))

        k = 0
        for j,l in enumerate(landmarks):
            polar = l.polar_img1
            bearing_err.append(np.linalg.norm(state[6+k,0] - polar[0,0]))
            range_err.append(np.linalg.norm(state[6+k+1,0] - polar[1,0]))
            phi_err.append(np.linalg.norm(phis[j] - l.real_phi))
            k += 2

    X = [x_init, x_err]
    Y = [y_init, y_err]
    Z = [z_init, z_err]
    Roll = [roll_init, roll_err]
    Pitch = [pitch_init, pitch_err]
    Yaw = [yaw_init, yaw_err]
    # Box plot for pose
    fig = plt.figure()
    ax1 = plt.subplot()
    ax1.set_title("Absolute Error for Estimated Pose")
    labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    bplot1 = ax1.boxplot(X, positions=[1,2], notch=True, sym='rx', widths=0.6, patch_artist=True)
    setBoxColors(bplot1)
    bplot2 = ax1.boxplot(Y, positions=[4,5], notch=True, sym='rx', widths=0.6, patch_artist=True)
    setBoxColors(bplot2)
    bplot3 = ax1.boxplot(Z, positions=[7,8], notch=True, sym='rx', widths=0.6, patch_artist=True)
    setBoxColors(bplot3)
    bplot4 = ax1.boxplot(Roll, positions=[10,11], notch=True, sym='rx', widths=0.6, patch_artist=True)
    setBoxColors(bplot4)
    bplot5 = ax1.boxplot(Pitch, positions=[13,14], notch=True, sym='rx', widths=0.6, patch_artist=True)
    setBoxColors(bplot5)
    bplot6 = ax1.boxplot(Yaw, positions=[16,17], notch=True, sym='rx', widths=0.6, patch_artist=True)
    setBoxColors(bplot6)
    ax1.yaxis.grid(True)
    ax1.set_xticklabels(labels)
    ax1.set_xticks([1.5, 4.5, 7.5, 10.5, 13.5, 16.5])
    plt.xlim(0,18)

    # data = [range_err, bearing_err, phi_err]
    # ax2 = plt.subplot(122)
    # ax2.set_title("Absolute Error for Estimated Landmarks")
    # labels = ['Range', 'Bearing', 'Elevation']
    # bplot2 = ax2.boxplot(data, notch=True, sym='rx', labels=labels, patch_artist=True)
    # colors = ['pink', 'lightblue', 'lightgreen']
    # for box in bplot2:
        # for patch, color in zip(bplot2['boxes'], colors):
            # patch.set_facecolor(color)
            # patch.set(linewidth=2)
    # for whisker in bplot2['whiskers']:
        # whisker.set(linewidth=2)
    # for median in bplot2['medians']:
        # median.set(linewidth=2)
    # for cap in bplot2['caps']:
        # cap.set(linewidth=2)
    # ax2.yaxis.grid(True)
