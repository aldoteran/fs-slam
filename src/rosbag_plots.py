import rosbag
import numpy as np
import matplotlib.pyplot as plt
import tf
import pickle as pkl

def readmsg(topic_names, bag_name):
    bag = rosbag.Bag(bag_name)
    dead_reckoning = []
    estimated = []
    true = []
    Iter = 0
    for topic, msg, t in bag.read_messages(topics=topic_names):
        if topic == '/slam/dead_reckoning/path':
            if Iter%10 == 0:
                x = msg.poses[-1].pose.position.x
                y = msg.poses[-1].pose.position.y
                z = msg.poses[-1].pose.position.z
                quat = msg.poses[-1].pose.orientation
                dead_reckoning.append((x,y,z,quat))
        elif topic == '/slam/optimized/path':
            x = msg.poses[-1].pose.position.x
            y = msg.poses[-1].pose.position.y
            z = msg.poses[-1].pose.position.z
            quat = msg.poses[-1].pose.orientation
            estimated.append((x,y,z,quat))
        else:
            x = msg.poses[-1].pose.position.x
            y = msg.poses[-1].pose.position.y
            z = msg.poses[-1].pose.position.z
            quat = msg.poses[-1].pose.orientation
            true.append((x,y,z,quat))
        Iter += 1
    bag.close()

    estimated = np.asarray(estimated)
    dead_reckoning = np.asarray(dead_reckoning)
    true = np.asarray(true)

    data_array = np.array([dead_reckoning, estimated, true])
    with open(bag_name+'data.pkl', 'wb') as f:
        pkl.dump(data_array, f)

    plot_all(dead_reckoning, estimated, true)

def plot_all(dead_reckoning, estimated, true):
    # return dead_reckoning, estimated, true
    plt.plot(dead_reckoning[:,0], dead_reckoning[:,1], linestyle='--',
             alpha=0.9, color='r', label='Dead Reckoning', linewidth=4)
    plt.plot(estimated[:,0], estimated[:,1], linestyle='-.', alpha=0.9,
             color='b', label='ISAM2 Estimate', linewidth=4)
    plt.plot(true[:,0], true[:,1], linestyle='-', alpha=0.8,
             color='g', label='Fiducials', linewidth=4)
    plt.scatter(true[0,0], true[0,1], marker='X', color='k',
                s=120, label='Start')
    plt.scatter([true[-1,0],estimated[-1,0],dead_reckoning[-1,0]],
                [true[-1,1],estimated[-1,1],dead_reckoning[-1,1]],
                marker='*', color='y', edgecolors='k', s=300, label='End')
    plt.title("Path", size=18, style='oblique')
    plt.xlabel("X", size=14, style='oblique')
    plt.ylabel("Y", size=14, style='oblique')
    plt.legend()

    plt.figure()
    plt.plot(dead_reckoning[:,2], linestyle='--', alpha=0.9,
             color='r', label='Dead Reckoning', linewidth=4)
    plt.plot(estimated[:,2], linestyle='-.', alpha=0.9,
             color='b', label='ISAM2 Estimate', linewidth=4)
    plt.plot(true[:,2], linestyle='-', alpha=0.8,
             color='g', label='Fiducials', linewidth=4)
    plt.title("Elevation", size=18, style='oblique')
    plt.ylabel("Z", size=14, style='oblique')
    plt.xlabel("timestep", size=14, style='oblique')
    plt.legend()

    fig = plt.figure()
    dead_quat = dead_reckoning[:,-1]
    estimated_quat = estimated[:,-1]
    true_quat = true[:,-1]

    ax1 = fig.add_subplot(131)
    dead_roll = [d.x/d.w for d in dead_quat]
    estimated_roll = [d.x/d.w for d in estimated_quat]
    true_roll = [d.x/d.w for d in true_quat]
    ax1.plot(dead_roll, linestyle='--', alpha=0.9,
            color='r', label='Dead Reckoning', linewidth=4)
    ax1.plot(estimated_roll, linestyle='-.', alpha=0.9,
            color='b', label='ISAM2 Estimate', linewidth=4)
    ax1.plot(true_roll, linestyle='-', alpha=0.8,
            color='g', label='Fiducials', linewidth=4)
    ax1.set_ylabel("Roll [rad]", size=14, style='oblique')
    ax1.set_xlabel("timestep", size=14, style='oblique')
    plt.legend()

    ax2 = fig.add_subplot(132)
    dead_pitch = [d.y/d.w for d in dead_quat]
    estimated_pitch = [d.y/d.w for d in estimated_quat]
    true_pitch = [d.y/d.w for d in true_quat]
    ax2.plot(dead_pitch, linestyle='--', alpha=0.9,
            color='r', label='Dead Reckoning', linewidth=4)
    ax2.plot(estimated_pitch, linestyle='-.', alpha=0.9,
            color='b', label='ISAM2 Estimate', linewidth=4)
    ax2.plot(true_pitch, linestyle='-', alpha=0.8,
            color='g', label='Fiducials', linewidth=4)
    ax2.set_ylabel("Pitch [rad]", size=14, style='oblique')
    ax2.set_xlabel("timestep", size=14, style='oblique')

    ax3 = fig.add_subplot(133)
    dead_yaw = [d.z/d.w for d in dead_quat]
    estimated_yaw = [d.z/d.w for d in estimated_quat]
    true_yaw = [d.z/d.w for d in true_quat]
    ax3.plot(dead_yaw, linestyle='--', alpha=0.9,
            color='r', label='Dead Reckoning', linewidth=4)
    ax3.plot(estimated_yaw, linestyle='-.', alpha=0.9,
            color='b', label='ISAM2 Estimate', linewidth=4)
    ax3.plot(true_yaw, linestyle='-', alpha=0.8,
            color='g', label='Fiducials', linewidth=4)
    ax3.set_ylabel("Yaw [rad]", size=14, style='oblique')
    ax3.set_xlabel("timestep", size=14, style='oblique')
    plt.suptitle("Orientation", size=18, style='oblique')







