# OS-Specific Installation Instructions

## Linux

1. Install Docker.
```
./install_docker.sh
```
2. Restart computer.

## Mac

1. Install [Docker Desktop for Mac](https://docs.docker.com/docker-for-mac/install/) (not Docker Toolbox):
[https://download.docker.com/mac/stable/Docker.dmg](https://download.docker.com/mac/stable/Docker.dmg)
2. Install [XQuartz](https://www.xquartz.org).
3. Enable the following setting in XQuartz:

XQuartz > Preferences > Security > Allow connections from network clients

4. Install [TurboVNC](https://sourceforge.net/projects/turbovnc/files/).
5. Run Docker.

## Windows

Partner up with a student with a Mac or Ubuntu laptop, or use the VM provided
last year: [VM Install Guide](https://docs.google.com/document/d/1ley_pauriyx0PrH8XYfkIrZwXnL3s-xBQvcUY6RE02I/edit?usp=sharing)

# ROS Setup

1. Create the Catkin workspace (`catkin_ws`) and Docker network
   (`aa274_net`). Other ROS packages can be put into `catkin_ws` as well.
```
./init_aa274.sh
```
2. Build the Docker image. This should be run any time `docker/Dockerfile` is changed.
```
./build_docker.sh
```
3. Build the Catkin workspace. This should be run any time a new ROS package is added to `catkin_ws`.
```
./rosdep_install.sh
```
4. Whenever you make changes to your own ROS package, compile it with the
   following command:
```
./run.sh catkin_make
```

# Running ROS

Roscore needs to be running before any other nodes. In one terminal window, run:
```
./run.sh roscore
```

Nodes can now be run in separate terminal windows.
```
./run.sh <shell command>
```

To choose the ROS master URI, call `run.sh` with `--rosmaster <hostname>` and/or
`--rosport <port>`. By default, `master:11311` is used.
```
./run.sh --rosmaster master --rosport 11311 <shell command>
```

*Note*: The Catkin workspace and ROS logs are written to the host filesystem (under
`catkin_ws` and `.ros`, respectively). Any changes made to these folders in the
host OS will also be reflected in the Docker containers. However, changes in the
Docker containers outside these folders are temporary and will not persist
across sessions.

If you are using Docker on a Mac (or on a remote Linux host via SSH) and need to
view the GUI, call the command with `--display <display_id>`. This will stream
the rendered GUI through a TurboVNC server. The display ID must be unique and
nonzero.
```
./run.sh --display 1 roslaunch turtlebot3_gazebo turtlebot3_world.launch
```
This command will ask you to create a password for the VNC session. You can then
connect to this session by opening TurboVNC and connecting to the host's address
with the display ID or VNC port. The following examples are equivalent:
```
localhost:1
localhost:5901
127.0.0.1:1
```
The optional `vncport` parameter can be manually specified to avoid port
collisions. Otherwise, the port number will default to `5900 + <display_id>`.
