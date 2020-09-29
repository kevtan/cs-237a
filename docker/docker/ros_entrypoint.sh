#!/bin/bash
set -e

# Initialize TurboVNC
if [[ -n $DISPLAY ]]; then
	if [[ -d /tmp/.X11-unix ]]; then
		# Set write permissions on X11
		sudo chmod 777 /tmp/.X11-unix
		echo '/usr/local/lib/x86_64-linux-gnu' | sudo tee -a /etc/ld.so.conf.d/glvnd.conf
		sudo ldconfig
	fi

	# Create VNC password
	if [[ -n $VNCPORT ]]; then
		mkdir -p $HOME/.vnc
		expect <<EOF
spawn vncpasswd
expect "Password:"
send "$VNCPASSWD\r"
expect "Verify:"
send "$VNCPASSWD\r"
expect "Would you like to enter a view-only password (y/n)?"
send "n\r"
expect eof
exit
EOF

		# Run VNC server
		vncserver $DISPLAY -rfbport $VNCPORT
	fi
fi

if [[ $1 == "bash" ]]; then
	# Run interactive Bash
	cd $HOME/catkin_ws
	bash -i
else
	# Run ROS command
	source /ros_env.sh
	cd $HOME/catkin_ws
	exec "$@"
fi
