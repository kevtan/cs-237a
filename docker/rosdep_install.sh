./run.sh rosdep update
./run.sh rosdep install --from-paths src --ignore-src -r -y
./run.sh catkin_make
./run.sh catkin_make install
