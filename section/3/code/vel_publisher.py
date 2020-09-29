import rospy
from geometry_msgs import Twist


def publisher():
    pass


if __name__ == "__main__":
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass


#!/usr/bin/env python

import rospy
from aa274_s2.msg import MyMessage


def publisher():
    pub = rospy.Publisher("my_topic", MyMessage, queue_size=10)
    rospy.init_node("my_node", anonymous=True)
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        my_message = MyMessage(True, 64, "Freddie")
        pub.publish(my_message)
        rate.sleep()


if __name__ == "__main__":
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
