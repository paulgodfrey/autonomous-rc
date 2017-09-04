#! /usr/bin/python

# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
from sensor_msgs.msg import Joy
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2

# Instantiate CvBridge
bridge = CvBridge()
frame = 0
throttle = 0
steering = 0

def image_callback(msg):
    global frame, throttle, steering

    current_steering = steering
    current_throttle = throttle

    print(str(msg.header.stamp.to_nsec()), frame, current_steering)

    try:
        # Convert your ROS Image message to OpenCV2
	font = cv2.FONT_HERSHEY_SIMPLEX
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
	cv2_img = cv2.putText(cv2_img, str(current_steering), (20, 40), font, 1, (255,255,255), 2)
	cv2_img = cv2.putText(cv2_img, str(current_throttle), (20, 80), font, 1, (255,255,255), 2)
    except CvBridgeError, e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg 
        frame += 1
        filename = "./images/cam" + str(frame) + ".jpeg"
        
	output = filename + "," + str(current_steering) + "," + str(current_throttle)

        with open('recording.csv', 'a') as recording:
            recording.write(output)
            recording.write("\n")

	cv2.imwrite(filename, cv2_img)

        # print(output)


def joystick_callback(msg):
    global throttle, steering

    throttle = msg.axes[1]
    steering = msg.axes[3]

    print(str(msg.header.stamp.to_nsec()), steering)

def main():
    rospy.init_node('image_listener')

    image_topic = "rgb/image_rect_color"
    joystick_topic = "vesc/joy"    

    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    rospy.Subscriber(joystick_topic, Joy, joystick_callback)

    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()
