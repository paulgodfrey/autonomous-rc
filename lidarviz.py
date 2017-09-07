#!/usr/bin/env python
import rospy

import cv2
import numpy as np
import math

from sensor_msgs.msg import LaserScan

def callback(data):
    frame = np.zeros((500, 500,3), np.uint8)
    angle = data.angle_min
    for r in data.ranges:
        if(r == float('inf')):
            r = 0    

        x = math.trunc( (r * 30)*math.cos(angle + (90*3.1416/180)) *-1 )
       	y = math.trunc( (r * 30)*math.sin(angle + (90*3.1416/180)) )
        cv2.line(frame,(x+249, y+249),(x+250,y+250),(255,0,0),2)
        angle= angle + data.angle_increment

    cv2.circle(frame, (250, 250), 2, (255, 255, 0))
    cv2.imshow('frame',frame)
    cv2.waitKey(1)

def laser_listener():
    rospy.init_node('laser_listener', anonymous=True)
    rospy.Subscriber("/scan", LaserScan,callback)
    rospy.spin()

if __name__ == '__main__':
    laser_listener()
