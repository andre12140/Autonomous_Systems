#!/usr/bin/env python
 
#remove or add the library/libraries for ROS
import rospy
import numpy as np
import scipy.stats
import message_filters
import time

from tf import TransformListener, transformations

#remove or add the message type
from sensor_msgs.msg import LaserScan
from tf.msg import tfMessage
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from math import *

class occupancy_grid_mapping():
	def __init__ (self, xsize, ysize, grid_size):
		#---------------------------------------------------------------------------------------------------------------------------------------------------
		#map parameters
		self.map = OccupancyGrid()
		self.map.header.frame_id = 'my_map'
		self.map.info.height = int(xsize/grid_size)
		self.map.info.width = int(ysize/grid_size)
		self.map.info.resolution = grid_size
		self.map.data = [0]*(int(xsize/grid_size)*int(ysize/grid_size))
		
		
		# set map origin [meters] adapted for better visualization of first_bag in rviz
		self.map.info.origin.position.x = - xsize + 3.5
		self.map.info.origin.position.y = - 3.5
		
		'''
		# set map origin [meters] adapted for better visualization of second_bag in rviz
		self.map.info.origin.position.x = - xsize + 8 
		self.map.info.origin.position.y = - ysize + 8
		'''
		'''
		# set map origin [meters] adapted for better visualization of third_bag in rviz
		self.map.info.origin.position.x = - xsize + 8
		self.map.info.origin.position.y = - ysize + 8
		'''
		#---------------------------------------------------------------------------------------------------------------------------------------------------
		#allocate a matrix for all the map cells (represented by the x and y coordinates of their center) according to the map frame
		self.line_of_cell_y_coordinates = np.arange(-ysize/2.0, ysize/2.0, grid_size, dtype = float)[:, None].T + grid_size/2.0
		self.column_of_cell_x_coordinates = (np.arange(-xsize/2.0, xsize/2.0, grid_size, dtype = float)[:, None] + grid_size/2.0)
		self.grid_cells_x_coordinates = np.tile(self.column_of_cell_x_coordinates, (1, int(ysize/grid_size))) # each column repeated ysize times
		self.grid_cells_y_coordinates = np.tile(self.line_of_cell_y_coordinates, (int(xsize/grid_size), 1)) #each line repeated xsize times
		self.grid_cells_center = np.array([self.grid_cells_x_coordinates, self.grid_cells_y_coordinates], dtype=float)
		
		#---------------------------------------------------------------------------------------------------------------------------------------------------		
		#allocate a matrix for all the relative bearings
		self.line_rel_bearing = np.tile([0], (1, int(ysize/grid_size)))
		self.rep_line_rel_bearing = np.tile(self.line_rel_bearing, (int(xsize/grid_size), 1)) #each column repeated xsize times
		self.relative_bearings = np.array(self.rep_line_rel_bearing, dtype = float)
		
		#---------------------------------------------------------------------------------------------------------------------------------------------------		
		#allocate a matrix for all the relative ranges
		self.line_rel_range = np.tile([0], (1, int(ysize/grid_size)))
		self.rep_line_rel_range = np.tile(self.line_rel_range, (int(xsize/grid_size), 1)) #each column repeated xsize times
		self.relative_ranges = np.array(self.rep_line_rel_range, dtype = float)

		#---------------------------------------------------------------------------------------------------------------------------------------------------		
		#allocate a matrix for all the distances from cell x coordinate to laser x coordinate
		self.line_cell_x = np.tile([0], (1, int(ysize/grid_size)))
		self.rep_line_cell_x = np.tile(self.line_cell_x, (int(xsize/grid_size), 1)) #each column repeated xsize times
		self.x_dist = np.array(self.rep_line_cell_x, dtype = float)
		
		#---------------------------------------------------------------------------------------------------------------------------------------------------		
		#allocate a matrix for all the distances from cell y coordinate to laser y coordinate
		self.line_cell_y = np.tile([0], (1, int(ysize/grid_size)))
		self.rep_line_cell_y = np.tile(self.line_cell_y, (int(xsize/grid_size), 1)) #each column repeated xsize times
		self.y_dist = np.array(self.rep_line_cell_y, dtype = float)
		
		#---------------------------------------------------------------------------------------------------------------------------------------------------		
		#inverse_range_sensor_model parameters
		#thickness of the obstacle
		self.alpha = 0.3 #Can change: thickness of obstacles
		#opening angle of the beam in radians (has to bebigger than the angle increment or it will have gaps between laser beams with un filled cells)
		self.beta = 0.005
		#maximum range of laser
		self.z_max = 30.0 #30 obtained from the laser specifications
		#minimum range of laser
		self.z_min = 0.1 #0.1 obtained from the laser specifications
		
		#----------------------------------------------------------------------------------------------------------------------------------------------------
		# Log-Probabilities to add or remove from the map cells
		self.l_occ = log(0.9 / (1 - 0.9))#Can change: logodds probability of a cell being occupied
		self.l_free = log(0.1 / (1 - 0.1))#Can change: logodds probability of a cell being free
		self.l_0 = log(0.5 / (1 - 0.5))#this is 0, hence we use it as prior
		
		#---------------------------------------------------------------------------------------------------------------------------------------------------		
		#allocate a matrix for all the logodds probabilities of each cell being free or occupied
		self.line_cells_logodds = np.tile([self.l_0], (1, int(ysize/grid_size)))
		self.cells_logodds = np.tile(self.line_cells_logodds, (int(xsize/grid_size), 1)) #each column repeated xsize times
		self.logodds = np.array(self.cells_logodds, dtype = float)
		
		#---------------------------------------------------------------------------------------------------------------------------------------------------		
		#allocate a marix for all the probabilities of each cell being free or occupied
		self.line_cells_prob = np.tile([0], (1, int(ysize/grid_size)))
		self.cells_prob = np.tile(self.line_cells_prob, (int(xsize/grid_size), 1)) #each column repeated xsize times
		self.prob = np.array(self.cells_prob, dtype = float)
		
		#----------------------------------------------------------------------------------------------------------------------------------------------------
		#Set up a tf listener to lookup transform from laser frame to odom frame
		self.tf_listener = TransformListener()
		#Setting up topic Subscribers
		self.sub_laser = message_filters.Subscriber('/scan', LaserScan)
		self.sub_laser_tf = message_filters.Subscriber('/laser_tf', TransformStamped)
		#Setting up topic Publishers
		self.pub_laser_tf = rospy.Publisher('/laser_tf', TransformStamped, queue_size=1, latch=True)
		self.pub_map = rospy.Publisher('/my_map', OccupancyGrid, queue_size=1, latch=True)

#------------------------------------------------------------------------------------------------------------------------------------------------------------
	#Runs update_map and publish_map every time it receives synchronized topic messages from laser pose and laser scan
	def Callback(self, laser_tf, scn):
		#laser scan ranges and bearings
		self.scanner_ranges = []
		self.scanner_bearings = []
		for beam_index in range(len(scn.ranges)):
			#sets the wide angle of the laser scanner to 270 degrees
			#if (scn.angle_min < scn.angle_min + (scn.angle_increment*beam_index) < scn.angle_max):
			
			#sets the wide angle of the laser scanner to 180 degrees
			#if (scn.angle_min + (np.pi)/4 < scn.angle_min + (scn.angle_increment*beam_index) < scn.angle_max - (np.pi)/4):
			
			#sets the wide angle of the laser scanner to 90 degrees
			if (scn.angle_min + (np.pi)/2 < scn.angle_min + (scn.angle_increment*beam_index) < scn.angle_max - (np.pi)/2):
			
			#sets the wide angle of the laser scanner to 45 degrees
			#if (scn.angle_min + (5*(np.pi)/16) < scn.angle_min + (scn.angle_increment*beam_index) < scn.angle_max - (5*(np.pi)/16)):
			
				scn_range = scn.ranges[beam_index]
				scn_bearing = scn.angle_min + (scn.angle_increment*beam_index)
				self.scanner_ranges.append(scn_range)
				self.scanner_bearings.append(scn_bearing)

		#pose of the laser (x, y, theta)
		self.pose1 = [0.0, 0.0, 0.0]
		pose = PoseStamped()
		pose.header.stamp = laser_tf.header.stamp
		pose.header.frame_id = "laser"
		pose.pose.position = laser_tf.transform.translation
		pose.pose.orientation = laser_tf.transform.rotation
		
		
		#set robot pose adapted for better visualization of first_bag in rviz
		self.pose1[0] = laser_tf.transform.translation.x + 11.5
		self.pose1[1] = laser_tf.transform.translation.y - 11.5
		
		'''
		#set robot pose adapted for better visualization of second_bag in rviz
		self.pose1[0] = laser_tf.transform.translation.x + 7
		self.pose1[1] = laser_tf.transform.translation.y + 7
		'''
		'''
		#set robot pose adapted for better visualization of third_bag in rviz
		self.pose1[0] = laser_tf.transform.translation.x + 7
		self.pose1[1] = laser_tf.transform.translation.y + 7
		'''
		
		euler = transformations.euler_from_quaternion([laser_tf.transform.rotation.x, laser_tf.transform.rotation.y, laser_tf.transform.rotation.z, laser_tf.transform.rotation.w])
		self.pose1[2] = euler[2]
		#calling the update map and publish_map functions in a loop
		#start = time.time()
		self.update_map()
		#end = time.time()
		#time_taken = end - start
		#print("Time: " + str(time_taken))
		self.convert_prob_and_publish_map()

#------------------------------------------------------------------------------------------------------------------------------------------------------------
	#Updates the map
	def update_map(self):

		dist = self.grid_cells_center.copy()
		#distance from cell center of mass x-coordinate to laser center x-coordinate
		dist[0, :, :] = self.grid_cells_center[0] - self.pose1[0]
		#distance from cell center of mass y-coordinate to laser center y-coordinate
		dist[1, :, :] = self.grid_cells_center[1] - self.pose1[1]
		#Calculate r (relative range), phi (relative bearing) and k (beam index of the closet beam to cell mi)
		self.relative_ranges = scipy.linalg.norm(dist, axis = 0)
		self.relative_bearings = np.arctan2(dist[1, :, :], dist[0, :, :]) - self.pose1[2]
		
		for i in range(len(self.scanner_ranges)):

			out_of_perceptual_field = (self.relative_ranges > min(self.z_max, self.scanner_ranges[i] + self.alpha/2.0)) | (np.abs(self.relative_bearings - self.scanner_bearings[i]) > self.beta/2.0) | (self.scanner_ranges[i] > self.z_max) #| (self.scanner_ranges[i] < self.z_min)
			inside_perceptual_field = ~(out_of_perceptual_field)

			occupied_cells = (np.abs(self.relative_ranges - self.scanner_ranges[i]) < self.alpha/2.0)
			free_cells = (self.relative_ranges <= self.scanner_ranges[i])
			
			occupied_mask = inside_perceptual_field & occupied_cells
			free_mask = inside_perceptual_field & free_cells

			occupied_mask = occupied_mask.T
			free_mask = free_mask.T
			
			self.logodds[free_mask] += self.l_free - self.l_0
			self.logodds[occupied_mask] += self.l_occ - self.l_0
		
#------------------------------------------------------------------------------------------------------------------------------------------------------------			
	#converts cell probabilities into correct format to be published as an OcuppancyGrid map and publishes said map
	def convert_prob_and_publish_map(self):
		#covert the logodds probabilities back to normal probabilities in the interval [0,100] (scaling done for the OccupancyGrid message type)
		prob = (np.exp(self.logodds)/(1 + np.exp(self.logodds)))*100
		#for each cell with unknown occupancy convert cell value to -1
		unknown_cells = (prob == 50.0)
		prob[unknown_cells]  = -1
		# stamp current ros time to the message
		self.map.header.stamp = rospy.Time.now()
		#get map data from the probabilities of each cell being occupied or free
		self.map.data = prob.flatten()
		#publish map to topic
		self.pub_map.publish(self.map)
#------------------------------------------------------------------------------------------------------------------------------------------------------------
def main_function():
	#start = time.time()
	rospy.init_node('occupancy_grid_mapping')
	#------------------------------------------------------------------------------------------------------------------------------------------------------
	# Define the parameters for the map. This is a 30x30 cell map with grid size 0.5x0.5m
	grid_size = 0.2 #Can change: bigger/smaller cell size; beware that the smaller the cell size the slower the mapping
	xsize = 30
	ysize = 30
	my_map = occupancy_grid_mapping(xsize, ysize, grid_size)
	ats = message_filters.ApproximateTimeSynchronizer([my_map.sub_laser_tf, my_map.sub_laser], 1, 0.01)#Can change: more/less slop between topic messages
	ats.registerCallback(my_map.Callback)
	#------------------------------------------------------------------------------------------------------------------------------------------------------
	while not rospy.is_shutdown():
		target_frame = "map"
		source_frame = "laser"
		if my_map.tf_listener.frameExists(target_frame) and my_map.tf_listener.frameExists(source_frame):
			#lookup transform from laser frame to odom frame
			t = my_map.tf_listener.getLatestCommonTime(target_frame, source_frame)
			translation, rotation = my_map.tf_listener.lookupTransform(target_frame, source_frame, t)
			transform = TransformStamped()
			transform.header.stamp = t
			transform.header.frame_id = source_frame
			transform.child_frame_id = target_frame
			transform.transform.translation.x = translation[0]
			transform.transform.translation.y = translation[1]
			transform.transform.translation.z = translation[2]
			transform.transform.rotation.x = rotation[0]
			transform.transform.rotation.y = rotation[1]
			transform.transform.rotation.z = rotation[2]
			transform.transform.rotation.w = rotation[3]
			my_map.pub_laser_tf.publish(transform)
		#--------------------------------------------------------------------------------------------------------------------------------------------------
	#end = time.time()
	#time_taken = end - start
	#print('Time: ', time_taken)
	rospy.spin()
	#------------------------------------------------------------------------------------------------------------------------------------------------------
