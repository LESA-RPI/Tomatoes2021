import os
import shutil
import time
import string
from functools import cmp_to_key
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy as sp

#Create new folder
def folder(name):
    if not os.path.exists('./'+name):
        os.makedirs('./'+name)
        print ('Created:', name)

#Input parsing
start = input("Enter start of night cycle in format of HH:MM:SS\n")
end = input("Enter end of night cycle in format HH:MM:SS\n")
inp = False
while inp == False:
	try:
		t = start.split(':')
		h = int(t[0])
		m = int(t[1])
		s = int(t[2])
		t = end.split(':')
		h = int(t[0])
		m = int(t[1])
		s = int(t[2])
		inp = True

	except:
		start = input("Enter start of night cycle in format of HH_MM_SS\n")
		end = input("Enter end of night cycle in format HH_MM_SS\n")

#This takes any input string
nd = input('Enter last date of recording. Ex: 6_12_2021\nType input: ')
folder('graphs')
folder('graphs_ci')
	

#Holds numpy array of plant data
#Takes data point for sorting
class Plant_Time_Data:

	def __init__(self,carbon_dioxide,temperature,humidity,time):
		self.co2 = carbon_dioxide,
		self.hm = humidity
		self.tmp = temperature
		self.time = time

		ymd = (time.split('_'))[0]
		hms = (time.split('_'))[1]
		y = int(ymd.split('-')[0])
		mo = int((ymd.split('-')[1]))
		d = int(ymd.split('-')[2])
		h = int(hms[0] + hms[1])
		mi = int(hms[2] + hms[3])
		s = int(hms[4] + hms[5])

		self.date_time = datetime(y,mo,d,h,mi,s)

	#Getter functions of time/co2/hm/tmp
	def get_time(self):
		return self.date_time

	def get_co2(self):
		return self.co2

	def get_hm(self):
		return self.hm

	def get_tmp(self):
		return self.tmp

	#equals function for self and x to compare
	def equals(self,x):
		if (self.time == x.time and self.hm == x.hm and self.tmp == x.tmp and self.co2 == x.co2):
			return True
		else:
			return False



#Object class of Plant
class Plant:
	def __init__(self,matrix):
		self.Plant_Array = matrix
		self.size = len(matrix)

	#Comparator for self and x object in Plant Class
	def compare(self,x):
		return self[0].get_time() - x[0].get_time()

	#Sort function for sorting the plant
	def sort(self):
		from functools import cmp_to_key
		self.Plant_Array.sort(key=lambda x: x.date_time, reverse=False)

	def ret_size(self):
		return self.size

	def ret_arr(self):
		return self.Plant_Array

#Helper functions for plant array

#Type testing functions
def ret_co2(type):
	if type == 'CO2' or 'CO2:':
		return True
	else:
		return False

def ret_hm(type):
	if type == 'temp' or 'temp:':
		return True
	else:
		return False

def ret_rh(type):
	if type == 'rh' or 'rh:':
		return True
	else:
		return False


#Stripper functions
#Strips number class
def str_num(str):
	str = str.strip('\n')
	str = str.strip(',')
	str = str.strip('C')
	str = str.strip('\'')
	str = str.strip('%')
	str = str.strip('ppm')
	str = str.strip('\'')
	str = str.strip('(')
	str = str.strip(')')
	str = str.strip(',')
	return str

#Strips types in class
def str_type(str):
	str = str.strip('\n')
	str = str.strip(' ')
	str = str.strip(':')	

#Add matrix of data from text directory:
#takes in all 3 matrix types
#adds values and returns matrices
def add_matrix(txt_dir,matrix, day_matrix, night_matrix):
	try:
		f = open(txt_dir)
		lines = f.readlines()

		for line in lines:
			#please feed time into add_matrix first, rest is goint str -> num
			#in co2 -> num -> temp -> num -> hm -> num
			m = line.split(' ')

			time = str(m[0])
			type1 = str(m[1])
			type1_data = float(str(str_num(str(m[2]))))
			type2 = str(m[3])
			type2_data = float(str(str_num(str(m[4]))))+ float(273.15)
			type3 = str(m[5])
			type3_data = float(str(str_num(str(m[6])))) 
			
			#Adding type and data to array
			co2 = 0.0000
			hm = 0.0000
			rh = 0.0000
			#checks data type to make sure which one it is
			if ret_co2(type1):
				co2 = type1_data
			if ret_hm(type2):
				hm = type2_data
			if ret_rh(type3):
				rh = type3_data
			#point added to point matrix
			point = Plant_Time_Data(co2,hm,rh,time)
			#Make sure that point is not in matrix
			matrix.append(point)

			#Splitting parsed time
			hms = (time.split('_'))[1]
			#This is supposed to add strings together and convert to int
			hms = hms[0] + hms[1] + ':' + hms[2] + hms[3] + ':' + hms[4] + hms[5]



			if hms < start or hms > end:
				day_matrix.append(point)
			else:
				night_matrix.append(point)

	finally:
		pass
	return matrix, day_matrix, night_matrix


#Create plant matrix
def create_array():
	input_dir = input("Where are the data files? ex: mnt/c/Users/..../txt_files\n")

	matrix = []
	day_matrix = []
	night_matrix = []
	files = os.listdir(input_dir)

	for file in files:
		print("Sorting..."+str(file) + '\n')
		add_matrix(input_dir + '/' + file,matrix,day_matrix,night_matrix)
	
	plant = Plant(matrix)
	dplant = Plant(day_matrix)
	nplant = Plant(night_matrix)
	plant.sort()
	dplant.sort()
	nplant.sort()
	return plant, dplant, nplant

#Modified function to give 95% confidence interval for graphs

#https://stackoverflow.com/questions/27164114/show-confidence-limits-and-prediction-limits-in-scatter-plot
def conf_graph(x,y,xlab,ylab,nd,ptype,cth):
	import scipy.stats as stats
	# Modeling with Numpy
	def equation(a, b):
	    """Return a 1D polynomial."""
	    return np.polyval(a, b) 

	p, cov = np.polyfit(x, y, 1, cov=True)                     # parameters and covariance from of the fit of 1-D polynom.
	y_model = equation(p, x)                                   # model using the fit parameters; NOTE: parameters here are coefficients
	# Statistics
	n = len(y)                                                 # number of observations
	m = p.size                                                 # number of parameters
	dof = n - m                                                # degrees of freedom
	t = stats.t.ppf(0.975, n - m)                              # used for CI and PI bands
	# Estimates of Error in Data/Model
	resid = y - y_model                           
	chi2 = np.sum((resid / y_model)**2)                        # chi-squared; estimates error in data
	chi2_red = chi2 / dof                                      # reduced chi-squared; measures goodness of fit
	s_err = np.sqrt(np.sum(resid**2) / dof)                    # standard deviation of the error
	# Plotting --------------------------------------------------------------------
	fig, ax = plt.subplots(figsize=(8, 6))
	# Data
	ax.plot(
	    x, y, "o", color="#b9cfe7", markersize=2, 
	    markeredgewidth=1, markeredgecolor="b", markerfacecolor="None"
	)

	# Fit
	ax.plot(x, y_model, "-", color="0.1", linewidth=1.5, alpha=0.5, label="Fit")  

	x2 = np.linspace(np.min(x), np.max(x), 100)
	y2 = equation(p, x2)


	# Prediction Interval
	pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))   
	#ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
	ax.plot(x2, y2 - pi, "--", color="0.5", label="95% Prediction Limits")
	ax.plot(x2, y2 + pi, "--", color="0.5")


	# Figure Modifications --------------------------------------------------------
	# Borders
	ax.spines["top"].set_color("0.5")
	ax.spines["bottom"].set_color("0.5")
	ax.spines["left"].set_color("0.5")
	ax.spines["right"].set_color("0.5")
	ax.get_xaxis().set_tick_params(direction="out")
	ax.get_yaxis().set_tick_params(direction="out")
	ax.xaxis.tick_bottom()
	ax.yaxis.tick_left() 

	# Labels
	plt.title(xlab + ' vs ' + ylab + ' ' + ptype, fontsize="14", fontweight="bold")
	plt.xlabel(xlab)
	plt.ylabel(ylab)
	plt.xlim(np.min(x) - 1, np.max(x) + 1)

	# Custom legend
	handles, labels = ax.get_legend_handles_labels()
	display = (0, 1)
	anyArtist = plt.Line2D((0, 1), (0, 0), color="#b9cfe7")    # create custom artists
	legend = plt.legend(
	    [handle for i, handle in enumerate(handles) if i in display] + [anyArtist],
	    [label for i, label in enumerate(labels) if i in display] + ["95% Confidence Limits"],
	    loc=9, bbox_to_anchor=(0, -0.21, 1., 0.102), ncol=3, mode="expand"
	)  
	frame = legend.get_frame().set_edgecolor("0.5")

	# Save Figure
	plt.tight_layout()
	plt.savefig(str('graphs_ci/ConfidenceInterval_'+nd+'_'+ptype+'_' +cth), bbox_extra_artists=(legend,), bbox_inches="tight")
	plt.clf()


#Holds numpy array data of parsed plants
class Plant_Matrix:

	def __init__(self):
		self.Plant = Plant()
		self.Plant = self.Plant.create_array()
		self.numpy_arr = np.empty([3, size], dtype=float)
		self.dt = np.empty([size], dtype= float)
		self.dto = np.empty([size], dtype= object)

	def __init__(self,Plant):
		self.Plant = Plant


	#Assigns and sorts the entire matrix of data
	def get_np_array(self):
		#Assigns time
		size = self.Plant.ret_size()
		self.numpy_arr = np.empty([3, size], dtype=float)
		arr = self.Plant.ret_arr()
		self.dt = np.empty([size], dtype=float)
		self.dto = np.empty([size], dtype= object)
		import time
		#Time in seconds from epoch
		start = arr[0].get_time()
		start_time = float(time.mktime(start.timetuple()))
		for i in range(size):
			pdp = arr[i]
			#Time from Epoch
			date_time_obj = pdp.get_time()
			t_int = float(time.mktime(date_time_obj.timetuple())) - float(start_time)
			self.dt[i] = (t_int/86400)
			#Getting variables
			co2 = float(str_num(str(pdp.get_co2())))
			tmp = float(pdp.get_tmp())
			hm = float(pdp.get_hm())
			self.dto[i] = date_time_obj
			#Assigning to numpy array
			self.numpy_arr[0][i] = co2
			self.numpy_arr[1][i] = tmp
			self.numpy_arr[2][i] = hm
		return self


	#Graph creating functions
	def create_co2_graph(self,nd,ptype):
		plt.plot(self.dto, self.numpy_arr[0],'o', markersize=2)
		plt.gcf().autofmt_xdate()
		# naming the x axis
		plt.xlabel('Time')
		# naming the y axis
		plt.ylabel('CO2 in ppm')
		# giving a title to my graph
		plt.title('CO2 vs Time Graph')
		# function to show the plot

		plt.savefig('graphs/CO2_Graph_'+nd+'_'+ptype)
		plt.clf()

	def create_hm_graph(self,nd,ptype):
		plt.plot(self.dto, self.numpy_arr[2],'o', markersize=2)
		plt.gcf().autofmt_xdate()
		# naming the x axis
		plt.xlabel('Time')
		# naming the y axis
		plt.ylabel('hm in %')
		# giving a title to my graph
		plt.title('Humidity vs Time Graph')
		# function to show the plot
		plt.savefig('graphs/HM_Graph_'+nd+'_'+ptype)
		plt.clf()

	def create_tmp_graph(self,nd,ptype):
		plt.plot(self.dto, self.numpy_arr[1],'o', markersize=2)
		plt.gcf().autofmt_xdate()

		# naming the x axis
		plt.xlabel('Time')
		# naming the y axis
		plt.ylabel('tmp in Kelvin')
		# giving a title to my graph
		plt.title('Temperature vs Time Graph')
		# function to show the plot
		plt.savefig('graphs/Tmp_Graph_'+nd+'_'+ptype)
		plt.clf()

	def graph_all(self,nd,ptype):
		plt.plot(self.dto, self.numpy_arr[2],'o', markersize=2,label='Tmp in \'K')
		plt.plot(self.dto, self.numpy_arr[1],'o', markersize=2,label='Hm in %')
		plt.plot(self.dto, self.numpy_arr[0],'o', markersize=2,label='Co2 in ppm')
		plt.legend()
		plt.gcf().autofmt_xdate()
		# naming the x axis
		plt.xlabel('Time')
		# naming the y axis
		plt.ylabel('Variables')
		# giving a title to my graph
		plt.title('Variables vs Time Graph')

		# function to show the plot
		plt.savefig('graphs/All_Graph_'+nd+'_'+ptype)
		plt.clf()


	#Plots confidence intervals for all graphs
	def plot_conf(self,nd,ptype):
		x = self.dt #time in days
		y = self.numpy_arr[0] #CO2
		xlab = 'Days'
		ylab = 'CO2 in ppm'
		conf_graph(x,y,xlab,ylab,nd,ptype,'co2')
		y = self.numpy_arr[2] #Humidity
		ylab = 'Humidity in %'
		conf_graph(x,y,xlab,ylab,nd,ptype,'hm')
		y = self.numpy_arr[1] #Temperature
		ylab = 'Temperature in K'
		conf_graph(x,y,xlab,ylab,nd,ptype,'tmp')

	#Exports parsed files into matlab dictionary 
	def matlab_export(self, nd):
		folder('Matlab')
		from scipy.io import savemat
		co2 = self.numpy_arr[0]
		hm = self.numpy_arr[2]
		tmp = self.numpy_arr[1]
		time = self.dt
		mdic = {"CO2": co2, "HM": hm, "TMP": tmp, "Time":time}
		savemat("matlab_vardata_"+nd+".mat", mdic)

	#Creates all graphs in program
	def create_graphs(self, nd, ptype):
		self.get_np_array()
		self.create_co2_graph(nd,ptype)
		self.create_hm_graph(nd,ptype)
		self.create_tmp_graph(nd,ptype)
		self.graph_all(nd,ptype)
		self.plot_conf(nd,ptype)




#Creates plants and graphs data
def create_data():
	whole_plant, day_plant, night_plant = create_array()

	#Whole, day, and night plants
	p = Plant_Matrix(whole_plant)
	d = Plant_Matrix(day_plant)
	n = Plant_Matrix(night_plant)

	p.create_graphs(nd, "All")
	d.create_graphs(nd, "Day")
	n.create_graphs(nd, "Night")

def main():
	create_data()




if __name__ == "__main__":
    main()

