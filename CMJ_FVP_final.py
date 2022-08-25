import streamlit as st 
import pandas as pd 
import statistics
import math
import numpy as np
from nptdms import TdmsFile
from scipy import integrate
from scipy import signal
from glob import glob 
import scandir
from pathlib import Path
import matplotlib.pyplot as plt

st.title("Force Velocity Profiling - CMJ")

col1, col2 = st.columns(2, gap="small")
with col1:
	Computor_select = st.radio("Which OS are you using?", ('Mac','Windows'))
with col2: 
	folder_path = st.text_input("Enter data's location")
	# this could be changed to an st.file_uploader("upload data here", accept_multiple_files=True)
FP_select = st.selectbox("select force plate type", options=['no selection','AMTI FP', 'Other FP options can be placed here']) # << Selection of FP options 

def FP_typer(name_of_x_df): # << takes FP_select input and runs through if statements to determine which columns will be selected of the dataframe
	if FP_select == 'no selection':
		st.error('no FP has been selected')
	elif FP_select == 'AMTI FP':
		global raw_z_total
		raw_z_left = temp_tdms_frame.iloc[:,2]
		raw_z_right = temp_tdms_frame.iloc[:,8]
		raw_z_total = raw_z_right + raw_z_left
	elif FP_select == 'Other FP options can be placed here':
		raw_z_total = x.iloc[:,1] # << this is an example for future additions 
	else:
		st.error('no FP selected')			

# two dataframes are generated for later use 
raw_FP_data = pd.DataFrame()
raw_graph_data = pd.DataFrame()

# manual calculation of hpo 
with st.sidebar:
	st.subheader("Input variables")
	sample_rate = st.number_input('Sample rate')
	athlete_mass = st.number_input('Athletes mass (kg)')

	
def cpu_select(): # << Mac and Windows have slightly different ways to read files in folders 
	global location
	if Computor_select == 'Mac':
		location = str(folder_path) + str(files.name)
	elif Computor_select == 'Windows':
		location = str(folder_path) + '\\' + str(files.name)

def Quite_stance_first(name_of_dataframe):
	global quiet_stance, quiet_start, quiet_end, start_unweighting, raw_FP_rolling, quiet_stance, data_end
	rolling_sd = name_of_dataframe.rolling(50).std()  #<< rolling SD is taken of the base data to determine first quite stance period 
	rolling_sd_a = rolling_sd.to_numpy() #<< an array is created to perform numpy manipulation 
	quiet_stance_thresh = np.where(np.logical_and(rolling_sd_a >-1, rolling_sd_a <1))[0] #<< threshold values are determined by where rolling SD is greater/less than 1 
	quite_range = rolling_sd.where(rolling_sd >5).first_valid_index()  #			 <<<<<<<<< unweighting start (End of quite range) is in accordance with Perez-Castilla 2019:	signal is detected by 5SD of quiet stance	
	quiet_stance_og = np.where(quiet_stance_thresh < quite_range)[0] # after unweighting has been detected, quite stance start/end is found between unweighting phase and onset of quite stance threshold
	quiet_start = quiet_stance_thresh[quiet_stance_og[0]]
	quiet_end = quiet_stance_thresh[quiet_stance_og[-1]]
	quiet_stance = name_of_dataframe.iloc[quiet_start:quiet_end].mean() # << quite stance = the mean of values between start and stop of quite stance 
	data_end = name_of_dataframe.last_valid_index() # is saved for later signal detections 


def weight_vars():
	global g, total_w, total_m
	g = 9.81
	total_w = round(quiet_stance, 3) #<< is the total weight of subject + system 
	total_m = float(round(total_w/g, 3)) 



# This is how files of a certain type are detected 

object = scandir(folder_path) #<< folder is scanned for files 
for files in object: # << for the files in the folder...
    if files.path.endswith('.tdms'): # if the file = .tdms, create a dataframe based on FP selection, then determine what weight that set of jumps was performed at 
        # For windows, this must have a '\\' added to it. 
        cpu_select()
        read_tdms = TdmsFile.read(location)
        temp_tdms_frame = read_tdms.as_dataframe()
        jump_output_data = pd.DataFrame(temp_tdms_frame)
        FP_typer(jump_output_data)
        Quite_stance_first(raw_z_total)
        weight_vars()
        jump_weight_words = str(round(total_m - athlete_mass, 1)) + "kg Extra" # << store the weight of the files jump as the general column of this data file
        jump_weight = round(total_m - athlete_mass,4)
        if raw_FP_data.last_valid_index() is None: # << if the file is the first one, then add it to the dataframe
        	raw_FP_data.insert(0, jump_weight_words, raw_z_total.values)
        	raw_graph_data.insert(0,jump_weight, raw_z_total.values)

        elif jump_output_data.last_valid_index() is not None: # << if the file is not the first one, then add it to the other dataframe, then name it based on weight of jump
        	raw_FP_data.insert(0, 'temp', raw_FP_data.loc[0])
        	raw_graph_data.insert(0,'temp',raw_FP_data.loc[0])
        	jump_series = pd.Series(raw_z_total)
        	raw_FP_data = pd.concat([raw_FP_data, jump_series.rename(jump_weight_words)], axis=1)
        	raw_graph_data = pd.concat([raw_graph_data, jump_series.rename(jump_weight)],axis=1)
        	raw_FP_data.drop('temp', inplace=True, axis=1)
        	raw_graph_data.drop('temp', inplace=True,axis=1)

def jumps_loaded():				# << collects the number of columns that went through the for loop and takes the index of them
	total_jumps = str((len(raw_FP_data.columns)))
	total_jumps_comb = total_jumps + ' Files have been loaded'
	st.success(total_jumps_comb)


# Sort the data frames by weights used (ascending)
raw_FP_data = raw_FP_data.sort_values(by=1, axis=1, ascending=True)
raw_graph_data = raw_graph_data.sort_values(by=1,axis=1,ascending=True)


#VVV-----FEATURE DETECION----VVV#

def integ_velocity(start): #<<<<<----------------------- This is how velocity and displacement can be integrated with as little error as possible
	global veloc
	inters_col = raw_FP_data[column].values # << values from a given column are placed into an array
	remove_nan = inters_col[np.logical_not(np.isnan(inters_col))] # N/a values are removed 
	inters = pd.DataFrame(remove_nan) # saved as a pandas df
	inters[0.1] = ((inters.iloc[start:data_end]) - total_w) / total_m # data is selected over a select interval then offset
	accel = inters[0.1].dropna()
	veloc = (integrate.cumtrapz(accel, x=None, dx=(1/sample_rate), initial=0)) # velocity is taken as the integral of acceleration
	veloc = pd.DataFrame(veloc)

def unweighting(quiet_start_num, name_of_dataframe):
	global eccentric_start, concentric_start
	rolling_sd = name_of_dataframe.rolling(50).std()
	# unweighting start is calculated by the difference of more than 10SD's from the regular data (after quite start has occurred)
	eccentric_start = rolling_sd.where(rolling_sd.iloc[quiet_start_num:-1] > 5).first_valid_index() -25 #<<<<-------  This was increased to 10SD's because errors would often occur at 5SD's (Perez-Castilla 2019)
	
	#backoff = float(sample_rate*0.3)
	#test = 	int(float(rolling_sd.where(rolling_sd.iloc[quiet_start_num:-1] > 5).first_valid_index()) - backoff)
	#st.write(test)
	#eccentric_start = test # for some reason this makes the print function write a lot / make an infinite loop..... not sure how / why
	# - sample_rate * 0.3 # this is equivalent to 30ms before 5SDSW was detected 

	# int(round(sample_rate*0.3, 1)))

	integ_velocity(eccentric_start)
	concentric_start = int(veloc.where(veloc > 0).first_valid_index()) + int(eccentric_start) #<<<<--------  Unweighting end = end of eccentric/beginning of concentric is = the point at which velocity is 0 (Lindberg et al., 2021)

def propulsion(name_of_dataframe): #<-------------- Once FP reads below quite stance, take-off is found; could be adjusted to 10N as per (Lindberg et al., 2021) <<--------------- THIS COULD BE CHANGED TO WHEN FORCE IS < QUITE STANCE
	global take_off
	take_off = name_of_dataframe.iloc[concentric_start:-1].where(name_of_dataframe < quiet_stance).first_valid_index()

def quiet_stance_etc(propulsion_num, name_of_dataframe): #<----------- is determined by the next point in which quiet stance is between +/- 5N (could change this to std to reflect previous statements)
	global quiet_stance_start_next 
	recovery_df = name_of_dataframe.iloc[take_off:-1] 
	recovery = (recovery_df < quiet_stance + 5) & (recovery_df > quiet_stance - 5) # finds next values that are between +/- 5N of quite stance
	recovery_rolling = recovery.rolling(50, axis=0).mean()
	quiet_stance_start_next = recovery_rolling.where(recovery_rolling == 1).first_valid_index() # finds the first place where mean of +/- 5 is = 1

def cmj(): # functions used to determine the first jump
	unweighting(quiet_start, jump_df)
	propulsion(jump_df)
	quiet_stance_etc(take_off, jump_df)

def cmj_etc(): # functions to find all jumps after 1
	unweighting(quiet_stance_start_next, jump_df)
	propulsion(jump_df)
	Impulse_mom(quiet_stance_start_next,take_off)
	quiet_stance_etc(take_off, jump_df)

def Impulse_mom(quiet_start_num, propulsion_num):
	global y_height, v_take_off, mean_force, power, y_hpo_auto
	weight_vars()
	inters_col = raw_FP_data[column].values # takes a singular column and makes its own df from it 
	remove_nan = inters_col[np.logical_not(np.isnan(inters_col))] # removes any n/a values that were acquired from initial data merging 
	inters = pd.DataFrame(remove_nan)
	inters[0.1] = ((inters.iloc[concentric_start:take_off]) - total_w) / total_m # data is offset by total weight then divided by total mass to give acceleration (from only during concentric contraction)
	accel = inters[0.1].dropna()
	veloc = (integrate.cumtrapz(accel, x=None, dx=(1/sample_rate), initial=0)) # velocity is integrated from acceleration
	# velocity is taken at when quite stance is crossed during take off phase
		# to calculate mean velocity
	v_take_off = float(pd.DataFrame(veloc).mean()) # <------------average velocity is calculated by integrating acceleration ( force - offset / mass ) over the period of the concentric phase (propulsion) () 

	#hpo can be calculated automatically by taking the integrated velocity values of the jump
	displace = (integrate.cumtrapz(veloc, x=None, dx=(1/sample_rate), initial=0))
	holder = pd.DataFrame(displace)
	h = holder.last_valid_index()
	y_hpo_auto = float(holder.iloc[h]) # Thus, Hpo = take off height - displacement at braking end = last value of displacement at quiet stance crossing 
	mean_force = (raw_FP_data[column].iloc[concentric_start:take_off]).mean() #<-------- average Force is calculated as the average force developed over concentric phase (start= unweighthing_end, end= take_off)
	power = round((mean_force * v_take_off),2)

#Dataframes of markers and kinematics are created 
jump_markers = pd.DataFrame(columns=['Quite Stance','Eccentric Start','Concentric Start','Take Off','Jump End'])
jump_kinematics = pd.DataFrame(columns=['Additional weight (kg)','Avg Velocity (m/s)', 'Avg Force (N)', 'Avg Power (kg⋅m2⋅s−3)', 'hpo Auto'])

#VVV--------For loop to determine first jump and all other jumps -----VVV#

for column in raw_FP_data:
	global jump_df
#taking each column of the raw dataset and creating its own df for calculations 
	jump_column = raw_FP_data[column].values
# removing an NA values 
	remove_nan = jump_column[np.logical_not(np.isnan(jump_column))]
	jump_df = pd.DataFrame(remove_nan)
#generating base points for quiet stance and data end point
	Quite_stance_first(jump_df)
# generating body mass kinematics 
	weight_vars()
# Executing the first jump calculation
	cmj()
	jump_marks = {'Quite Stance': quiet_start,'Eccentric Start': eccentric_start,'Concentric Start':concentric_start,'Take Off':take_off, 'Jump End': quiet_stance_start_next} # all values are placed into df 
	jump_markers = jump_markers.append(jump_marks, ignore_index=True)
	Impulse_mom(quiet_start, take_off)
	jump_kines = {'Additional weight (kg)': round(float(total_m - athlete_mass),4),'Avg Velocity (m/s)': v_take_off, 'Avg Force (N)': mean_force,'Avg Power (kg⋅m2⋅s−3)': power, 'hpo Auto': y_hpo_auto}
	jump_kinematics = jump_kinematics.append(jump_kines, ignore_index=True)

	while quiet_stance_start_next is not None: 
		if quiet_stance_start_next < data_end - sample_rate*2: # if jump is within 2sec of last possible value, then jump end = next quite stance
			jump_markers['Jump End'] = jump_markers['Jump End'].fillna(quiet_stance_start_next) 
			jump_marks = {'Quite Stance': quiet_stance_start_next}
			jump_markers = jump_markers.append(jump_marks, ignore_index=True)
			cmj_etc()# next jump is calculated
			jump_markers['Eccentric Start'] = jump_markers['Eccentric Start'].fillna(eccentric_start)
			jump_markers['Concentric Start'] = jump_markers['Concentric Start'].fillna(concentric_start)
			jump_markers['Take Off'] = jump_markers['Take Off'].fillna(take_off)
			jump_kines = {'Additional weight (kg)': round(float(total_m - athlete_mass),4),'Avg Velocity (m/s)': v_take_off, 'Avg Force (N)': mean_force,'Avg Power (kg⋅m2⋅s−3)': power, 'hpo Auto': y_hpo_auto}
			jump_kinematics = jump_kinematics.append(jump_kines, ignore_index=True)
		else: 
			quiet_stance_start_next = data_end # if next quite stance = last possible value, add last jump end, and break 
			jump_markers['Jump End'] = jump_markers['Jump End'].fillna(data_end)
			break
	else:
		quiet_stance_start_next = data_end
		jump_markers['Jump End'] = jump_markers['Jump End'].fillna(data_end)


col1, col2 = st.columns(2, gap="small")
with col1: 
	jumps_loaded()

with col2: #number of jumps calculated is = number of indexes of jump_kinimatics (or markers)
	jumps_state = str(len(jump_kinematics.index)) + " Jumps were detected"
	jumps_found = len(jump_kinematics.index)
	st.success(jumps_state)

jump_markers = jump_markers.astype(int)
jump_markers.index = np.arange(1, len(jump_markers) + 1)
jump_kinematics.index = np.arange(1, len(jump_kinematics) + 1)
jump_markers["Additional weight (kg)"] = jump_kinematics["Additional weight (kg)"] # extra weight column is added for next calculations

def Weight_graph(Graph_selector):
	fig, ax = plt.subplots()
	graph_base = raw_graph_data[Graph_selector]
	ax.plot(graph_base.index, graph_base)
	# for loop iterates over each columns data, then uses each corresponding phase marker 
	for num,row in enumerate(graph_temp_df.iterrows()):
		jump_num = num+1
		if jump_num <= 1:
			ax.axvspan(graph_temp_df["Quite Stance"].iloc[num],graph_temp_df["Eccentric Start"].iloc[num],color='green', alpha=0.2, label="Quite Stance")
			ax.axvspan(graph_temp_df["Eccentric Start"].iloc[num],graph_temp_df["Concentric Start"].iloc[num],color='gray', alpha=0.2, label="Eccentric")
			ax.axvspan(graph_temp_df["Concentric Start"].iloc[num],graph_temp_df["Take Off"].iloc[num],color='red', alpha=0.2, label="Concentric")
		elif jump_num >= 2:
			ax.axvspan(graph_temp_df["Quite Stance"].iloc[num],graph_temp_df["Eccentric Start"].iloc[num],color='green', alpha=0.2)
			ax.axvspan(graph_temp_df["Eccentric Start"].iloc[num],graph_temp_df["Concentric Start"].iloc[num],color='gray', alpha=0.2)
			ax.axvspan(graph_temp_df["Concentric Start"].iloc[num],graph_temp_df["Take Off"].iloc[num],color='red', alpha=0.2)
	plt.title(str(Graph_selector) + " Extra kg", fontweight="bold")
	plt.xlabel("Sample rate", fontweight = "bold")
	plt.ylabel("Force (N)", fontweight = "bold")
	fig.patch.set_facecolor('#ffffcc')
	ax.set_facecolor('#e0ebeb')
	fig.legend()
	st.pyplot(fig)

def Graph_and_Chart_func():
	if Graph_and_Chart == 'Chart':
		st.write(raw_FP_data)
	elif Graph_and_Chart == 'Graph':
		st.line_chart(raw_FP_data, width=450, height=225)
	elif Graph_and_Chart == 'Chart and Graph':
		st.write(raw_FP_data)
		st.line_chart(raw_FP_data,width=450, height=225)

raw_graph_data["None"] = (raw_graph_data.loc[0])
first_none = raw_graph_data.pop('None')
raw_graph_data.insert(0,'None', first_none)

graph_select = list(raw_graph_data) # selectable graph is determined by the name of each columns weight 

col1, col2, col3= st.columns(3, gap="small")

with col1:
	del_below_thresh = st.radio("Remove bad jumps?", options=["Yes", "keep all"])
	if del_below_thresh == "Yes":
		for column in jump_kinematics:
			jump_markers['hpo Auto'] = jump_kinematics['hpo Auto']
			jump_kinematics = jump_kinematics.drop(jump_kinematics[jump_kinematics['hpo Auto'] < 0.025].index)
			jump_markers = jump_markers.drop(jump_markers[jump_markers['hpo Auto'] < 0.025].index)

with col2:
	Graph_and_Chart = st.selectbox("Would you like to view your raw data?", options=['no selection','Chart','Graph','Chart and Graph'])


with col3:
	Graph_selector = st.selectbox("Select processed graph", options=graph_select)

Graph_and_Chart_func()
if Graph_selector is "None":
	st.write("")
elif Graph_selector is not None:
	#This selects the row of data from jump_markers that matches the weight selected 
	graph_temp_df = jump_markers.loc[jump_markers['Additional weight (kg)'] == Graph_selector]
	graph_temp_df = pd.DataFrame(graph_temp_df)
	graph_temp_df.index = range(len(graph_temp_df.index))
	#Develop a graph function for the data set
	Weight_graph(Graph_selector)
del raw_graph_data["None"]


col1, col2, col3 = st.columns(3, gap= "small")

with col1:
	FVP_selector = st.radio("Which values would you like to use?", options= ["Highest value", "Average"])
with col2:
	extra_data_selector = st.radio("Would you like to see all data?", options=["No", "Yes"])
Final_data = pd.DataFrame()

with col3: 
	hpo_selector = st.radio("select hpo calculation", options=['Auto', 'Manual'])
	if hpo_selector == "Auto":
		y_hpo = jump_kinematics['hpo Auto'].mean()

	elif hpo_selector == "Manual":
		y_hpo = "n/a"

if hpo_selector == "Manual":
	with st.sidebar: 
		y_squat_height = st.number_input('Athletes vertical squat height (m)') 
		y_leg_length = st.number_input('Athletes take off height (m)') 

	# expanded window on a brief explination of hs and hpo 

		with st.expander("See explanation"):
			st.write("""Athletes vertical squat height (hs) : The vertical distance from ground to right leg’s greater trochanter during 90-degree knee-angle squat position (set with a square)""")
			st.write("""Athletes take off height: The distance from right greater trochanter to end of foot (during maximum plantar flexion)""")
	y_hpo = y_leg_length - y_squat_height


for column in raw_graph_data:
	final_temp_df = jump_kinematics.loc[jump_kinematics['Additional weight (kg)'] == column] 
	final_temp_df = pd.DataFrame(final_temp_df)
	final_temp_df.index = range(len(final_temp_df.index))
	if FVP_selector == "Average":
		average_value = pd.DataFrame(final_temp_df.sum() / float(len(final_temp_df)))
		mass = float(average_value.loc["Additional weight (kg)"])
		Final_data[mass] = average_value
	elif FVP_selector == "Highest value": 
		final_temp_df = final_temp_df.sort_values(by="Avg Power (kg⋅m2⋅s−3)", ascending=False)
		final_temp_df.index = range(len(final_temp_df.index))
		highest_value = final_temp_df.loc[0]
		mass = float(highest_value.loc["Additional weight (kg)"])
		Final_data[mass] = highest_value

Final_data = Final_data.transpose()


# Calculating final FVP 

y = Final_data['Avg Force (N)'].astype(float)
x = Final_data['Avg Velocity (m/s)'].astype(float)
slope_intercept = np.polyfit(x,y,1) # this returns the slope (Sfv), and the y intercept (Fo)
Sfv = round(slope_intercept[0], 3)
Sfv_perkg = round(Sfv/athlete_mass, 3)
Fo = round(slope_intercept[1], 3)
Fo_perkg = round(Fo/athlete_mass,3)
Vo = round(-Fo / Sfv, 3)
Pmax = round((Fo*Vo) / 4,3)
Pmax_perkg = round(Pmax/athlete_mass,3)
def SFVopt_calc():
	global Sfv_opt
	Z_N = -(g**6)*y_hpo**6 - 18*g**3 * y_hpo**5 * Pmax_perkg**2 - 54*y_hpo**4 * Pmax_perkg**4 + 6*math.sqrt(3) * math.sqrt(2*g**3 * y_hpo**9 * Pmax_perkg**6 + 27*y_hpo**8 * Pmax_perkg**8)
	#Because (-Z)**(1/3) isnt mathmatically possible, the number is instead returned as a complex number instead
	Z = -((-1*Z_N)**(1/3))
	#Creating the full formula for SFVopt (N=numerator, D=denomenator)
	F1 = -((g**2) / (3*Pmax_perkg))
	N2 = (-(g**4)*y_hpo**4 - 12 * g * y_hpo**3 * Pmax_perkg**2)
	D2 = (3*y_hpo**2 * Pmax_perkg * Z)
	F2 = (N2 / D2)
	N3 = Z
	D3 = (3*y_hpo**2 * Pmax_perkg)
	F3 = N3 / D3
	# The excel doc uses + F3 instead of - F3
	# This conflicts with the Samozino et al Appendicies formulaes 
	Sfv_opt = F1 - F2 - F3

SFVopt_calc()
y_ax = Sfv_opt * 0 + (2* math.sqrt(-Pmax_perkg * Sfv_opt)) * athlete_mass / athlete_mass 
x2 = 0
y2 = 0
x_ax = 4 * Pmax_perkg / (2 * math.sqrt(-Pmax_perkg * Sfv_opt))

bias = Sfv_perkg / Sfv_opt
calculated_values = pd.DataFrame(columns= ["Fo", "Fo/kg", "Sfv", "Sfv/kg", "Vo", "Pmax", "Pmax/kg"])
calc_appended = {"Fo": Fo, "Fo/kg": Fo_perkg, "Sfv": Sfv, "Sfv/kg":Sfv_perkg, "Vo":Vo, "Pmax":Pmax, "Pmax/kg":Pmax_perkg, "Sfvopt": Sfv_opt}
calculated_values = calculated_values.append(calc_appended, ignore_index=True)
if extra_data_selector == "Yes":
	st.table(calculated_values)
	st.table(jump_kinematics)
	st.table(jump_markers)

col1, col2, col3, col4 = st.columns(4, gap= "small")
with col1:
    Vo_words = round(Vo,2).astype(str) + "m/s"
    col1.metric("Vo", Vo_words)
with col2:
	Fo_word_per = round(Fo_perkg,2).astype(str) + "N/kg"
	col2.metric("Fo/kg", Fo_word_per)
with col3:
	Pmax_perkg_word = round(Pmax_perkg,2).astype(str) + "watts"
	col3.metric("Pmax/kg", Pmax_perkg_word)
with col4:
	Bias_words = (round(bias*100,2)).astype(str) + "%"
	if bias < 1:
		Bias_decision = ("Velocity deficient")
	elif bias >= 1:
		Bias_decision = ("Force deficient")
	col4.metric("Ratio of Optimal", Bias_words, Bias_decision)

if bias < 1:
	Bias_decision = ("Develop Force")
elif bias >= 1:
	Bias_decision = ("Develop Velocity")

fig1, ax = plt.subplots()
ax.axline((0, Fo_perkg), slope=Sfv_perkg, color='black', label='Actual Performance')
ax.axline((x_ax,0), (0, y_ax), color="blue", linestyle="dashed", label="Optimal Performance")
if Vo > x_ax:
	x_val = Vo
elif Vo <= x_ax:
	x_val = x_ax
if Fo_perkg > y_ax:
	y_val = Fo_perkg
elif Fo_perkg <= y_ax:
	y_val = y_ax
xmax = x_val + 3
ymax = y_val + 10
plt.xlim([0,xmax])
plt.ylim([0,ymax])
plt.xlabel("Velocity (m/s)", fontweight="bold")
plt.ylabel("Force (N/kg)", fontweight="bold")
plt.title("Force Velocity Profile", fontweight="bold")
fig1.patch.set_facecolor('#ffffcc')
ax.set_facecolor('#e0ebeb')
fig1.legend()
st.pyplot(fig1)

@st.cache
def convert_df_to_csv(df):
  # IMPORTANT: Cache the conversion to prevent computation on every rerun
  return df.to_csv().encode('utf-8')

with st.sidebar:
	athlete_name = st.text_input("Athletes name")
	col1, col2 = st.columns(2, gap= "small")
	with col1:
		file_name_kine = athlete_name + '_jump_kinematics.csv'
		st.download_button(label="Download jump kinematics",data=convert_df_to_csv(jump_kinematics),file_name=file_name_kine,mime='text/csv')
	with col2:
		file_name_prof = athlete_name + '_FVP_data.csv'
		st.download_button(label="Download Profiling Data",data=convert_df_to_csv(calculated_values),file_name=file_name_prof,mime='text/csv')


# Make a way to delete columns and a way to make the next jump still calculated
	# try and make jump detection more robust 
		# if unweighting phase is not detected & less than data_end						<<<<---------- what would make the unweighting phase incorrect? then use incorrect item for signal detection in an if statement 
			# find next quite_stance 

			# if this occurs, then the dataframe will get messed up 



# Make a concept map for the whole thing
	# then build a presentation with that
		# keeping in mind the protocols for how to conduct a FV profile



#####  OTHER IDEAS   ###############

# - could you plot a series of slopes for sfv (similar to a family of curves used in stats?)
