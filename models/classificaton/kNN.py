import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')



def L2norm(X1, X2):		#Function to calculate L2 Norm
	distance = 0
	for i in range(len(X1)):
		distance += (X1[i] - X2[i])**2

	distance = distance**0.5
	return distance


def centroid(X):		#Function to calculate Centroid
	centroid = X[0]
	n = len(X)

	for i in range(len(X)):
		for j in range(len(X[i])):
			centroid[j] += X[i][j]

	for i in range(len(centroid)):
		centroid[i] = centroid[i]/n

	return centroid


#######################################################################

# Load the data in Dataframe
df = pd.read_csv("cancer.csv")

# Chose relevant columns
X = df.iloc[:,2:32]

# Convert Dataframe to Array
X = X.values


# Initialize Centers Randomly
center1 = np.random.randint(low=1000, size=(len(X[0]),))
center2 = np.random.randint(low=1000, size=(len(X[0]),))

# Initialize dummy variables to keep track of previous centers
center1_prev = np.random.randint(low=1000, size=(len(X[0]),))
center2_prev = np.random.randint(low=1000, size=(len(X[0]),))

# Dictionary to store the cluster
label_dict = {}
label_dict[0] = []
label_dict[1] = []

while L2norm(center1, center1_prev) > 1e-10 and L2norm(center2, center2_prev) > 1e-10:

	for i in range(len(X)):
		d1 = L2norm(X[i], center1)
		d2 = L2norm(X[i], center2)

		if d1 < d2:
			label_dict[0].append(X[i])
		else:
			label_dict[1].append(X[i])

	if len(label_dict[0]) == 0 or len(label_dict[1]) == 0:
		center1 = np.random.randint(low=1000, size=(len(X[0]),))
		center2 = np.random.randint(low=1000, size=(len(X[0]),))
		continue


	center1_prev = center1
	center2_prev = center2

	center1 = centroid(label_dict[0])
	center2 = centroid(label_dict[1])

	if L2norm(center1, center1_prev) > 1e-10 and L2norm(center2, center2_prev) > 1e-10:
		label_dict[0] = []
		label_dict[1] = []

n_cluster1 = len(label_dict[0])
n_cluster2 = len(label_dict[1])

print("Number of points in Cluster 1: {}".format(n_cluster1))
print("Number of points in Cluster 2: {}".format(n_cluster2))


# Plotting Section

ind_x = list(df.iloc[:,2:32].columns).index("radius_mean")
ind_y = list(df.iloc[:,2:32].columns).index("texture_mean")


x0 = []
y0 = []

x1 = []
y1 = []

for i in range(len(label_dict[0])):
	x0.append(label_dict[0][i][ind_x])
	y0.append(label_dict[0][i][ind_y])

for i in range(len(label_dict[1])):
	x1.append(label_dict[1][i][ind_x])
	y1.append(label_dict[1][i][ind_y])

plt.scatter(x0, y0, marker='o', color='red', label="Label 0, n="+str(n_cluster1))
plt.scatter(x1, y1, marker='o', color='blue', label="Label 1, n="+str(n_cluster2))

plt.xlabel("radius_mean")
plt.ylabel("texture_mean")

plt.legend()

plt.title("kNN Clusters for Cancer Dataset")

plt.show()