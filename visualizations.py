# # Module 4 - Programming Assignment - k Nearest Neighbors

from typing import List, Tuple
import matplotlib.pyplot as plt
from tabulate import tabulate
import knn

# ## Plotting dataset
# `plot_dataset` plots the data set as a parallel coordinates plot. 
# Each line represents an observation in the data set. 
# The color of the line is given by the target value of the observation. 
# The x-axis represents the features and the y-axis represents the normalized values of the features.
# 
# * **data** List[List]: list containing features and labels of data point instances.
# 
# **returns** Tuple: returns figure and axis of the plot.

def plot_dataset(data: List[List]) -> None:
    # Number of columns in the data
    num_vars = len(data[0])
    # Create a colormap for the lines
    cmap = plt.get_cmap("viridis")
    # Get target values for color mapping
    target_values = [row[-1] for row in data]
    # Generate list of colors
    colors = [cmap(value) for value in target_values]
    # Create a figure and axis
    fig, ax = plt.subplots(dpi=350, figsize=(8, 4))
    ax.set_title('Normalized Properties of Concrete', fontsize=8, color='white')
    # Color properties 
    fig.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    ax.spines['bottom'].set_color('darkgray')
    ax.spines['top'].set_color('darkgray')
    ax.spines['right'].set_color('darkgray')
    ax.spines['left'].set_color('darkgray')
    ax.tick_params(axis='both', which='major', labelsize=8, color='darkgray')
    # Plot each point in database as an individual line
    for i in range(0, len(data), 1):
        ax.plot(range(num_vars), data[i], color=colors[i], linewidth=0.4, alpha=0.2)
    # Set the xticks to properties of concrete
    ax.set_xticks(range(num_vars))
    ax.set_xticklabels(['Cement', 'Slag', 'Ash', 'Water', 'Superplasticizer', 'Coarse\nAggregate', 'Fine\nAggregate', 'Age', 'Compressive\nStrength'], rotation=45, fontsize=8, color='white')
    for label in ax.get_yticklabels():
        label.set_color('white')
    # Add vertical axis lines for each variable
    for x in ax.get_xticks():
        ax.axvline(x, color='darkgray', linewidth=0.4)
    # Add a colorbar for the target values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0,1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, )
    cbar.set_label('Compressive Strength', size=8, color='white')
    cbar.ax.tick_params(labelsize=8, color='darkgray')
    for label in cbar.ax.get_yticklabels():
        label.set_color('white')
    return (fig, ax)


# ## Update plot
# `update_plot` updates the parallel coordinates plot with the query point and k-nearest neighbors.
# The query point is plotted as a red line and the k-nearest neighbors are plotted as white lines.
#
# * **fig** Figure: figure of the plot.
# * **ax** Axis: axis of the plot.
# * **min_max_mean** List[Tuple[float, float, float]]: list containing min, max, and mean values of each feature.
# * **k_nearest_neighbors** List[List]: list containing k-nearest neighbors of the query point.
# * **query** List[float]: list containing features and labels of the query point instance.
#
# **returns** Tuple: returns figure and axis of the plot.

def update_plot(fig, ax, min_max_mean: List[Tuple], k_nearest_neighbors: List[List], query: List) -> Tuple:
    # Extract the features of the k-nearest neighbors
    neighbors = []
    for i in range(len(k_nearest_neighbors)):
        neighbors.append(k_nearest_neighbors[i][1])
    # Normalize the data for visualization
    neighbors= knn.normalize(neighbors, min_max_mean)
    query = knn.normalize([query], min_max_mean)
    # Flatten query point from 2D to 1D array 
    query = [item for sublist in query for item in sublist]
    # Number of columns in the data
    num_vars = len(query)
    # Plot the k nearest neighbors as white lines
    for neighbor in neighbors:
        ax.plot(range(num_vars), neighbor, color='white', linewidth=1.0)
    # Plot the query point as a red line
    ax.plot(range(num_vars), query, color='red', linewidth=1.5)
    return (fig, ax)


# ## Tabulate data
# `tabulate_data` tabulates the query point and k-nearest neighbors in a table.
# The table contains the features and target values of the query point and k-nearest neighbors.
#
# * **query** List[float]: list containing features and labels of the query point instance.
# * **k_nearest_neighbors** List[List]: list containing k-nearest neighbors of the query point.
#
# **returns** str: returns the table as a string.

def tabulate_data(query: List[List], k_nearest_neighbors: List[List]) -> str:
    # Extract the features of the k-nearest neighbors and add a label to each neighbor
    neighbors = []
    for i in range(len(k_nearest_neighbors)):
        neighbors.append(k_nearest_neighbors[i][1])
        neighbors[i].insert(0, 'k' + str(i+1))
    headers = ['', 'Cement', 'Slag', 'Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age', 'Compressive Strength']
    # Convert the data to a table using tabulate
    query = ['query'] + query
    table_data = [query] + neighbors
    table = tabulate(table_data, headers=headers, tablefmt='pipe', floatfmt=".3f")
    return table


# ## Proximity plot
# `proximity_plot` plots the query point and k-nearest neighbors on a polar axis.
# The distance of the k-nearest neighbors from the query point is represented by the radius of the polar plot.
# The color and size of the k-nearest neighbors are proportional to their distance from the query point.
#
# * **k_nearest_neighbors** List[List]: list containing k-nearest neighbors of the query point.
#
# **returns** Tuple: returns figure and axis of the plot.

def proximity_plot(k_nearest_neighbors: List[List]) -> None:
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, dpi=350, figsize=(8, 8))
    fig.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    # Create a list of thetas for the k-nearest neighbors
    thetas = []
    # Create a list of scaled and shifted radii for the k-nearest neighbors for visualization
    radii = []
    # Create a list of colors for the k-nearest neighbors
    colors = []
    # Create a list of areas for the k-nearest neighbors for visualization
    areas = []
    for i in range(len(k_nearest_neighbors)):
        thetas.append(i * 2 * 3.14159 / len(k_nearest_neighbors))
        radii.append(k_nearest_neighbors[i][0]*3.5 - (k_nearest_neighbors[0][0]*3))
        colors.append(k_nearest_neighbors[i][0])
        areas.append(70000*(1/(1+radii[i])))
    # Plot the k-nearest neighbors
    ax.scatter(thetas, radii, c=colors, norm = 'log', s = areas, alpha = 0.75, edgecolors='white', linewidth=0.5, cmap='Greys') # viridis
    # Plot the query point
    ax.scatter(0, 0, color='red', s=200, label='Query')
    # Add labels and plot lines for the k-nearest neighbors
    for i in range(len(k_nearest_neighbors)):
        # Add label for neighbor point
        ax.text(thetas[i], radii[i], '        k' + str(i+1), fontsize=10, color='white')
        # Draw a line from the query point to the neighbor point
        ax.plot([0, thetas[i]], [0, radii[i]], color='white', linewidth=1)
    # Modify axis properties
    ax.set_title('Proximity of Neighbors to Query Point', fontsize=10, color='white', weight='bold')
    ax.set_rticks([])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)
    ax.set_rlim(top=radii[-1]*1.3)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    # Add a text box
    textstr = 'Query point is shown as a red dot. The grayscale color and size of each neighbor is proportional to its distance\n from the query point. The closer neighbor is to the query point, the lighter is its color and the larger is its size.'
    ax.text(0.00, 0.98, textstr, transform=ax.transAxes, fontsize=8, color='white', verticalalignment='top')
    # Save the figure as an image
    fig.savefig('proximity_fig.png')
    return None