import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import radians, sin, cos, sqrt, atan2
import matplotlib.lines as mlines

#### TASK 1

# Create a graph
G = nx.Graph()

# Define stations (nodes) and their connections (edges with real distances in kilometers)
stations = {
    'Hyde Park Corner': {'Green Park': 0.8},
    'Green Park': {'Piccadilly Circus': 0.7},
    'Piccadilly Circus': {'Leicester Square': 0.5},
    'Leicester Square': {'Covent Garden': 0.3},
    'Covent Garden': {'Holborn': 0.6}
}

# Add edges to the graph with distances as weights
for station, connections in stations.items():
    for connected_station, distance in connections.items():
        G.add_edge(station, connected_station, weight=distance)

# Define positions of nodes for better visualization
pos = {
    'Hyde Park Corner': (-0.7, 0.2),
    'Green Park': (0.2, 1),
    'Piccadilly Circus': (1.3, 1),
    'Leicester Square': (2.1, 1),
    'Covent Garden': (2.6, 1.5),
    'Holborn': (3.2, 2.2)
}

# Draw the graph
plt.figure(figsize=(13, 6))

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=400, node_color='blue')

# Draw edges
nx.draw_networkx_edges(G, pos, width=2, edge_color='blue')

ax = plt.gca()
for station, position in pos.items():
    x, y = position
    ax.text(x - 0.1, y + 0.18, station, fontsize=8, ha='center', va='center', rotation_mode='anchor')

# Draw edge labels (distances between stations)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

# Add a title and display the plot
plt.title("Public transport network of London (in km)")
plt.plot([], [], color='blue', label='Piccadilly Line')
plt.legend(loc='lower right')
plt.axis('off')  # Hide axis
plt.show()

###TASK 2

stations = pd.read_csv('stations.csv')
stations2 = pd.read_csv('Station dataset.csv')
connections = pd.read_csv('lines2.csv')
routes = pd.read_csv('routes.csv')

piccadilly_from = stations2['Station'][0:6]
piccadilly_from_df = pd.DataFrame({'From Station': piccadilly_from})

piccadilly_to = stations2['Station'][1:7]
piccadilly_to_df = pd.DataFrame({'To Station': piccadilly_to}).reset_index()

piccadilly_line = pd.concat([piccadilly_from_df, piccadilly_to_df], axis=1,)
piccadilly_line = piccadilly_line.loc[:,['From Station', 'To Station']]
piccadilly_line

bakerloo_from = stations2['Station'][7:12]
bakerloo_from_df = pd.DataFrame({'From Station': bakerloo_from}).reset_index()

bakerloo_to = stations2['Station'][8:13]
bakerloo_to_df = pd.DataFrame({'To Station': bakerloo_to}).reset_index()

bakerloo_line = pd.concat([bakerloo_from_df, bakerloo_to_df], axis=1,)
bakerloo_line = bakerloo_line.loc[:,['From Station', 'To Station']]
bakerloo_line

central_from = stations2['Station'][13:19]
central_from_df = pd.DataFrame({'From Station': central_from}).reset_index()

central_to = stations2['Station'][14:20]
central_to_df = pd.DataFrame({'To Station': central_to}).reset_index()

central_line = pd.concat([central_from_df, central_to_df], axis=1,)
central_line = central_line.loc[:,['From Station', 'To Station']]
central_line

northern_from = stations2['Station'][20:25]
northern_from_df = pd.DataFrame({'From Station': northern_from}).reset_index()

northern_to = stations2['Station'][21:]
northern_to_df = pd.DataFrame({'To Station': northern_to}).reset_index()

northern_line = pd.concat([northern_from_df, northern_to_df], axis=1,)
northern_line = northern_line.loc[:,['From Station', 'To Station']]
northern_line

result = pd.merge(stations2, stations, left_on='Station', right_on='name', how='left')
result

result.isnull().sum()

result.loc[result['Station'] == "Piccadilly Circus", 'latitude'] = 51.5099
result.loc[result['Station'] == "Piccadilly Circus", 'longitude'] = -0.1343

# final_result = result.loc[result['ID','Line','Station','latitude','longitude' ]]
final_result = result.loc[:, ['ID', 'Line', 'Station', 'latitude', 'longitude']]
final_result

# Function to break long station names into multiple lines
def break_long_string(label, max_words=2):
    """Breaks a string into multiple lines if it contains more than max_words."""
    words = label.split()
    if len(words) >= max_words and label != "Leicester Square" and label != "Covent Garden" and label != "Charing Cross":
        return '\n'.join(words)  # Join words with newline character
    return label

G = nx.DiGraph()

# Add nodes to the graph with positions
for _, row in final_result.iterrows():
    G.add_node(row["Station"], pos=(row["longitude"], row["latitude"]))

# Define colors for each line
line_colors = {
    'Piccadilly': 'blue',
    'Bakerloo': 'brown',
    'Central': 'red',
    'Northern': 'black'
}

interchange_station = {"Oxford Circus", "Tottenham Court Road", "Holborn", "Leicester Square", "Piccadilly Circus", "Charing Cross"}

# Add edges to the graph with color mapping
for df, line_name in zip([piccadilly_line, bakerloo_line, central_line, northern_line], line_colors.keys()):
    color = line_colors[line_name]
    for _, row in df.iterrows():
        G.add_edge(row['From Station'], row['To Station'], color=color)

# Assign colors to nodes
node_colors = []
for node in G.nodes():
    if node in interchange_station:
        node_colors.append("white")  # Interchange station in white
    else:
        # Check each line to assign the line color for the node
        for line_name, line_df in zip(line_colors.keys(), [piccadilly_line, bakerloo_line, central_line, northern_line]):
            if node in line_df.values:
                node_colors.append(line_colors[line_name])
                break

fig, ax = plt.subplots(figsize=(14, 9))

# Convert positions to map projection coordinates
pos = {station: (lon, lat) for station, (lon, lat) in nx.get_node_attributes(G, 'pos').items()}

# Offset positions for labels to avoid overlapping with nodes
label_pos = {station: (lon + 0.001, lat + 0.0016) for station, (lon, lat) in pos.items()}

# Adjust the label position of a specific node by name
specific_node = "Covent Garden"
if specific_node in label_pos:
    label_pos[specific_node] = (label_pos[specific_node][0] + 0.0033, label_pos[specific_node][1] - 0.0017)

new_specific_node = "Leicester Square"
if new_specific_node in label_pos:
    label_pos[new_specific_node] = (label_pos[new_specific_node][0] + 0.0035, label_pos[new_specific_node][1] - 0.0017)

other_specific_node = "Charing Cross"
if other_specific_node in label_pos:
    label_pos[other_specific_node] = (label_pos[other_specific_node][0] + 0.0033, label_pos[other_specific_node][1] - 0.0014)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=120, node_color=node_colors, ax=ax, edgecolors="black", linewidths=1)

# Draw edges with assigned colors
edges = G.edges(data=True)
colors = [edge[2]['color'] for edge in edges]  # Get colors for each edge
nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, ax=ax, arrows=False, edge_color=colors)

# Prepare labels with line breaks using the correct function
label_dict = {station: break_long_string(station, max_words=2) for station in G.nodes()}

# Draw labels with adjusted positions
nx.draw_networkx_labels(G, label_pos, labels=label_dict, font_size=8, font_color="black", ax=ax)

# Add a legend for each line
# legend_patches = [mpatches.Patch(color=color, label=line_name) for line_name, color in line_colors.items()]
# ax.legend(handles=legend_patches, title="Transport Lines", loc="lower left")

# Add a legend with line styles for each line
legend_lines = [mlines.Line2D([], [], color=color, linewidth=2, label=line_name) for line_name, color in line_colors.items()]
ax.legend(handles=legend_lines, title="Transport Lines", loc="lower left")

plt.title("Map of Public Transport Stations")
plt.margins(0.05)

# Function to calculate Haversine distance between two points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c  # Distance in kilometers

# Add weighted edges to the graph
for df, line_name in zip([piccadilly_line, bakerloo_line, central_line, northern_line], line_colors.keys()):
    color = line_colors[line_name]
    for _, row in df.iterrows():
        from_station = row['From Station']
        to_station = row['To Station']
        
        # Retrieve coordinates for both stations
        from_coords = final_result.loc[final_result['Station'] == from_station, ['latitude', 'longitude']].values[0]
        to_coords = final_result.loc[final_result['Station'] == to_station, ['latitude', 'longitude']].values[0]
        
        # Calculate distance
        distance = haversine(from_coords[0], from_coords[1], to_coords[0], to_coords[1])
        
        # Add edge with color and distance as weight
        G.add_edge(from_station, to_station, color=color, weight=distance)

# Visualization code (unchanged)
fig, ax = plt.subplots(figsize=(14, 9))

# Draw nodes and edges with distances as weights
pos = {station: (lon, lat) for station, (lon, lat) in nx.get_node_attributes(G, 'pos').items()}
# nx.draw_networkx_nodes(G, pos, node_size=120, node_color="white", ax=ax, edgecolors="black", linewidths=1.5)
nx.draw_networkx_nodes(G, pos, node_size=120, node_color=node_colors, ax=ax, edgecolors="black", linewidths=1)
edges = G.edges(data=True)
colors = [edge[2]['color'] for edge in edges]
nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, ax=ax, arrows=False, edge_color=colors)

# Show distances on edges
edge_labels = {(u, v): f"{data['weight']:.1f}" for u, v, data in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

# Draw labels and legend
label_dict = {station: break_long_string(station, max_words=2) for station in G.nodes()}
nx.draw_networkx_labels(G, label_pos, labels=label_dict, font_size=8, font_color="black", ax=ax)

# Add a legend for each line 
legend_lines = [mlines.Line2D([], [], color=color, linewidth=2, label=line_name) for line_name, color in line_colors.items()]
ax.legend(handles=legend_lines, title="Transport Lines", loc="lower left")

plt.title("Map of Public Transport Stations with Distances (in km)")
plt.margins(0.09)
plt.show()


###TASK 3
# Extract all distances from the graph edges
distances = [data['weight'] for _, _, data in G.edges(data=True)]

# Calculate the total length of the network
total_length = sum(distances)

# Calculate the average distance between stations
average_distance = np.mean(distances)

# Calculate the standard deviation of distances between stations
std_dev_distance = np.std(distances)

# Display the results with formatted values
print("Total length of the transport network: {:.2f} km".format(total_length))
print("Average distance between stations: {:.2f} km".format(average_distance))
print("Standard deviation of distances between stations: {:.2f} km".format(std_dev_distance))
