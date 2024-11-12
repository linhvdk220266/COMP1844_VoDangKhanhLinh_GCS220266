import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

stations = pd.read_csv('stations.csv')
stations2 = pd.read_csv('Station dataset.csv')
connections = pd.read_csv('lines2.csv')
routes = pd.read_csv('routes.csv')

piccadilly_from = stations2['Station'][0:5]
piccadilly_from_df = pd.DataFrame({'From Station': piccadilly_from})

piccadilly_to = stations2['Station'][1:6]
piccadilly_to_df = pd.DataFrame({'To Station': piccadilly_to}).reset_index()

piccadilly_line = pd.concat([piccadilly_from_df, piccadilly_to_df], axis=1,)
piccadilly_line = piccadilly_line.loc[:,['From Station', 'To Station']]

bakerloo_from = stations2['Station'][6:11]
bakerloo_from_df = pd.DataFrame({'From Station': bakerloo_from}).reset_index()

bakerloo_to = stations2['Station'][7:12]
bakerloo_to_df = pd.DataFrame({'To Station': bakerloo_to}).reset_index()

bakerloo_line = pd.concat([bakerloo_from_df, bakerloo_to_df], axis=1,)
bakerloo_line = bakerloo_line.loc[:,['From Station', 'To Station']]

central_from = stations2['Station'][12:17]
central_from_df = pd.DataFrame({'From Station': central_from}).reset_index()

central_to = stations2['Station'][13:18]
central_to_df = pd.DataFrame({'To Station': central_to}).reset_index()

central_line = pd.concat([central_from_df, central_to_df], axis=1,)
central_line = central_line.loc[:,['From Station', 'To Station']]

northern_from = stations2['Station'][18:23]
northern_from_df = pd.DataFrame({'From Station': northern_from}).reset_index()

northern_to = stations2['Station'][19:]
northern_to_df = pd.DataFrame({'To Station': northern_to}).reset_index()

northern_line = pd.concat([northern_from_df, northern_to_df], axis=1,)
northern_line = northern_line.loc[:,['From Station', 'To Station']]

result = pd.merge(stations2, stations, left_on='Station', right_on='name', how='left')
result.loc[result['Station'] == "Piccadilly Circus", 'latitude'] = 51.5099
result.loc[result['Station'] == "Piccadilly Circus", 'longitude'] = -0.1343

# final_result = result.loc[result['ID','Line','Station','latitude','longitude' ]]
final_result = result.loc[:, ['ID', 'Line', 'Station', 'latitude', 'longitude']]

# Function to break long station names into multiple lines
def break_long_string(label, max_words=2):
    """Breaks a string into multiple lines if it contains more than max_words."""
    words = label.split()
    if len(words) >= max_words:
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

# Add edges to the graph with color mapping
for df, line_name in zip([piccadilly_line, bakerloo_line, central_line, northern_line], line_colors.keys()):
    color = line_colors[line_name]
    for _, row in df.iterrows():
        G.add_edge(row['From Station'], row['To Station'], color=color)


fig, ax = plt.subplots(figsize=(14, 9))

# Convert positions to map projection coordinates
pos = {station: (lon, lat) for station, (lon, lat) in nx.get_node_attributes(G, 'pos').items()}

# Offset positions for labels to avoid overlapping with nodes
label_pos = {station: (lon + 0.001, lat + 0.0016) for station, (lon, lat) in pos.items()}

# Adjust the label position of a specific node by name
specific_node = "Covent Garden" 
if specific_node in label_pos:
    label_pos[specific_node] = (label_pos[specific_node][0] + 0.002, label_pos[specific_node][1] - 0.001)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=300, node_color="green", ax=ax)

# Draw edges with assigned colors
edges = G.edges(data=True)
colors = [edge[2]['color'] for edge in edges]  # Get colors for each edge
nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, ax=ax, arrows=False, edge_color=colors)

# Prepare labels with line breaks using the correct function
label_dict = {station: break_long_string(station, max_words=2) for station in G.nodes()}

# Draw labels with adjusted positions
nx.draw_networkx_labels(G, label_pos, labels=label_dict, font_size=8, font_color="black", ax=ax)

# Add a legend for each line
legend_patches = [mpatches.Patch(color=color, label=line_name) for line_name, color in line_colors.items()]
ax.legend(handles=legend_patches, title="Transport Lines", loc="lower left")

plt.title("Map of Public Transport Stations")
plt.margins(0.05)
plt.show()
