import matplotlib.pyplot as plt
import numpy as np

# stats
import scipy.stats as stats

# rosbag
import sqlite3
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

class BarChartWithStats:
	def __init__(self, ax,
				 title=None,
				 color_list=None,
				 alpha=0.8,
				 ylim=None,
				 title_fontsize=14,
				 label_fontsize=12,
				 ticklabel_fontsize=10,
				 legend_fontsize=10,
				 annotation_fontsize=12,
				 spine_width=1.5,
				 tick_width=1.5,
				 errorbar_linewidth=1.5,
				 bar_edge_width=1.0,
				 bar_group_spacing=0.8):
		self.ax = ax
		self.title = title
		self.color_list = color_list or plt.cm.tab10.colors
		self.alpha = alpha
		self.ylim = ylim

		# Font sizes
		self.title_fontsize = title_fontsize
		self.label_fontsize = label_fontsize
		self.ticklabel_fontsize = ticklabel_fontsize
		self.legend_fontsize = legend_fontsize
		self.annotation_fontsize = annotation_fontsize

		# Thicknesses
		self.spine_width = spine_width
		self.tick_width = tick_width
		self.errorbar_linewidth = errorbar_linewidth
		self.bar_edge_width = bar_edge_width

		self.bar_group_spacing = bar_group_spacing

		self._style_axes()

	def _style_axes(self):
		for spine in self.ax.spines.values():
			spine.set_linewidth(self.spine_width)
		self.ax.tick_params(axis='both', width=self.tick_width, length=5)

	def plot(self, data_nested, group_labels, condition_labels, comparisons=None, p_values=None, 
		ebar = True,ylabel='Data',legendloc='best',legendncol=1, legendoffset = None, comparisonoffset = 0.0):
		"""
		Parameters:
		- data_nested: list of conditions, each containing list of (group_idx, [samples]).
		"""
		n_conditions = len(data_nested)
		all_group_indices = set()
		for cond in data_nested:
			for group_idx, _ in cond:
				all_group_indices.add(group_idx)
		all_group_indices = sorted(list(all_group_indices))
		n_groups = len(all_group_indices)

		group_idx_map = {g: i for i, g in enumerate(all_group_indices)}

		# Initialize mean and std matrices (groups x conditions)
		mean_data = np.zeros((n_groups, n_conditions))
		std_data = np.zeros((n_groups, n_conditions))
		counts = np.zeros((n_groups, n_conditions))  # for optional weighting/debug

		for c_idx, cond in enumerate(data_nested):
			for group_idx, samples in cond:
				g_idx = group_idx_map[group_idx]
				samples_arr = np.array(samples)
				if len(samples_arr) > 0:
					mean_data[g_idx, c_idx] = np.mean(samples_arr)
					std_data[g_idx, c_idx] = np.std(samples_arr)
					counts[g_idx, c_idx] = len(samples_arr)

		index = np.arange(n_groups)
		bar_width = self.bar_group_spacing / n_conditions

		for i in range(n_conditions):
			if ebar:
				self.ax.bar(
					index + i * bar_width,
					mean_data[:, i],
					bar_width,
					yerr=std_data[:, i],
					label=condition_labels[i],
					color=self.color_list[i % len(self.color_list)],
					alpha=self.alpha,
					capsize=5,
					error_kw={'elinewidth': self.errorbar_linewidth, 'capthick': self.errorbar_linewidth},
					edgecolor='black',
					linewidth=self.bar_edge_width
				)
			else:
				self.ax.bar(
					index + i * bar_width,
					mean_data[:, i],
					bar_width,
					label=condition_labels[i],
					color=self.color_list[i % len(self.color_list)],
					alpha=self.alpha,
					capsize=5,
					edgecolor='black',
					linewidth=self.bar_edge_width
				)

		self.ax.set_ylabel(ylabel, fontsize=self.label_fontsize)
		self.ax.set_xticks(index + bar_width * (n_conditions - 1) / 2)
		self.ax.set_xticklabels([group_labels[g] for g in all_group_indices], fontsize=self.ticklabel_fontsize)

		if self.title:
			self.ax.set_title(self.title, fontsize=self.title_fontsize)
		if self.ylim:
			self.ax.set_ylim(self.ylim)

		self.ax.legend(fontsize=self.legend_fontsize,loc=legendloc, ncol=legendncol, bbox_to_anchor=legendoffset)

		if comparisons and p_values:
			for (cond1, grp1, cond2, grp2), p in zip(comparisons, p_values):

				if p < 0.001:
					text = '***'
				elif p < 0.01:
					text = '**'
				elif p < 0.05:
					text = '*'
				else:
					text = 'ns'
					continue

				g_idx1 = group_idx_map[grp1]
				g_idx2 = group_idx_map[grp2]
				bar1 = index[g_idx1] + cond1 * bar_width
				bar2 = index[g_idx2] + cond2 * bar_width
				y1 = mean_data[g_idx1, cond1] + std_data[g_idx1, cond1] + comparisonoffset
				y2 = mean_data[g_idx2, cond2] + std_data[g_idx2, cond2] + comparisonoffset
				max_y = max(y1, y2)
				line_y = max_y * 1.05

				self.ax.plot([bar1, bar1, bar2, bar2],
							 [line_y, line_y + 0.02 * max_y, line_y + 0.02 * max_y, line_y],
							 lw=self.spine_width, c='black')

				

				self.ax.text(
					(bar1 + bar2) / 2,
					line_y + 0.02 * max_y,
					text,
					ha='center',
					va='bottom',
					fontsize=self.annotation_fontsize
				)


def compute_p_values(data, comparisons):
	"""
	Compute p-values (Mann-Whitney U) for specified comparison pairs.

	Parameters:
	- data: list of list of (group_idx, [values]) pairs
	- comparisons: list of (cond1_idx, group1_idx, cond2_idx, group2_idx)

	Returns:
	- p_values: list of p-values corresponding to the comparisons list (None if insufficient data)
	"""
	p_values = []
	for cond1_idx, group1_idx, cond2_idx, group2_idx in comparisons:
		# Convert list of tuples to dict for access
		data1_dict = dict(data[cond1_idx])
		data2_dict = dict(data[cond2_idx])
		data1 = data1_dict.get(group1_idx, [])
		data2 = data2_dict.get(group2_idx, [])
		
		# Check if both groups have at least one data point
		if len(data1) >= 1 and len(data2) >= 1:
			stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
		else:
			p_value = None  # Not enough data
		
		p_values.append(p_value)
	
	return p_values


def extract_rosbag_data(path: str, topic: str):
    """
    Extracts data and timestamps from a ROS 2 bag file for a given topic.

    Args:
    - path (str): Path to the rosbag directory.
    - topic (str): The topic name to extract data from.

    Returns:
    - data_array (numpy.ndarray): The data extracted from the topic.
    - time_array (numpy.ndarray): The corresponding timestamps (in nanoseconds).
    """
    bagpath = Path(path)

    with AnyReader([bagpath]) as reader:
        connections = [x for x in reader.connections if x.topic == topic]

        if not connections:
            print(f"Topic '{topic}' not found in the rosbag.")
            return np.array([]), np.array([])

        all_data = []
        all_timestamps = []

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            all_data.append(msg.data)
            all_timestamps.append(timestamp)  # This is in nanoseconds (int)

        data_array = np.array(all_data)
        time_array = np.array(all_timestamps, dtype=np.uint64)

    return time_array, data_array



# ==== Example usage ====
if __name__ == '__main__':
	global_params = {
		'bar_group_spacing': 0.7,
		'color_list': None,
		'title_fontsize': 16,
		'label_fontsize': 14,
		'ticklabel_fontsize': 12,
		'legend_fontsize': 12,
		'annotation_fontsize': 14,
		'spine_width': 2,
		'tick_width': 2,
		'errorbar_linewidth': 2,
		'bar_edge_width': 1.5,
		'alpha': 0.8,
		'ylim': None
	}

	fig, ax = plt.subplots(figsize=(8, 6))

	# Explicit group indices
	data_example = [
		[  # Condition 0
			(0, [5, 6, 5.5]),
			(1, [7, 7.5]),
			(2, [6, 6.3, 6.1, 6.5])
		],
		[  # Condition 1
			(0, [6, 6.5, 6.2, 6.8]),
			(1, [0, 0]),
			(2, [7, 7.5, 7.2])
		],
		[  # Condition 2
			(0, [5.5, 5.7]),
			(1,[2.1,1.1,0.8]),
			(2, [7.8, 8.1, 7.9])  # Group 2 missing here → auto-handled
		]
	]

	group_labels = {0: 'Group1', 1: 'Group2', 2: 'Group3'}
	condition_labels = ['CondA', 'CondB', 'CondC']
	comparisons = [(0, 0, 1, 0), (0, 1, 1, 1), (1, 2, 2, 1)]
	p_values = [0.04, 0.002, 0.2]

	params = global_params.copy()
	params['title'] = 'Explicit Group Indices Example'
	params['color_list'] = ['#4daf4a', '#e41a1c', '#377eb8']

	chart = BarChartWithStats(ax, **params)
	chart.plot(data_example, group_labels, condition_labels, comparisons=comparisons, p_values=p_values)

	plt.tight_layout()
	plt.show()
