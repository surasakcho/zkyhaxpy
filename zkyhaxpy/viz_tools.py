import numpy as np

def generate_histogram_string(data, bins=10, min_val=None, max_val=None, marker='*', resolution=50, orientation='vertical', label_position='origin'):
    if min_val is None:
        min_val = min(data)
    if max_val is None:
        max_val = max(data)
    
    hist, bin_edges = np.histogram(data, bins=bins, range=(min_val, max_val))
    max_count = max(hist)
    
    # Determine the maximum label length
    max_label_length = max(len(f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}") for i in range(len(bin_edges)-1))
    
    histogram_str = ""
    if orientation == 'vertical':
        for i, count in enumerate(hist):
            bar_length = int((count / max_count) * resolution)
            bin_label = f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}"
            if label_position == 'origin':
                histogram_str += f"{bin_label.rjust(max_label_length)}: {marker * bar_length}\n"
            else:
                histogram_str += f"{bin_label}: {marker * bar_length}\n"
    elif orientation == 'horizontal':
        for i, count in enumerate(hist):
            bar_length = int((count / max_count) * resolution)
            bin_label = f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}"
            if label_position == 'origin':
                histogram_str += f"{bin_label.rjust(max_label_length)}: {marker * bar_length}\n"
            else:
                histogram_str += f"{marker * bar_length} :{bin_label}\n"
    
    min_val = np.min(data)
    max_val = np.max(data)
    median_val = np.median(data)
    p25_val = np.percentile(data, 25)
    p75_val = np.percentile(data, 75)
    
    stats_str = f"Min: {min_val:.2f}, Max: {max_val:.2f}, Median: {median_val:.2f}, p25: {p25_val:.2f}, p75: {p75_val:.2f}"
    
    return histogram_str, stats_str
