import numpy as np
import torch
import os

import sys
import os
sys.path.append('/Users/moshetannenbaum/bindset 4/bindsnet')

from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, AdaptiveLIFNodes
from bindsnet.network.topology import MulticompartmentConnection
from bindsnet.network.topology_features import Weight
from matplotlib.animation import FuncAnimation


# Class for a spiking neural network that uses STDP and Q-Learning to learn
class STDP_Q_Learning(Network):
  def __init__(self,
               in_size: int,  # Number of association (input) neurons
               out_size: int,  # Number of motor control neurons
               w_exc_out: np.ndarray,  # Association to motor control weights
               w_out_out: np.ndarray,  # Motor control to motor control weights
               alpha: float,  # Q-Learning Learning rate (Q-table learning)
               gamma: float,  # Q-Learning discount factor
               num_actions: int,  # Number of possible actions
               wmin: float,  # Minimum synaptic weight value
               wmax: float,  # Maximum synaptic weight value
               decay: float,  # Weight decay factor
               lr: float,  # Learning rate for STDP (synaptic learning)
               hyper_params: dict,  # Dictionary of hyperparameters
               max_history_length: int,  # Maximum length of weight history
               fluctuation_threshold: float,  # Threshold for weight fluctuations
               device: str = 'cpu'):
    super().__init__()

    # Motor population size should be multiple of number of actions
    assert out_size % num_actions == 0, "Number of motor control neurons must be multiple of number of actions"
    self.motor_pop_size = int(out_size / num_actions)


    ## Layers ##
    input = Input(n=in_size)    # Inputs from association area
    output = AdaptiveLIFNodes(  # Motor control neurons
      n=out_size,
      thresh=hyper_params['thresh_out'],
      theta_plus=hyper_params['theta_plus_out'],
      refrac=hyper_params['refrac_out'],
      reset=hyper_params['reset_out'],
      tc_theta_decay=hyper_params['tc_theta_decay_out'],
      tc_decay=hyper_params['tc_decay_out'],
      traces=True,
    )
    output_monitor = Monitor(output, ["s"], device=device)
    input_monitor = Monitor(input, ["s"], device=device)
    self.output_monitor = output_monitor
    self.input_monitor = input_monitor
    self.add_monitor(input_monitor, name='input_monitor')
    self.add_monitor(output_monitor, name='output_monitor')
    self.add_layer(input, name='input')
    self.add_layer(output, name='output')


    ## Connections ##
    # in - resivour 
    # out - motor out 
    # wfeat - is the weight feature between the input and output
    in_out_wfeat = Weight(name='in_out_weight_feature', value=torch.Tensor(w_exc_out))
    in_out_conn = MulticompartmentConnection(
      source=input, target=output,
      device=device, pipeline=[in_out_wfeat],
    )
    out_out_wfeat = Weight(name='out_out_weight_feature', value=torch.Tensor(w_out_out))
    out_out_conn = MulticompartmentConnection(
      source=output, target=output,
      device=device, pipeline=[out_out_wfeat],
    )
    self.add_connection(in_out_conn, source='input', target='output')
    self.add_connection(out_out_conn, source='output', target='output')
    self.weights = in_out_wfeat
    self.w_mask = in_out_wfeat.value != 0

    ## Weight Tracking Parameters ##
    self.weight_history = []
    self.max_history_length = max_history_length
    self.fluctuation_threshold = fluctuation_threshold
    self.weight_stats = {'mean': [], 'std': [], 'min': [], 'max': []}
    self.weight_fluctuations = {}  # Dictionary to track fluctuations per synapse
    self.fluctuating_synapses = set()  # Set to track synapses that fluctuate too much
    self.weight_moving_avg = None  # Will store exponential moving average of weights
    self.ema_alpha = 0.1  # Weight for exponential moving average

    ## Q-Learning Parameters ##
    self.gamma = gamma
    self.alpha = alpha
    self.num_actions = num_actions
    self.q_table = {}

    ## STDP Parameters ##
    self.wmin, self.wmax = wmin, wmax
    self.decay = decay
    self.lr = lr

  def STDP_RL(self, reward: float, input_spikes, output_spikes):
    # Calculate STDP learning eligibility
    eligibility = torch.outer(input_spikes.sum(0), output_spikes.sum(0))

    # Update weights according to reward and eligibility
    dw = self.lr * eligibility * reward
    #TODO: check the fluctuations in the weights - and if if fluctates to much that just set its value 
    # trak using something like tensor to see the fluctuations
    # maybe craete some cisualizaions of the weights (synaptic weight matrix) (synaptis = weight matrix)
    self.weights.value += dw
    self.weights.value = torch.clamp(self.weights.value, self.wmin, self.wmax)

    # Update weight history
    self.weight_history.append(self.weights.value.clone())
    if len(self.weight_history) > self.max_history_length:
      self.weight_history.pop(0)

    # Update weight statistics
    self.weight_stats['mean'].append(self.weights.value.mean().item())
    self.weight_stats['std'].append(self.weights.value.std().item())
    self.weight_stats['min'].append(self.weights.value.min().item())
    self.weight_stats['max'].append(self.weights.value.max().item())

    # Update weight moving average
    if self.weight_moving_avg is None:
      self.weight_moving_avg = self.weights.value.clone()
    else:
      self.weight_moving_avg = self.ema_alpha * self.weights.value + (1 - self.ema_alpha) * self.weight_moving_avg

    # Update weight fluctuations
    for i in range(self.weights.value.shape[0]):
      for j in range(self.weights.value.shape[1]):
        if (i, j) not in self.weight_fluctuations:
          self.weight_fluctuations[(i, j)] = []
        self.weight_fluctuations[(i, j)].append(self.weights.value[i, j].item())

    # Check for fluctuating synapses
    for synapse, fluctuations in self.weight_fluctuations.items():
      if np.std(fluctuations) > self.fluctuation_threshold:
        self.fluctuating_synapses.add(synapse)

  def Q_Learning(self, state: tuple, action: int, reward: float, next_state: tuple):
    if state not in self.q_table:
      self.q_table[state] = np.zeros(self.num_actions)
    if next_state not in self.q_table:
      self.q_table[next_state] = np.zeros(self.num_actions)
    og_val = self.q_table[state][action]
    next_max_val = self.q_table[next_state].max()
    self.q_table[state][action] = og_val + self.alpha * (reward + self.gamma * next_max_val - og_val)
    delta_q = self.q_table[state][action] - og_val
    return delta_q

  # Take in_spikes (association area spikes) and return action based on motor area spikes
  def select_action(self, in_spikes: np.ndarray, sim_time: int):
    self.run(inputs={"input": torch.Tensor(in_spikes)}, time=sim_time)
    out_spikes = self.output_monitor.get("s")
    out_spikes = out_spikes.squeeze(1)    # Remove batch dimension
    # If no spikes, return random action
    # Artificial spikes to encourage STDP learning
    if torch.max(out_spikes) == 0:
      action = np.random.randint(self.num_actions)
      out_spikes = torch.zeros_like(out_spikes)
      motor_pop_range = (action * self.motor_pop_size, (action + 1) * self.motor_pop_size)
      out_spikes[:, motor_pop_range[0]:motor_pop_range[1]] = torch.rand(sim_time, self.motor_pop_size) < 0.05
    else:
      summed_spikes = out_spikes.sum(0)
      max_val = summed_spikes.numpy().max()
      max_inds = np.where(summed_spikes == max_val)[0]
      action = max_inds[np.random.randint(0, len(max_inds))]
      # action = torch.argmax(out_spikes.reshape(sim_time, self.num_actions).sum(0))
    return action, out_spikes

  def plot_weights(self, ax):
    w = self.weights.value
    ax.imshow(w, cmap='viridis')
    ax.set_title('Synaptic Weights')
    ax.set_xlabel('Motor Control Neurons')
    ax.set_ylabel('Association Neurons')
    ax.set_aspect('auto')
    return ax

  def plot_spikes(self, ax, spikes):
    ax.imshow(spikes.T, aspect='auto', cmap='binary')
    ax.set_title('Spikes')
    ax.set_xlabel('Time')
    ax.set_ylabel('Neuron')
    return ax

  def visualize_weight_fluctuations(self, num_weights=20, save_dir=None):
    """Create detailed, modular visualizations of weight fluctuations.
    
    Args:
        num_weights: Number of top fluctuating weights to analyze
        save_dir: Optional directory to save visualization images
        
    Returns:
        List of indices of the most fluctuating weights
        Dictionary of file paths if save_dir is provided
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.ticker as ticker
    import os
    from datetime import datetime
    import seaborn as sns
    
    if len(self.weight_history) == 0:
      print("No weight history available yet.")
      return []
    
    # Determine save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = {}
    
    # Stack history into array: (steps, in_size, out_size)
    weight_array = np.stack([w.cpu().numpy() for w in self.weight_history], axis=0)
    num_steps, in_size, out_size = weight_array.shape
    flattened = weight_array.reshape(num_steps, -1)
    
    # Choose weights with highest fluctuation
    variances = np.var(flattened, axis=0)
    indices = np.argsort(variances)[-num_weights:]  # Top fluctuating weights
    
    # Set a modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # ============ FIGURE 1: OVERVIEW OF TOP FLUCTUATING WEIGHTS ============
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    
    # Use a pleasing color palette
    colors = sns.color_palette("husl", num_weights)
    
    for i, idx in enumerate(indices):
      y = flattened[:, idx]
      input_idx = idx // out_size
      output_idx = idx % out_size
      ax1.plot(np.arange(num_steps), y, 
               label=f'W({input_idx},{output_idx})', 
               color=colors[i], linewidth=2, alpha=0.8)
    
    # Beautify the plot
    ax1.set_title('Top Fluctuating Synaptic Weights Over Time', fontsize=18, pad=15)
    ax1.set_xlabel('Time Step', fontsize=14)
    ax1.set_ylabel('Weight Value', fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Create a better legend
    if num_weights <= 10:
      ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    else:
      ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                fontsize=10, ncol=2)
    
    ax1.set_xlim(0, num_steps-1)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(10))
    
    fig1.tight_layout()
    
    if save_dir:
      overview_file = os.path.join(save_dir, f'weight_overview_{timestamp}.png')
      fig1.savefig(overview_file, dpi=150, bbox_inches='tight')
      saved_files['overview'] = overview_file
      plt.close(fig1)
    else:
      plt.show()
    
    # ============ FIGURE 2: DETAIL VIEW OF TOP 6 WEIGHTS ============
    num_detail = min(6, num_weights)
    fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i in range(num_detail):
      ax = axes[i]
      idx = indices[i]
      y = flattened[:, idx]
      
      # Get weight coordinates
      input_idx = idx // out_size
      output_idx = idx % out_size
      
      # Create a plot with shaded area for the range
      time_steps = np.arange(num_steps)
      mean_val = np.mean(y)
      std_val = np.std(y)
      
      ax.plot(time_steps, y, color=colors[i], linewidth=2.5, label=f'Weight Value')
      ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.7, label='Mean')
      
      # Set plot limits and labels
      ax.set_xlim(0, num_steps-1)
      y_range = y.max() - y.min()
      if y_range > 0:
        ax.set_ylim(y.min() - 0.1*y_range, y.max() + 0.1*y_range)
      
      # Add clean, clear labels
      ax.set_title(f'Input Neuron {input_idx} to Output Neuron {output_idx}', 
                  fontsize=14, pad=10)
      ax.set_xlabel('Time Step', fontsize=12)
      ax.set_ylabel('Weight Value', fontsize=12)
      ax.tick_params(axis='both', which='major', labelsize=10)
      
      # Add statistics in a clearly formatted text box
      stats_text = (f'Mean: {mean_val:.2f}\n'
                   f'Std Dev: {std_val:.2f}\n'
                   f'Min: {np.min(y):.2f}\n'
                   f'Max: {np.max(y):.2f}\n'
                   f'Change: {abs(y[-1] - y[0]):.2f}')
      
      ax.text(0.03, 0.97, stats_text,
              verticalalignment='top', horizontalalignment='left',
              transform=ax.transAxes, fontsize=10,
              bbox=dict(facecolor='white', alpha=0.8, 
                        edgecolor='gray', boxstyle='round,pad=0.5'))
      
      ax.legend(loc='lower right', fontsize=10)
      
    # If we have fewer than 6 weights, hide the empty subplots
    for i in range(num_detail, 6):
      axes[i].axis('off')
      
    fig2.suptitle('Detailed View of Top Fluctuating Weights', fontsize=18)
    fig2.tight_layout()
    fig2.subplots_adjust(top=0.92)
    
    if save_dir:
      detail_file = os.path.join(save_dir, f'weight_details_{timestamp}.png')
      fig2.savefig(detail_file, dpi=150, bbox_inches='tight')
      saved_files['detail'] = detail_file
      plt.close(fig2)
    else:
      plt.show()
      
    # ============ FIGURE 3: HEATMAP VIEW ============
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    
    variance_matrix = variances.reshape(in_size, out_size)
    
    # Create a better heatmap with seaborn
    sns.heatmap(variance_matrix, cmap='magma', ax=ax3, cbar_kws={'label': 'Variance'})
    
    ax3.set_title('Synaptic Weight Variance Heatmap', fontsize=18)
    ax3.set_xlabel('Output Neuron Index', fontsize=14)
    ax3.set_ylabel('Input Neuron Index', fontsize=14)
    
    # Add threshold markers
    high_var_positions = np.where(variance_matrix > self.fluctuation_threshold)
    if len(high_var_positions[0]) > 0:
      # Draw circles around high variance synapses
      for i, j in zip(high_var_positions[0], high_var_positions[1]):
        ax3.add_patch(plt.Circle((j+0.5, i+0.5), 0.4, fill=False, 
                               edgecolor='white', linewidth=1.5, alpha=0.8))
      
      # Add an annotation about the threshold in a better position
      threshold_text = f'White circles: Variance > {self.fluctuation_threshold:.3f} (threshold)'
      # Place text at top of plot instead of bottom to avoid overlap with axis labels
      ax3.text(0.5, 1.05, threshold_text, horizontalalignment='center',
              transform=ax3.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.3'))
    
    if save_dir:
      heatmap_file = os.path.join(save_dir, f'weight_heatmap_{timestamp}.png')
      fig3.savefig(heatmap_file, dpi=150, bbox_inches='tight')
      saved_files['heatmap'] = heatmap_file
      plt.close(fig3)
    else:
      plt.show()
      
    # ============ FIGURE 4: DISTRIBUTION VIEW ============
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    
    # Create a more informative histogram using seaborn
    sns.histplot(variances, bins=30, kde=True, ax=ax4, color='purple')
    
    # Add the threshold line
    ax4.axvline(self.fluctuation_threshold, color='red', linestyle='--', linewidth=2.5,
               label=f'Threshold ({self.fluctuation_threshold:.3f})')
    
    # Annotate the percentages
    percent_above = 100 * np.sum(variances > self.fluctuation_threshold) / len(variances)
    
    ax4.text(0.98, 0.95, 
             f'{percent_above:.1f}% of synapses\nabove threshold',
             horizontalalignment='right', verticalalignment='top',
             transform=ax4.transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax4.set_title('Distribution of Synaptic Weight Variances', fontsize=18)
    ax4.set_xlabel('Variance', fontsize=14)
    ax4.set_ylabel('Count', fontsize=14)
    ax4.tick_params(axis='both', which='major', labelsize=12)
    ax4.legend(fontsize=12)
    
    if save_dir:
      hist_file = os.path.join(save_dir, f'weight_distribution_{timestamp}.png')
      fig4.savefig(hist_file, dpi=150, bbox_inches='tight')
      saved_files['distribution'] = hist_file
      plt.close(fig4)
    else:
      plt.show()
    
    # Create a combined image for the report if needed
    if save_dir:
      # Create one combined figure with all visualizations
      fig_combined = plt.figure(figsize=(20, 24))
      gs = fig_combined.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.3)
      
      # Add each image as a subplot
      for i, (name, path) in enumerate(saved_files.items()):
        ax = fig_combined.add_subplot(gs[i, 0])
        img = plt.imread(path)
        ax.imshow(img)
        ax.axis('off')
      
      combined_file = os.path.join(save_dir, f'weight_fluctuations_{timestamp}.png')
      fig_combined.savefig(combined_file, dpi=150, bbox_inches='tight')
      saved_files['combined'] = combined_file
      plt.close(fig_combined)
    
    # Reset style to default
    plt.style.use('default')
      
    # Return indices of fluctuating weights and the saved files
    if save_dir:
      return indices, saved_files
    else:
      return indices

  def plot_weight_stats(self, ax=None):
    # Plot the weight statistics over time.
    import matplotlib.pyplot as plt
    import numpy as np
    
    if len(self.weight_stats['mean']) == 0:
      print("No weight statistics available yet.")
      return
    
    created_fig = False
    if ax is None:
      fig, ax = plt.subplots(figsize=(10, 6))
      created_fig = True
    
    x = np.arange(len(self.weight_stats['mean']))
    mean = np.array(self.weight_stats['mean'])
    std = np.array(self.weight_stats['std'])
    min_vals = np.array(self.weight_stats['min'])
    max_vals = np.array(self.weight_stats['max'])
    
    # Plot mean with std deviation
    ax.plot(x, mean, 'b-', label='Mean')
    ax.fill_between(x, mean - std, mean + std, alpha=0.3, color='blue')
    
    # Plot min and max
    ax.plot(x, min_vals, 'g--', alpha=0.7, label='Min')
    ax.plot(x, max_vals, 'r--', alpha=0.7, label='Max')
    
    # If we have fluctuating synapse data, add count
    if hasattr(self, 'fluctuating_synapses') and len(self.fluctuating_synapses) > 0:
      ax_count = ax.twinx()  # Create a second y-axis
      counts = [len(self.fluctuating_synapses) if i == len(x)-1 else 0 for i in range(len(x))]
      ax_count.plot(x, counts, 'm-', label='# Fluctuating Synapses')
      ax_count.set_ylabel('# Synapses Above Threshold', color='m')
      ax_count.tick_params(axis='y', colors='m')
    
    ax.set_title('Synaptic Weight Statistics Over Time')
    ax.set_xlabel('Step')
    ax.set_ylabel('Weight Value')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    if created_fig:
      plt.tight_layout()
      plt.show()

  def identify_fluctuating_synapses(self, threshold=None):
    """Identify synapses that fluctuate more than the threshold.
    
    Args:
        threshold: Optional custom threshold. If None, uses self.fluctuation_threshold.
        
    Returns:
        list of tuples: (i, j) coordinates of over-fluctuating synapses
        numpy.ndarray: variance matrix of all synapses
    """
    import numpy as np
    
    if len(self.weight_history) < 2:
      print("Insufficient weight history to calculate fluctuations.")
      return [], None
    
    # Use class threshold if none provided
    if threshold is None:
      threshold = self.fluctuation_threshold
    
    # Stack history into array
    weight_array = np.stack([w.cpu().numpy() for w in self.weight_history], axis=0)
    in_size, out_size = weight_array.shape[1], weight_array.shape[2]
    
    # Calculate variance for each synapse
    variances = np.var(weight_array, axis=0)  # Shape: (in_size, out_size)
    
    # Find synapses with variance above threshold
    high_variance_indices = np.where(variances > threshold)
    fluctuating_synapses = list(zip(high_variance_indices[0].tolist(), 
                                   high_variance_indices[1].tolist()))
    
    # Update the class's set of fluctuating synapses
    self.fluctuating_synapses = set(fluctuating_synapses)
    
    return fluctuating_synapses, variances
    
  def stabilize_fluctuating_synapses(self, method='clamp', custom_threshold=None, damping_factor=0.5):
    """Apply stabilization to synapses that fluctuate too much.
    
    Args:
        method: Stabilization method. One of:
            - 'clamp': Fix weights of fluctuating synapses
            - 'dampen': Reduce learning rate for fluctuating synapses
            - 'average': Set weights to their moving average
        custom_threshold: Optional custom threshold to use instead of self.fluctuation_threshold
        damping_factor: Factor to reduce learning rate by for 'dampen' method
        
    Returns:
        int: Number of synapses stabilized
    """
    import numpy as np
    import torch
    
    # Identify over-fluctuating synapses
    fluctuating_synapses, variances = self.identify_fluctuating_synapses(custom_threshold)
    
    if not fluctuating_synapses:
      return 0
    
    # Apply stabilization based on method
    if method == 'clamp':
      # Clamp weights to their current values
      mask = torch.ones_like(self.weights.value)
      for i, j in fluctuating_synapses:
        mask[i, j] = 0  # Zero masks out this synapse from future updates
      
      # Store the mask for use in update steps
      self.stabilization_mask = mask
      
    elif method == 'dampen':
      # Create a damping mask to reduce learning rate for fluctuating synapses
      mask = torch.ones_like(self.weights.value)
      for i, j in fluctuating_synapses:
        mask[i, j] = damping_factor  # Reduce learning rate for this synapse
      
      # Store the mask for use in update steps
      self.stabilization_mask = mask
      
    elif method == 'average':
      # Set weights to their moving average
      if self.weight_moving_avg is not None:
        for i, j in fluctuating_synapses:
          self.weights.value[i, j] = self.weight_moving_avg[i, j]
    
    print(f"Stabilized {len(fluctuating_synapses)} synapses with method '{method}'")
    return len(fluctuating_synapses)
    
  def create_weight_animation(self, filename=None, fps=5, dpi=100, episodes_per_step=None):
    """Create an animation of the weight matrix evolution.
    
    Args:
        filename: If provided, save animation to this file
        fps: Frames per second for the animation
        dpi: Resolution for the saved animation
        episodes_per_step: Optional mapping of steps to episode numbers
        
    Returns:
        animation object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation
    import matplotlib.cm as cm
    
    if len(self.weight_history) < 2:
      print("Insufficient weight history for animation.")
      return None
    
    # Stack history into array: (steps, in_size, out_size)
    weight_array = np.stack([w.cpu().numpy() for w in self.weight_history], axis=0)
    num_steps, in_size, out_size = weight_array.shape
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Find global min/max for consistent color scaling
    vmin, vmax = weight_array.min(), weight_array.max()
    
    # Initialize plots
    im = ax1.imshow(weight_array[0], cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
    fig.colorbar(im, ax=ax1, label='Weight Value')
    ax1.set_title('Weight Matrix')
    ax1.set_xlabel('Output Neuron Index')
    ax1.set_ylabel('Input Neuron Index')
    
    # Initialize histogram
    hist_bins = np.linspace(vmin, vmax, 30)
    n, bins, patches = ax2.hist(weight_array[0].flatten(), bins=hist_bins, alpha=0.7)
    ax2.set_title('Weight Distribution')
    ax2.set_xlabel('Weight Value')
    ax2.set_ylabel('Count')
    
    # Calculate episodes if not provided
    if episodes_per_step is None:
      # Create a default mapping - assume 40 steps per episode
      episodes = [i // 40 + 1 for i in range(num_steps)]
    else:
      episodes = episodes_per_step
      
    # Add step counter and episode info
    counter_text = fig.text(0.5, 0.95, f'Episode: 1, Step: 0/{num_steps-1}', 
                       ha='center', va='center', fontsize=12)
    
    # Update function for animation
    def update(frame):
      # Update weight matrix
      im.set_array(weight_array[frame])
      
      # Update histogram
      ax2.clear()
      ax2.hist(weight_array[frame].flatten(), bins=hist_bins, alpha=0.7)
      ax2.set_title('Weight Distribution')
      ax2.set_xlabel('Weight Value')
      ax2.set_ylabel('Count')
      ax2.set_xlim(vmin, vmax)
      
      # Update step counter with episode info
      if isinstance(episodes, list):
        episode = episodes[frame] if frame < len(episodes) else episodes[-1]
      else:
        # If we just have a dictionary mapping steps to episodes
        episode = episodes.get(frame, 1)  # Default to episode 1 if not found
        
      counter_text.set_text(f'Episode: {episode}, Step: {frame}/{num_steps-1}')
      
      return [im] + [patch for patch in ax2.patches]
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=num_steps, 
                        interval=1000/fps, blit=False)
    
    plt.tight_layout()
    
    # Save if filename provided
    if filename:
      ani.save(filename, fps=fps, dpi=dpi)
      print(f"Animation saved to {filename}")
    
    plt.show()
    return ani
