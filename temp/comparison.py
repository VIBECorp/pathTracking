#!/usr/bin/env python
"""
Compare reference trajectory with optimized trajectory in joint space.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add parent directory to path to import tracking modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tracking.utils.spline_utils import Spline


def load_reference_trajectory(filepath, num_samples=1000, spline_config_path=None, 
                             start_dis=None, end_dis=None):
    """
    Load reference trajectory from JSON file.
    Supports both trajectory format (with 'joint' key) and spline format (with '_b_spline' or '_tck' key).
    
    Args:
        filepath: Path to reference trajectory or spline JSON file
        num_samples: Number of samples to extract from spline (if spline format)
        spline_config_path: Path to spline_config.json file (optional, auto-detected if None)
        start_dis: Start distance along spline (for partial spline comparison)
        end_dis: End distance along spline (for partial spline comparison)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Check if it's a spline format
    is_spline = '_b_spline' in data or '_tck' in data
    
    if is_spline:
        # Load spline config if provided or auto-detect
        length_correction_step_size = None
        use_normalized_length_correction_step_size = None
        
        if spline_config_path and os.path.exists(spline_config_path):
            with open(spline_config_path, 'r') as f:
                spline_config = json.load(f)
            length_correction_step_size = spline_config.get('length_correction_step_size')
            use_normalized_length_correction_step_size = spline_config.get('use_normalized_length_correction_step_size')
            print(f"Using spline config from: {spline_config_path}")
        else:
            # Try to find spline_config.json in the same directory or parent directory
            spline_dir = os.path.dirname(filepath)
            parent_dir = os.path.dirname(spline_dir)
            config_candidates = [
                os.path.join(spline_dir, 'spline_config.json'),
                os.path.join(parent_dir, 'spline_config.json'),
            ]
            
            for config_path in config_candidates:
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        spline_config = json.load(f)
                    length_correction_step_size = spline_config.get('length_correction_step_size')
                    use_normalized_length_correction_step_size = spline_config.get('use_normalized_length_correction_step_size')
                    print(f"Auto-detected spline config from: {config_path}")
                    break
        
        # Load spline and sample it
        print(f"Detected spline format. Loading spline...")
        spline = Spline.load_from_json(
            filepath,
            length_correction_step_size=length_correction_step_size,
            use_normalized_length_correction_step_size=use_normalized_length_correction_step_size
        )
        
        # Set start and end distance if provided (for partial spline comparison)
        if start_dis is not None and end_dis is not None:
            # Convert distance to u_arc using spline's length
            spline.reset()
            spline_length = spline.get_length()
            
            if spline_length > 0:
                # Normalize distances to [0, 1] range
                u_arc_start = start_dis / spline_length
                u_arc_end = end_dis / spline_length
                
                # Clamp to valid range
                u_arc_start = max(0.0, min(1.0, u_arc_start))
                u_arc_end = max(0.0, min(1.0, u_arc_end))
            else:
                u_arc_start = 0.0
                u_arc_end = 1.0
            
            spline.set_u_start(u_arc_start=u_arc_start)
            spline.set_u_end_index(u_arc_end_min=u_arc_end)
            print(f"Using spline segment: start_dis={start_dis:.4f}, end_dis={end_dis:.4f}")
            print(f"  -> u_arc_start={u_arc_start:.4f}, u_arc_end={u_arc_end:.4f}")
            print(f"  -> spline length={spline_length:.4f}")
        else:
            # Use full spline
            u_arc_start = 0.0
            u_arc_end = 1.0
        
        # Get u parameter range from spline's start and end
        if hasattr(spline, 'u_start') and spline.u_start is not None:
            u_min = spline.u_start
        else:
            u_min = spline.u[0] if hasattr(spline, 'u') and len(spline.u) > 0 else 0.0
        
        if hasattr(spline, 'u_end_index') and spline.u_end_index is not None:
            u_max = spline.u[spline.u_end_index] if hasattr(spline, 'u') and len(spline.u) > spline.u_end_index else 1.0
        else:
            u_max = spline.u[-1] if hasattr(spline, 'u') and len(spline.u) > 0 else 1.0
        
        # Sample uniformly in u space
        u_samples = np.linspace(u_min, u_max, num_samples)
        
        # Evaluate spline at sampled points
        joint_data = spline.evaluate(u_samples).T  # Shape: (num_samples, num_joints)
        
        # Create time array based on arc length
        # Use normalized u for time (assuming uniform speed)
        time_data = (u_samples - u_min) / (u_max - u_min) if u_max > u_min else u_samples
        
        # Scale time by execution_time if present; otherwise by spline length
        execution_time = data.get('execution_time')
        if execution_time is not None:
            time_data = time_data * execution_time
        elif hasattr(spline, 'get_length'):
            spline_length = spline.get_length()
            # Estimate time based on length (assuming constant speed)
            # You can adjust this based on your needs
            time_data = time_data * spline_length
        
        print(f"Spline loaded: {len(joint_data)} samples, {joint_data.shape[1]} joints")
        print(f"  Initial position: {joint_data[0]}")
        print(f"  Final position: {joint_data[-1]}")
        
    else:
        # Original trajectory format
        # Extract joint positions
        joint_data = np.array(data['joint'])  # Shape: (num_waypoints, num_joints)
        
        # Create time array if available, otherwise use index
        if 'time' in data and len(data['time']) == len(joint_data):
            time_data = np.array(data['time'])
            # Normalize time to start from 0
            time_data = time_data - time_data[0]
        else:
            time_data = np.arange(len(joint_data))
        
        #nanoseconds to seconds
        time_data = time_data.astype(float)
        time_data *= 10e-9 
    
    return joint_data, time_data


def load_optimized_trajectory(filepath, use_measured=True):
    """Load optimized trajectory from evaluation output JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Use measured actual values if available, otherwise use setpoints
    if use_measured and 'trajectory_measured_actual_values' in data:
        trajectory_data = data['trajectory_measured_actual_values']
    elif 'trajectory_setpoints' in data:
        trajectory_data = data['trajectory_setpoints']
    else:
        raise ValueError("No trajectory data found in optimized file")
    
    # Extract positions
    positions = np.array(trajectory_data['positions'])  # Shape: (num_waypoints, num_joints)
    
    # Create time array based on trajectory_time_step
    # trajectory_time_step = data.get('trajectory_time_step', 0.1)
    trajectory_time_step = data.get('simulation_time_step', 0.1)
    time_data = np.arange(len(positions)) * trajectory_time_step
    
    return positions, time_data, data


def plot_trajectory_comparison(reference_file, optimized_file, output_file=None, use_measured=True, 
                               spline_config_path=None, num_samples=1000, use_optimized_spline_range=True):
    """
    Plot comparison between reference and optimized trajectories in joint space.
    
    Args:
        reference_file: Path to reference trajectory JSON file
        optimized_file: Path to optimized trajectory JSON file
        output_file: Path to save the plot (optional)
        use_measured: If True, use measured actual values; otherwise use setpoints
        spline_config_path: Path to spline_config.json file (optional)
        num_samples: Number of samples to extract from spline (if spline format)
        use_optimized_spline_range: If True, use the same spline range as optimized trajectory
    """
    
    # Load optimized trajectory first to get spline range info
    print(f"Loading optimized trajectory from {optimized_file}...")
    opt_positions, opt_time, opt_data = load_optimized_trajectory(optimized_file, use_measured=use_measured)
    
    # Extract spline range from optimized trajectory if available
    start_dis = None
    end_dis = None
    if use_optimized_spline_range and 'reference_spline_start_dis' in opt_data and 'reference_spline_end_dis' in opt_data:
        start_dis = opt_data['reference_spline_start_dis']
        end_dis = opt_data['reference_spline_end_dis']
        print(f"Using optimized trajectory's spline range: start_dis={start_dis:.4f}, end_dis={end_dis:.4f}")
    
    # Load reference trajectory
    print(f"Loading reference trajectory from {reference_file}...")
    ref_positions, ref_time = load_reference_trajectory(reference_file, 
                                                        num_samples=num_samples,
                                                        spline_config_path=spline_config_path,
                                                        start_dis=start_dis,
                                                        end_dis=end_dis)
    num_joints = ref_positions.shape[1]
    
    print(f"Reference: {len(ref_positions)} waypoints, {num_joints} joints")
    print(f"Optimized: {len(opt_positions)} waypoints, {opt_positions.shape[1]} joints")
    
    # Ensure same number of joints
    if opt_positions.shape[1] != num_joints:
        min_joints = min(num_joints, opt_positions.shape[1])
        print(f"Warning: Joint count mismatch. Using first {min_joints} joints.")
        ref_positions = ref_positions[:, :min_joints]
        opt_positions = opt_positions[:, :min_joints]
        num_joints = min_joints
    
    # Create subplots: one for each joint
    fig, axes = plt.subplots(num_joints, 1, figsize=(12, 3 * num_joints))
    if num_joints == 1:
        axes = [axes]
    
    fig.suptitle('Trajectory Comparison: Reference vs Optimized (Joint Space)', fontsize=16, fontweight='bold')
    
    # Plot each joint
    for joint_idx in range(num_joints):
        ax = axes[joint_idx]
        
        # Plot reference trajectory
        ax.plot(ref_time, ref_positions[:, joint_idx], 
               'b-', linewidth=2, label='Reference', alpha=0.7)
        
        # Plot optimized trajectory
        ax.plot(opt_time, opt_positions[:, joint_idx], 
               'r--', linewidth=2, label='Optimized', alpha=0.7)
        
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel(f'Joint {joint_idx+1} Position (rad)', fontsize=10)
        ax.set_title(f'Joint {joint_idx+1}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
    
    # Calculate and print statistics
    print("\n=== Comparison Statistics ===")
    print(f"Reference trajectory duration: {ref_time[-1]:.3f} s")
    print(f"Optimized trajectory duration: {opt_time[-1]:.3f} s")
    
    # Calculate mean squared error for each joint
    # Interpolate to common time points for comparison
    common_time = np.linspace(0, min(ref_time[-1], opt_time[-1]), 1000)
    ref_interp = np.array([np.interp(common_time, ref_time, ref_positions[:, j]) 
                           for j in range(num_joints)]).T
    opt_interp = np.array([np.interp(common_time, opt_time, opt_positions[:, j]) 
                           for j in range(num_joints)]).T
    
    mse_per_joint = np.mean((ref_interp - opt_interp) ** 2, axis=0)
    rmse_per_joint = np.sqrt(mse_per_joint)
    
    print("\nRoot Mean Squared Error (RMSE) per joint:")
    for j in range(num_joints):
        print(f"  Joint {j+1}: {rmse_per_joint[j]:.6f} rad")
    
    print(f"\nOverall RMSE: {np.mean(rmse_per_joint):.6f} rad")
    
    return fig


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare reference and optimized trajectories')
    parser.add_argument('--reference', type=str, 
                        # default='temp/trajectory_kuka.json',
                        default='temp/splines/spline_kuka.json',
                        help='Path to reference trajectory JSON file')
    
    parser.add_argument('--optimized', type=str, 
                        default='/home/astra/tracking_evaluation/TrackingEnvSpline/industrial/no_balancing/target_point/20251215T141441/trajectory_data/episode_1_73064.json',
                        help='Path to optimized trajectory JSON file')
    
    parser.add_argument('--output', type=str, default='',
                        help='Output file path for the plot')
    parser.add_argument('--use_setpoints', action='store_true',
                        help='Use setpoints instead of measured actual values')
    
    parser.add_argument('--spline_config', type=str, 
                        default='temp/splines/spline_config.json',
                        help='Path to spline_config.json file (auto-detected if not provided)')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to extract from spline (if spline format)')
    parser.add_argument('--no_use_optimized_range', action='store_true',
                        help='Do not use optimized trajectory\'s spline range (use full spline)')
    
    args = parser.parse_args()
    
    plot_trajectory_comparison(
        reference_file=args.reference,
        optimized_file=args.optimized,
        output_file=args.output if args.output else None,
        use_measured=not args.use_setpoints,
        spline_config_path=args.spline_config,
        num_samples=args.num_samples,
        use_optimized_spline_range=not args.no_use_optimized_range
    )
