#!/usr/bin/env python
"""
Trajectory optimization script for KUKA joint trajectories.
Converts continuous waypoints to spline and resamples to reduce waypoints.
"""

import json
import numpy as np
import sys
import os

# Add parent directory to path to import tracking modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tracking.utils.spline_utils import Spline


def optimize_trajectory(input_file, output_file, resampling_distance=0.01, 
                        use_curvature=False, curvature_sampling_distance=0.05):
    """
    Optimize trajectory by converting to spline and resampling.
    
    Args:
        input_file: Path to input trajectory JSON file
        output_file: Path to output optimized trajectory JSON file
        resampling_distance: Distance between resampled points (in normalized space if use_normalized=True)
        use_curvature: If True, use curvature-based resampling (more points in curved regions)
        curvature_sampling_distance: Distance for curvature-based resampling
    """
    
    # Load trajectory data
    print(f"Loading trajectory from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract joint positions
    joint_data = np.array(data['joint'])  # Shape: (num_waypoints, num_joints)
    num_waypoints, num_joints = joint_data.shape
    
    print(f"Original trajectory: {num_waypoints} waypoints, {num_joints} joints")
    
    # Convert to spline format (transpose: joints x waypoints)
    curve_data = joint_data.T  # Shape: (num_joints, num_waypoints)
    
    # Create spline from trajectory
    print("Creating spline from trajectory...")
    spline = Spline(
        curve_data=curve_data,
        curve_data_slicing_step=1,
        length_correction_step_size=0.01,
        use_normalized_length_correction_step_size=True,
        method="auto",
        curvature_at_ends=0
    )
    
    print(f"Original spline length: {spline.get_length():.4f}")
    print(f"Original max distance between knots: {spline.max_dis_between_knots:.4f}")
    
    # Resample spline to reduce waypoints
    if use_curvature:
        print(f"Resampling with curvature-based method (distance: {curvature_sampling_distance})...")
        optimized_spline = spline.copy_with_resampling(
            resampling_distance=curvature_sampling_distance,
            use_normalized_resampling_distance=False,
            use_curvature_for_resampling=True,
            length_correction_step_size=0.01,
            use_normalized_length_correction_step_size=True,
            curvature_at_ends=0
        )
    else:
        print(f"Resampling with uniform distance (normalized: {resampling_distance})...")
        optimized_spline = spline.copy_with_resampling(
            resampling_distance=resampling_distance,
            use_normalized_resampling_distance=True,
            use_curvature_for_resampling=False,
            length_correction_step_size=0.01,
            use_normalized_length_correction_step_size=True,
            curvature_at_ends=0
        )
    
    # Extract optimized waypoints
    optimized_curve_data = optimized_spline.curve_data_spline  # Shape: (num_joints, num_waypoints)
    optimized_joint_data = optimized_curve_data.T  # Shape: (num_waypoints, num_joints)
    num_optimized_waypoints = optimized_joint_data.shape[0]
    
    print(f"Optimized trajectory: {num_optimized_waypoints} waypoints")
    print(f"Reduction: {num_waypoints} -> {num_optimized_waypoints} ({100*(1-num_optimized_waypoints/num_waypoints):.1f}% reduction)")
    print(f"Optimized max distance between knots: {optimized_spline.max_dis_between_knots:.4f}")
    
    # Create new dictionary for optimized trajectory
    optimized_data = {}
    
    # Copy metadata fields if they exist
    # if 'type' in data:
    #     optimized_data['type'] = data['type']
    # if 'task' in data:
    #     optimized_data['task'] = data['task']
    # if 'tool' in data:
    #     optimized_data['tool'] = data['tool']
    
    # Add optimized joint data
    optimized_data['joint'] = optimized_joint_data.tolist()
    
    # If time data exists, interpolate it
    if 'time' in data and len(data['time']) == num_waypoints:
        # Interpolate time based on normalized arc length
        original_arc_lengths = np.linspace(0, 1, num_waypoints)
        optimized_arc_lengths = np.linspace(0, 1, num_optimized_waypoints)
        original_times = np.array(data['time'])
        
        # Simple linear interpolation of time
        from scipy.interpolate import interp1d
        time_interp = interp1d(original_arc_lengths, original_times, kind='linear', 
                               fill_value='extrapolate')
        optimized_data['time'] = time_interp(optimized_arc_lengths).tolist()
        print(f"Interpolated time data: {len(optimized_data['time'])} time points")
    
    # Save optimized trajectory
    print(f"Saving optimized trajectory to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(optimized_data, f, indent=2)
    
    print("Optimization complete!")
    return optimized_spline, num_waypoints, num_optimized_waypoints


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize KUKA trajectory by spline resampling')
    parser.add_argument('--input', type=str, default='temp/trajectory_kuka.json',
                        help='Input trajectory JSON file')
    parser.add_argument('--output', type=str, default='temp/trajectory_kuka_optimized.json',
                        help='Output optimized trajectory JSON file')
    parser.add_argument('--resampling_distance', type=float, default=0.01,
                        help='Resampling distance (normalized, 0-1). Smaller = more waypoints')
    parser.add_argument('--use_curvature', action='store_true',
                        help='Use curvature-based resampling (more points in curved regions)')
    parser.add_argument('--curvature_sampling_distance', type=float, default=0.05,
                        help='Distance for curvature-based resampling (in joint space units)')
    
    args = parser.parse_args()
    
    optimize_trajectory(
        input_file=args.input,
        output_file=args.output,
        resampling_distance=args.resampling_distance,
        use_curvature=args.use_curvature,
        curvature_sampling_distance=args.curvature_sampling_distance
    )

