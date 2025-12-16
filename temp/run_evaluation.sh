#!/bin/bash
# Script to run evaluation with custom spline

# Convert trajectory to spline format
echo "Converting trajectory to spline format..."
python temp/convert_to_spline.py \
    --input temp/trajectory_kuka.json \
    --output_dir temp/splines \
    --output_filename spline_kuka.json \
    --length_correction_step_size 2e-4

# Create spline_config.json if it doesn't exist
if [ ! -f temp/splines/spline_config.json ]; then
    echo "Creating spline_config.json..."
    cat > temp/splines/spline_config.json << EOF
{
  "max_dis_between_knots": 1.0,
  "num_train": 0,
  "num_test": 1,
  "resampling_distance": 0.1,
  "curvature_sampling_distance": 0.3,
  "length_correction_step_size": 0.0002,
  "use_normalized_length_correction_step_size": false
}
EOF
fi

# Run evaluation
echo "Running evaluation..."
python tracking/evaluate.py \
    --checkpoint industrial/no_balancing/target_point \
    --episodes 1 \
    --num_workers 0 \
    --spline_dir temp/splines \
    --spline_name_list '["spline_kuka.json"]' \
    --store_metrics \
    --store_trajectory \
    --plot_spline \
    --logging_level INFO

