d:\research
├── data                   # Contains raw data files
│   └── images             # Image dataset folder
│       ├── rgb          
│       ├── depth       
│       ├── mask        
│       ├── scene_camera.json
│       ├── scene_gt.json
│       └── scene_gt_info.json  # Ground truth information for scenes
├── experiments            # Where experimental outputs are stored
│   ├── checkpoints        # Model checkpoint files (saved during training runs)
│   ├── logs               # Training and evaluation logs, e.g. text files or TensorBoard logs
│   └── results            # Directory to store experimental results
├── models                 # Model definitions
│   ├── multi_task_model.py      # Implements a multi-task network
│   ├── keypoint_pose_model.py   # Contains KeypointPoseDecoder for keypoint/pose predictions
│   └── decoders                # Directory for decoder modules
│       ├── keypoint_decoder.py  # Decoder for keypoint predictions
│       └── vector_field_decoder.py  # Decoder for vector field predictions
├── test                   # Evaluation scripts
│   └── evaluate_pose.py         # Evaluate pose estimation performance with various metrics
├── train                  # Training scripts for various model types and learning paradigms
│   ├── train_multi_task.py      # Training script for the multi-task model
│   ├── train_teacher_student.py # Training script for teacher-student paradigm (model distillation)
│   └── train_keypoint_model.py  # Training script for the keypoint/pose model
├── utils                  # Utility modules containing helper functions and classes
│   ├── dataset_loader.py   # Loads dataset and parses image, depth, mask, and pose data
│   ├── loss_functions.py   # Defines loss functions used for multi-task learning
│   ├── heatmap_utils.py    # Contains functions to visualize heatmaps
│   ├── pose_utils.py       # Provides helper functions to compute pose errors
│   └── visualization.py    # Functions to visualize predictions, overlays, and vector fields
└── main.py                # Main script for testing dataset loading and preliminary checks