README.txt
==========

Deep Comparison Between DVR and DeepSDF
---------------------------------------
This guide helps you set up, train, and run the two 3D reconstruction methods:
- Differentiable Volumetric Rendering (DVR)
- Deep Signed Distance Functions (DeepSDF)

Environment Assumptions:
- Ubuntu (WSL recommended)
- Python 3.8
- Basic understanding of Python/terminal commands

Note: For a full log for what we did on the assignment refer to After Milestone Logs and Before Milestone Logs .txt
Note: I had to remove the datasets (Shapenet, DTU, and Chair Shapenet Dataset for DeepSDF) from the submitted files because it would've been too large
, and take very long to zip them or upload them anywhere. Also, had to remove the processed dataset for training too. Send us an email if you need them.:
sma231@sfu.ca, ana106@sfu.ca
Or you can request ShapeNet Dataset from here: https://shapenet.org/

===========================
📦 1. SETUP INSTRUCTIONS
===========================

🔧 Install Python 3.8 and tools:
    sudo apt update
    sudo apt install python3.8 python3.8-venv python3.8-dev -y

🌀 Create virtual environments:
    python3.8 -m venv ~/projects/group_project/differentiable_volumetric_rendering/dvr_env
    python3.8 -m venv ~/projects/group_project/deepsdf2/DeepSDF/deepsdf_env

✳️ Activate environments:
    DVR:     source ~/projects/group_project/differentiable_volumetric_rendering/dvr_env/bin/activate
    DeepSDF: source ~/projects/group_project/deepsdf2/DeepSDF/deepsdf_env/bin/activate

📁 Install dependencies:
- DVR:
    Check Github page and our logs in order to find right versions
    https://github.com/autonomousvision/differentiable_volumetric_rendering
    Environment.yaml also provides versions

- DeepSDF:
    Check Github page and oour logs in order to find right versions
    https://github.com/facebookresearch/DeepSDF

===========================
📂 2. DVR USAGE GUIDE
===========================

🔹 A. Pre-trained Demo (ShapeNet):
    cd differentiable_volumetric_rendering
    source dvr_env/bin/activate
    python setup.py build_ext --inplace
    python generate.py configs/demo/demo_combined.yaml

🔹 B. Train DVR on ShapeNet Chairs:
    Modify config: configs/single_view_reconstruction/multi_view_supervision/ours_depth.yaml

    Key Changes:
    - classes: ['03001627']  # ShapeNet chair
    - batch_size: (reduce to fit GPU)
    - lambda_normal: 0
    - color supervision: False

    Train:
    python train.py configs/single_view_reconstruction/multi_view_supervision/ours_depth.yaml

    Generate:
    python generate.py configs/single_view_reconstruction/multi_view_supervision/ours_depth.yaml

    Evaluate:
    python eval_meshes.py configs/single_view_reconstruction/multi_view_supervision/ours_depth.yaml

    View output: in `out/.../meshes` folder

===========================
📂 3. DEEPSDF USAGE GUIDE
===========================

🔹 A. Preprocessing ShapeNet Meshes:

    - Download ShapeNetCore.v2 and place `.obj` models in: `ShapeNetCore.v2`
    - This project used class ID `'03001627'` (chairs)
    - Use the custom preprocessing script to convert meshes into SDF samples:
        python preprocess_chairs.py
    - Output: Signed distance samples stored in `./processed_data/chair_train/`

    Notes:
    - The script uses `sample_sdf.py` to generate SDFs for each mesh
    - We used only **50 chair models** from ShapeNet to reduce training time

🔹 B. Training the Decoder:

    Run:
        python model/train_chair.py

    Training configuration:
    - Epochs: 822
    - Latent vector size: 256
    - Batch size: 10
    - Learning rate (decoder): 0.0005
    - Learning rate (latent vectors): 0.001
    - Training loss converged to: **0.0015**
    - Training time: ~1.6 hours (on 8GB GPU)
    - Model checkpoint saved to: `./checkpoints/chairs/model_epoch_822.pt`
    - Model file size: ~23MB
    - Total parameters: ~1.8M

    Notes:
    - `train_chair.py` uses `model/train.py` logic
    - TensorBoard logs were stored in `runs/chair/`
    - No encoder used — latent vectors were learned per-shape

🔹 C. Mesh Reconstruction:

    Run:
        python reconstruct.py

    Output:
    - Meshes saved to `./reconstruction/chairs/`
    - Each `.ply` corresponds to one ShapeNet chair (e.g., `example_364.ply`)
    - GT (ground truth) mesh IDs are matched via `split.npy` index mapping

    Evaluation metrics:
    - Average Chamfer Distance: **239.33**
    - Average IoU: **0.005**
    - Inference time: ~22.17 sec per shape
    - GPU memory usage: ~684 MB

    Notes:
    - Low IoU is expected due to implicit representation + voxelization losses
    - Chamfer Distance is the more reliable metric here
    - Resolution of Marching Cubes and voxel size were tuned (voxel_size = 0.05)

    Tip:
    - Mesh IDs like `example_364.ply` match to GT via `split.npy` → `beae73bea87f36a8f797b840852566c8`

===========================
📊 4. COMPARISON SETUP
===========================

- Both methods trained on ShapeNet "chair" category
- DVR: depth-only supervision (no color, normals)
- DeepSDF: latent auto-decoding (no encoder)
- Metrics evaluated: Chamfer Distance, IoU
- GPU Memory & Inference Time tracked with TensorBoard and manual logging

===========================
❗ 5. TROUBLESHOOTING NOTES
===========================

- WSL users: Avoid OpenGL (Pangolin) in GUI apps; use command line only
- CUDA errors? Lower batch size or check PyTorch/CUDA versions
- If preprocessing fails in DeepSDF, verify all `model_normalized.obj` paths are correct
- For DVR, ensure dataset directory structure matches what's expected in config