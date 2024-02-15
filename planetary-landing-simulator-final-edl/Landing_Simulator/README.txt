Author: Giulia Ciabatti, PhD @Sapienza, University of Rome 
with: Shreyansh Daftry, JPL@NASA and Roberto Capobianco, Sapienza, University of Rome and Sony AI. 

If you choose to use our simulator - or a customized version of it - please cite as: 

@InProceedings{Ciabatti_2021_CVPR,
    author    = {Ciabatti, Giulia and Daftry, Shreyansh and Capobianco, Roberto},
    title     = {Autonomous Planetary Landing via Deep Reinforcement Learning and Transfer Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2021},
    pages     = {2031-2038}
}

and/or: 

@InProceedings{Ciabatti2021,
    author    = {Giulia Ciabatti and Shreyansh Daftry and Roberto Capobianco}, 
    booktitle = {{ASCEND} 2021},
    title     = {Learning Transferable Policies for Autonomous Planetary Landing via Deep Reinforcement Learning}, 
    year      = {2021}, 
    month     = {nov}, 
    publisher = {American Institute of Aeronautics and Astronautics}, 
    doi       = {10.2514/6.2021-4006}, 
}

As first presented in: "Autonomous Planetary Landing via Deep Reinforcement Learning and Transfer Learning", G. Ciabatti, S. Daftry, R. Capobianco

Extended in: "Learning Transferable Policies for Autonomous Planetary Landing via Deep Reinforcement Learning", G. Ciabatti, S. Daftry, R. Capobianco

Real-Physics Simulator for Planetary Landing. 

- Physical Engine: Bullet/PyBullet -> bulletphysics.org
- Lunar and Martian 3D meshes for terrain development: from NASA's official repo -> https://github.com/nasa/NASA-3D-Resources
- Interfaceable with Gym 
- Lander architecture: .URDF (ROS-compatible)
- Titan terrain: reconstructed from SAR imagery retrieved during NASA'a Cassini-Huygens mission -> https://photojournal.jpl.nasa.gov/target/titan  
 - Lander's sensors: RGB camera, Depth camera, Segmentation mask, LiDAR (Position, Orientation)
 
- Try interactions with lander in chosen environment: run main_interactive.py
 
- Train lander/agent by means of Soft Actor Critic - SAC: run SAC_train_torch.py
  + implemented in Pytorch framework.
  + weights will be saved in SAC_weights after training session
  + find in SAC_images reward plots on different environments as presented in "Learning Transferable Policies for Autonomous Planetary Landing via Deep Reinforcement Learning", G. Ciabatti, S. Daftry, R. Capobianco

- Videos folder contains videos of controlled landing on the 4 environments. 

- Different environments are provided - i.e. Lunar environments, Mars and Titan both with and without atmospheric disturbances modeled as gusts. Chose by commenting/uncommenting as per instructions in scripts. 

Note: this README.txt is being updated and codes are being currently improved and further commented :)
