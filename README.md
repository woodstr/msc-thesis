# ML-based Data Matrix decoding
This project seeks to create a Data Matrix decoder that relies on ML based methods.

Weekly supervisor meetings occur on Thursdays at 13:30.

# Quick Links
* [Week 1](#week-1---6-feb-2025)
* [Week 2](#week-2---13-feb-2025)
* [Week 3](#week-3---20-feb-2025)
* [Week 4](#week-4---27-feb-2025)
* [Week 5](#week-5---6-march-2025)
* [Week 6](#week-6---13-march-2025)
* [Week 7](#week-7---20-march-2025)
* [Week 8](#week-8---27-march-2025)
* [Week 9](#week-9---3-april-2025)
* [Week 10](#week-10---10-april-2025)
* [Week 11](#week-11---17-april-2025)
* [Week 12](#week-12---24-april-2025)
* [Week 13](#week-13---1-may-2025)
* [Week 14](#week-14---8-may-2025)
* [Week 15](#week-15---15-may-2025)
* [Week 16](#week-16---22-may-2025)
* [Week 17](#week-17---29-may-2025)

# Week 1 - 6 feb 2025
## Goals
### Initial Setup ✔️:
This includes repository, notes, communication methods, schedule, etc.

### Literature Review ✔️:
Reading papers covering similar topics, emphasis on rectification / decoding method as this is new.

### Roadmap :on:
An initial plan / idea should be written out to setup scope, requirements, and maybe a schedule for this project.

## Outcome of Week
### 4 Decoding Steps Proposition
**Step 1**: cropping with YOLO to reduce image size (increase efficiency). Basic version of this is DONE - but an oriented version could be better! I don't see a reason to remove this, it's a free performance boost.

**Step 2**: binarization with U-Net (deals with light / blur distortions). Basic version of this is DONE - but an improvement in speed would be nice.

**Step 3**: rectification with improved ResNet (deals with geometric distortions). Method from [this paper](https://www.researchgate.net/publication/378524337_An_efficient_distorted_QR_code_correction_method_based_on_improved_ResNet_architecture) could possibly be implemented. The papers modified ResNet model predicts control and reference points, which in turn are used to correct for geometric distortion. _May_ work on dot-peen markings, if not, perhaps separate rectification models are needed with a classification head for dot-peen / lazer. _May_ work on images with light/blur distortions, in which case step 2 _may_ not be required, but step 4 will become more complicated.

**Step 4**: decoding using grid approach (decodes DMC). If step 3 provides us with correct points, we can use them to overlay a grid on the reconstructed DMC. For each grid square we would need to predict if it is black or white. If the grid squares are accurate this is trivial due to our binarization. Note that if the binarization step is removed but grid squares can still be achieved, it may be useful to use the adjecency method from [this paper](https://www.mdpi.com/2076-3417/12/5/2291) for predicting the colors of grid squares. If points are inaccurate, an alternative method for finding points is required, or a method using [edge detection and line fitting](https://www.degruyter.com/document/doi/10.1515/cdbme-2016-0079/html) for finding the points could be used.

### Alternative Idea - Pose Estimation
Alternatively, perhaps a method similar to some form of pose estimation can be used for predicting x,y coords of every module of a given DMC to form the grid, which can then be used almost directly for decoding. If possible, this could _possibly_ skip binarization (unlikely) and rectification (maybe), and jump straight to decoding. This may be good because of the similarities between the two problems. Pose estimation is typically looking for interesting points on a person such as their joints, where the connectedness between joints is used in the models. With DMCs we have a similar connectedness, but between the modules of the DMCs instead of joints on the human body.

Pose estimation models:
- [YOLO11-pose](https://docs.ultralytics.com/tasks/pose/): potentially a good lightweight option. NO CLUE how it works though - would have to figure that out. Also unneccessarily handles multiple people in one image.
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose): Popular real-time pose estimation model. Unnecessarily handles multiple people as well.
- [RTMPose](https://arxiv.org/pdf/2303.07399): Similar to above.

### Alt Alt Idea
For proof of concept, can quickly test out if YOLO11n-pose works well with easy examples of DMCs. If so - we can conclude if pose estimation makes sense or not. YOLO11 simply because of its usability.

# Week 2 - 13 feb 2025
# Week 3 - 20 feb 2025
# Week 4 - 27 feb 2025
# Week 5 - 6 march 2025
# Week 6 - 13 march 2025
# Week 7 - 20 march 2025
# Week 8 - 27 march 2025
# Week 9 - 3 april 2025
# Week 10 - 10 april 2025
# Week 11 - 17 april 2025
# Week 12 - 24 april 2025
# Week 13 - 1 may 2025
# Week 14 - 8 may 2025
# Week 15 - 15 may 2025
# Week 16 - 22 may 2025
# Week 17 - 29 may 2025
## Goals
### Report Writing
Report is due 2 June 2025.
