# ML-based Data Matrix decoding
This project seeks to create a Data Matrix decoder that relies on ML based methods.

Weekly supervisor meetings occur on Thursdays at 13:30.

# Quick Links
* [Week 1](#week-1---6-feb-2025)
* [Week 2+3+4](#week-2+3+4---13-27-feb-2025)
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

### Roadmap ✔️:
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

### Quick & Dirty Roadmap
List of requirements in order:
- Finalize research decisions at week 2 meeting
- (lazer only - pose) Try with easy synthetic examples. Can model(s) estimate points correctly and decode them?
- (lazer only) Try new method(s) on already established MAN dataset. (Probably requires new annotation). Try with pre and post binarization - which works better?
- Implement custom decoder using point information.
- (method works for lazer) Analyze dot peen marking data from MAN. What kind of issues are present in them? Think about how to synthesize realistic dot-peen markings.
- Synthesize dot-peen marking examples.
- Train and see if models can generalize for both lazer and dot-peen. If not, may need separate models.

# Week 2+3+4 - 13-27 feb 2025
## Goals
After meeting, decided on new approach to try! Yucheng away for next 2 meetings so have made plenty of new things to code and try. Below two new methods are possibly robust for both lazer and dot-peen samples!

### YOLO Decoding :x:
With good DMC examples and consistent DMC dimensions, decoding could be possible purely with YOLO!

Would have to match the YOLO grid-size with DMC dimensions. This probably means we cannot use pretrained models.

By matching the grid-size, we can have YOLO predict the x,y centers of every black DMC module! If a module is detected, we know it is black. If a module is not detected, we know it is white. We inherently know where the modules belong, which allows us to decode when we know which modules are detected or not (by assigning appropriate matrix values as 1s or 0s - which we can feed to a decoder algorithm for max possible decode speed).

To try this approach, should first try with synthesized good examples, then later with (probably painfully) manually annotated examples to see if it works with the real-world example. Note that it almost definitely won't work with wrongly oriented and distorted samples, which the next method should fix.

### Stacked Hourglass Localization & Rectification :heavy_check_mark:
By using a stacked hourglass method for detecting the 4 corners of a DMC in an image, we can effectifely localize and rectify.

This rectification method works poorly for perspective distortions, but usually a user can point the camera more straight to the DMC so this may not be an issue.

Should try with predicting 4 channels with heatmaps of the corner locations (as heatmaps - probably gaussian distributed).

Probably start with 3 stacks as well, more and fewer should be experimented with though.

Can use previous research paper synthesis for making relevant heatmap labels for training. Will have to alter to introduce small codes, since we will no longer use YOLO for cropping but this method for (effectively) cropping and rectifying.

### Custom Decoding :x:
Depending on how things go, custom decoding methods mentioned in last weeks outcome could be useful.

## Outcome of Week
### YOLO Decoding
YOLO decoding harder than anticipated. Tried both with classic YOLO object detection for each module, and YOLO pose/keypoint detection with each module, but both resulted in absolutely no detections on the test set... will focus on hourglass method and revisit at a later time.

### Stacked Hourglass Localization & Rectification
High success with very simple synthesised examples! Example image below.
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/hourglass_simple.png" width="1500">

Low success with complex synthesised examples... example image below of predicted and true heatmaps.
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/hourglass_complex_pred.png" width="1500">
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/hourglass_complex_true.png" width="1500">
Notes on test set losses from tried parameters / settings:
- 0.06 with 3 stacks and varying epochs/patiences
- 2.09 with 6 stacks and varying epochs/patiences
- 36.09 with 6 stacks and early stopping patiences 1 and all epochs 100
- 0.16 with 3 stacks and varying epochs/patiences and wider heatmaps (sigma = 5 instead of old of 2)
- 2.34 with 3 stacks and varying epochs/patiences and wider heatmaps (sigma = 10 instead of old of 2)
- 0.13 with 3 stacks and varying epochs/patiences and wider heatmaps (sigma = 3 instead of old of 2)
- 0.15 with 3 stacks and varying epochs/patiences and wider heatmaps (sigma = 2 - should be exactly like the first run!)

Notes on things to try for improvements:
- Smaller heatmaps
- More different stacks
- Try with no augmentations (should be equal to simple viz) and then incrementally add more augments until it breaks again. This could help figure out what type of images it struggles with.
  - Perhaps doing shape transformations on the heatmaps is confusing for the model? Maybe after the shape transformations the heatmaps should be redone so that they are equally sized.
  - Perhaps train in a way where augmentations are gradually introduced as the model learns. The model can learn the simple synthesis easily, so perhaps learning that first and then adding other transformations could allow it to be robust to different types of complexities over time, instead of learning them all at once.
- Some optimization of gpu usage for faster training...
- ...?

# Week 5 - 6 march 2025
## Goals
### Stacked Hourglass Fixes :on:
After discussion with Yucheng, some problems with my synthesis were discovered and other good changes discussed.

Firstly, I have been generating the heatmaps incorrectly for augmented data. I had been applying shape and perspective transformations directly on the generated heatmaps, which is incorrect as the model should learn based on gaussian distributions of consistent size and shape, just in different locations. I should fix this so that the x,y coords for the location of these heatmaps is gotten by the augmentations, but the heatmaps should be generated afterwards so that they are round and consistently shaped.

Secondly, as the gaussian distributions cover small areas of the heatmaps, it may be beneficial to assign more weight on the gaussian distributions than the surrounding pixels during loss calculation. The section on the loss function from [this page](https://medium.com/towards-data-science/human-pose-estimation-with-stacked-hourglass-network-and-tensorflow-c4e9f84fd3ce) might be useful (thanks Yucheng).

Thirdly, trying with bigger heatmaps may also help.

### Spatial Transformer Networks :on:
If there is time, spatial transformer networks should also be tried. They could be very useful in terms of rectification as their loss is calculated based on the comparison between the rectified image from model output and the ground truth image.

# Outcome of Week
TBD.

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
