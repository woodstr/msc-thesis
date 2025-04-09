# ML-based Data Matrix decoding
This project seeks to create a Data Matrix decoder that relies on ML based methods.

Weekly supervisor meetings occur on Thursdays at 13:30.

# Quick Links
* [Week 1](#week-1---6-feb-2025)
* [Week 2 to 4](#week-2-to-4---13-to-27-feb-2025)
* [Week 5](#week-5---6-march-2025)
* [Week 6](#week-6---13-march-2025)
* [Week 7](#week-7---20-march-2025)
* [Week 8](#week-8---27-march-2025)
* [Week 9](#week-9---3-april-2025)
* [Week 10](#week-10---10-april-2025) <-- current
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

# Week 2 to 4 - 13 to 27 feb 2025
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
### Stacked Hourglass Fixes :x:
After discussion with Yucheng, some problems with my synthesis were discovered and other good changes discussed.

Firstly, I have been generating the heatmaps incorrectly for augmented data. I had been applying shape and perspective transformations directly on the generated heatmaps, which is incorrect as the model should learn based on gaussian distributions of consistent size and shape, just in different locations. I should fix this so that the x,y coords for the location of these heatmaps is gotten by the augmentations, but the heatmaps should be generated afterwards so that they are round and consistently shaped.

Secondly, as the gaussian distributions cover small areas of the heatmaps, it may be beneficial to assign more weight on the gaussian distributions than the surrounding pixels during loss calculation. The section on the loss function from [this page](https://medium.com/towards-data-science/human-pose-estimation-with-stacked-hourglass-network-and-tensorflow-c4e9f84fd3ce) might be useful (thanks Yucheng).

Thirdly, trying with bigger heatmaps may also help.

### Small Optimization :x:
It should be possible to decrease actual training time by further utilizing my GPU. By loading the generated data to GPU before saving, I should be able to load the data back directly to the GPU, skipping the step of loading to CPU and then GPU afterwards. This is more of a niceness thing to reduce time messing around with different training parameters.

### Spatial Transformer Networks :x:
If there is time, spatial transformer networks should also be tried. They could be very useful in terms of rectification as their loss is calculated based on the comparison between the rectified image from model output and the ground truth image.

# Outcome of Week
### Stacked Hourglass Fixes
Made many fixes and changes to the stacked hourglass network and complex synthesis, but the models still fail to produce accurate heatmaps. Changes made include:
- Consistent heatmap generation. Heatmaps with single pixels are used before shape transformation, and only afterwards are used to generate full heatmaps. This ensures the heatmaps themselves are not warped.
- Std of generated heatmaps increased.
- MSE loss is weighted to increase loss for pixels close to center of heatmaps more than pixels away. Because I don't use "patches" for heatmaps, but distributions that cover the entire 64x64 heatmaps, I could not simply multiply over the patch, but raised the heatmaps to the power of 2. In this way the difference between lower and higher values is increased, effectively weighing as we desire.
- Heatmap generation error during shape transform identified, reduced and fixed. Shape transform could sometimes result in the corners of the DMC coming outside the image. When this happened the pixels for generating heatmaps would be lost, resulting in poisonous training examples. A check was implemented to ensure all 4 pixels are present after the transform, and if not, the process retried.
- Increased number of stacks to 8 to see if more model complexity was required.
- Minor error during training found and fixed.

Example of current produced heatmaps:
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/hourglass_complex_pred1.png" width="1500">
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/hourglass_complex_true1.png" width="1500">

### Small Optimization
While I could save from GPU and therefore load directly to GPU later, for some reason the augmentation process was slower on the GPU, so I've kept them on CPU.

### Spatial Transformer Networks
No time left to explore this.

# Week 6 - 13 march 2025
## Goals
### More Hourglass Fixes ✔️:
A fundamental mistake in the approach of stacked hourglass training has been identified. Losses should be calculated at every hourglass module, NOT just on the final output. These losses should be summed to perform backpropagation on. This is called intermediate supervision.

Other fixes/changes include:
- Normalizing heatmaps to range from 0-1
- Add threshold to gaussian distributions to only include 95% confidence interval (mu +- 2*sigma)
- Vizualize intermediate results (merge 4 corner heatmaps into 1). Then plot the results from each hourglass sequentially (1, 2, 3, ..., N)
- Start with 3 hourglasses
- Start without weighted losses

## Outcome of Week
## Successful Hourglass Results!
Above changes, along with the below extra changes lead to successfull hourglass models!
- RMSprop optimizer instead of Adam (RMSprop used in original paper)
- Extra training loop with learning rate reduced by factor 5 after validation plateau
- Shape transform rotations changed from any random angle to +-20 degrees. This was done because the hourglasses failed to distinguish between the corners on higher rotations, leading to each individual heatmap predicting all 4 corners. Potentially ok if we can detect orientation after, but for reduced scope to this change.

For some reason the train example failed, but it is a special case. The average loss across all train/val/test samples was relatively equal.

### Train Example
Predicted heatmaps
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/successful_shape/train_heatmaps_pred.png" width="1500">
True heatmaps
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/successful_shape/train_heatmaps_true.png" width="1500">
Hourglass heatmaps
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/successful_shape/train_hourglasses.png" width="1500">

### Validation Example
Predicted heatmaps
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/successful_shape/val_heatmaps_pred.png" width="1500">
True heatmaps
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/successful_shape/val_heatmaps_true.png" width="1500">
Hourglass heatmaps
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/successful_shape/val_hourglasses.png" width="1500">

### Test Example
Predicted heatmaps
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/successful_shape/test_heatmaps_pred.png" width="1500">
True heatmaps
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/successful_shape/test_heatmaps_true.png" width="1500">
Hourglass heatmaps
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/successful_shape/test_hourglasses.png" width="1500">

### Note
Note that a version trained with color transforms (blur, photometric distort, sharpness) yielded a failed model! Will need to look into this, as blur is very common when trying to scan these codes IRL.

# Week 7 - 20 march 2025
## Goals
New ideas for improving hourglass performance, and a fundamental change in its structure.

### Hourglass Fundamental Output Change ✔️
We realized that having 4 channels of heatmaps to produce is redundant in this case. After rectification, the orientation of the dmc is trivial, since we can attempt to decode 4 times (1 for each rotation of the code). Therefore, why not ask the model to simply produce a single heatmap with all 4 points in it? I have already seen this work when the augmentation includes all possible rotations, as it resulted in hourglass models that learned to produce, for each heatmap channel, all 4 points.

This will simplify the training process for the model, and hopefully lead it to be less confused with complicated orientations of dmc's, since the hourglasses no longer need to differentiate between corners (which is especially hard to do with blurred images).

Another change could be to introduce a second channel, as an inverted version of the first one. The weight mapping should also be inverted for this channel. This could further help the models performances.

### Small Heatmap Change ✔️
Try with bigger sigma (try with 2 first).

### Augmentation Changes ✔️
Changes can be done on image augmentations to potentially improve the model performance IRL:
- random shifting
- random scaling
- full rotations
If a corner ends up out-of-view, do not generate a gaussian dist for it. This will allow the model to give partial outputs, which may be useful to us when applying the model practically.

If full rotations performs badly, can slowly introduce more rotations per epoch (fx. every XX epoch increase possible rotations by 10 degrees).

### Loss Function Change :x:
Perhaps binary cross entropy could perform better for us, should try it out.

## Outcome of Week
### Hourglass Fundamental Output Change
Good results from changing to single heatmap output!

Train loader:

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/successful_new/train_true.png" width="250"> <img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/successful_new/train_pred.png" width="250">
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/successful_new/train_hourglass.png" width="1500">

Val loader:

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/successful_new/val_true.png" width="250"> <img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/successful_new/val_pred.png" width="250">
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/successful_new/val_hourglass.png" width="1500">

Test loader:

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/successful_new/test_true.png" width="250"> <img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/successful_new/test_pred.png" width="250">
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/successful_new/test_hourglass.png" width="1500">

### Small Heatmap Change
Increasing the std (sigma) to 2 to yield bigger heatmaps resulted in some similar results in the model with sigma 1, but with a poor example from a training sample.

Good result (from test loader):

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/sigma_examples/sigma_2_test_pred.png" width="250">
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/sigma_examples/sigma_2_test_hourglasses.png" width="1500">

Bad result (from train loader):

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/sigma_examples/sigma_2_train_pred.png" width="250">
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/sigma_examples/sigma_2_train_hourglasses.png" width="1500">

Without more thorough testing, reverted to sigma=1. Will possibly explore higher sigma later down the line when optimizing the model for real-time usage.

### Augmentation Changes
Introducing more complex augmentations resulted in poor model learning. However, starting with simple augmentations and increasing augmentation complexity linearly each epoch resulted in good model performance! Results further below. Curriculum training method was done as the following, for specific epochs:
- 0-9: no scale or translation augmentation
- 10: introduce scale and translation in very small amounts
- 11+ increase scale and translation by a small amount
- ...
- 150: (a reasonable) max scale and translation augmentation reached
- train until convergence
The introduction and increase of the two augmentations are done linearly.

Train loader:

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/curriculum_learning/train_pred.png" width="250"> <img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/curriculum_learning/train_true.png" width="250">
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/curriculum_learning/train_hourglass.png" width="1500">

Val loader:

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/curriculum_learning/val_pred.png" width="250"> <img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/curriculum_learning/val_true.png" width="250">
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/curriculum_learning/val_hourglass.png" width="1500">

Test loader:

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/curriculum_learning/test_pred.png" width="250"> <img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/curriculum_learning/test_true.png" width="250">
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/curriculum_learning/test_hourglass.png" width="1500">

Notes:
- The 4 corner extraction method from the heatmap is still primitive (it currently selects top 4 brightest heatmap points) so the predictions failed on the validation example here, but can easily be fixed.
- The test example failed to produce a good 4th corner heatmap. This is weird, but can perhaps be fixed with some minor changes to the current model and training methods used. It could also be possible to implement a method as part of the point extractor which guess a 4th point based on 3 other points.

### Loss Function Change
This has not been tried.

### Test on MAN Data
A basic test on real world MAN data shows that the model still needs extra work, especially when it comes to false positives.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/failed_real_test/pred.png" width="250"> <img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/failed_real_test/true.png" width="250">
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/failed_real_test/hourglasses.png" width="1500">

# Week 8 - 27 march 2025
## Goals
### Augmentation Changes ✔️
There is a mismatch between synthetic images and MAN images. The real world MAN images have text, symbols, and patterns around the DMCs, and because there are no examples of these in the synthetic dataset the hourglass models are detecting corners of these.

We can reduce this problem by introducing these elements in the synthetic dataset, specifically:
- generate randomly places strings and patterns/icons around the DMC
- these should be a part of the shape transform, as we see in the IRL images that they follow the same perspective etc.
- can also randomly change the color of the DMC since this changes IRL sometimes too

### Finetune Model ✔️
We can finetune the model to the MAN dataset to produce better results.

The train/val/test split of the dataset should be the same as in the 7.5ECTS research project for the same comparison.

### Retinex ✔️
We can use retinex to equalize the light in the input image before feeding to the model. These parameters will need to be manually tuned.

This could potentially improve the models performance for images with heavy reflections / changes in light across the image.

A comparison of the previous finetuned model should be made with a model finetuned to MAN images with retines preprocessing the images.

## Outcome of Week
### Augmentation Changes
Added randomized text and symbols around synthetic DMC images. It appears to have reduced the problem of false positives somewhat. However, there are still false positives present in many of the real images, but the heatmaps produced are more likely to have the brightest pixels in the correct locations. The model still fails with dot-peen markings.

Keep in mind that the point extraction method is still rudimentary.

Successfull extraction.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/text_synth/success.png" width="250">
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/text_synth/success_hourglasses.png" width="1500">

Somewhat successful extraction.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/text_synth/somewhat.png" width="250">
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/text_synth/somewhat_hourglasses.png" width="1500">

Failed extraction.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/text_synth/dotpeen.png" width="250">
<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/text_synth/dotpeen_hourglasses.png" width="1500">

### Finetuning
Finetuning the model proves difficult. With the same training method as the synthetic training, or with a reduced learning rate, the model fails to improve on the validation loss after the second epoch. I suspect we need more images for training, and therefore propose that I go to the factory this Friday to take more images of different components with the codes myself.

### Retinex
While retinex was implemented, there is no difference between a model trained with or without its use. Until there are better results on the real images, it will be hard to see the affect of retinex.

# Week 9 - 3 april 2025
## Goals
### Return of YOLO ✔️
Due to amount of false positive corners detected by stacked hourglasses, oriented YOLO will be tried as step 1 for cropping down to the DMC.

This new oriented model will be finetuned to the MAN data, and then used in the training process of a new stacked hourglass model.

On failed YOLO crops, the failed image is saved for further training the YOLO model to deal with failure cases.

## Outcome of Week
### Return of YOLO
The new oriented YOLO model works well enough for cropping. Some padding was added on its crops to reduce the rate at which it cuts off the corners of the DMC. There were still cases of the model cutting off the DMCs, or cropping to some falsely detected DMC.

These cases are currently still being saved as a stacked hourglass is still training.

The training process for the hourglass has become longer due to the use of YOLO _during_ training, but will hopefully yield more consistent results that previous stacked hourglasses.

# Week 10 - 10 april 2025
## Goals
### YOLO Improvement :on:
Yucheng proposed that if we alter the loss calculation of the YOLO rotation to be clamped between 0-90, we can have the model be uninterested in terms of rotations of 90, 180, 270 degrees etc., and learn the bounding boxes better.
E.g. the angle 91 would become 1, and 180 would become 90.

To make this change I will have to delve into the YOLO code and find where I can change this loss calculation. I may have to set up the yolo files locally, which could be a pain.

### YOLO class changes ✔️:
We can further utilize the YOLO model by involving class predictions for whether a DMC is laser or dot-peen marked. This is an easy change and just involves going through the annotations in roboflow again.

## Outcome of Week
### YOLO class changes
YOLO class changes were made and the class predictions work well :)

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
