# ML-based Data Matrix decoding
This project seeks to create a Data Matrix decoder that relies on ML based methods.

Weekly supervisor meetings occur on Thursdays at 13:30.

# Quick Links
* [Week 1](#week-1---6-feb-2025)
* [Week 2 to 4](#week-2-to-4---27-feb-2025)
* [Week 5](#week-5---6-march-2025)
* [Week 6](#week-6---13-march-2025)
* [Week 7](#week-7---20-march-2025)
* [Week 8](#week-8---27-march-2025)
* [Week 9](#week-9---3-april-2025)
* [Week 10](#week-10---10-april-2025)
* [Week 11 to 12](#week-11-to-12---24-april-2025)
* [Week 13](#week-13---1-may-2025)
* [Week 14](#week-14---8-may-2025)
* [Week 15](#week-15---15-may-2025) <-- current
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

# Week 2 to 4 - 27 feb 2025
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
### YOLO Improvement ✔️:
Yucheng proposed that if we alter the loss calculation of the YOLO rotation to be clamped between 0-90, we can have the model be uninterested in terms of rotations of 90, 180, 270 degrees etc., and learn the bounding boxes better.
E.g. the angle 91 would become 1, and 180 would become 90.

To make this change I will have to delve into the YOLO code and find where I can change this loss calculation. I may have to set up the yolo files locally, which could be a pain.

### YOLO class changes ✔️:
We can further utilize the YOLO model by involving class predictions for whether a DMC is laser or dot-peen marked. This is an easy change and just involves going through the annotations in roboflow again.

## Outcome of Week
### YOLO class changes
YOLO class changes were made and the class predictions work well :)

### YOLO Improvement
A custom OBBTrainer class was made, which copies the original ultralytics OBBTrainer but adds to the preprocess_batch function. Now the preprocess_batch does all of the original preprocessing, but the ground truth angle is folded to be between 0-90 degrees.

Strangley, this change fails to force some values in between the range. Some rotation values .01 above 90 appear (which is fine), but some values appear very close to 180, instead of being 0.

Also, when forcing the angle to be 0, the model still sometimes predicts angles outside of this range. Perhaps training from scratch will make better use of this method, instead of using the pretrained model.

### Extra: YOLO + Hourglass Training
I did some experimenting with YOLO cropping followed by hourglass corner prediction. Initial results look very good!

# Week 11 to 12 - 24 april 2025
## Goals
### YOLO fixes :x:
Need to add the same changes I did to the training process, but to validation and test.

### Template Matching (Thx Yucheng!) ✔️:
Yucheng showed me a promising method to help decode dot-peen marked components!

Basically, given a DMC, his method will (from memory, may be a bit incorrect, he will send me example code):
- Use Retinex to reduce lighting noise by getting reflection map
- Use OpenCV methods for estimating a valid template. A template in this case is e.g. a crop of a single dot-peen mark, representing how all other dot-peen markings should look.
- Template is then used for template matching. This process finds all other dot-peen markings that are similar to the template dot-peen.
- Once matched, most (hopefully all, but previous methods can be improved) dot-peen marks will be found and can be used for estimating a grid for the DMC.
- The grid can be used to decode the DMC without the decoder.
- Alternatively, the dot-peen marks locations could be used to rectify and transform the image in a way that the standard decoder can use.

I need to mess around with Yuchengs code a bit to understand how it all works, but I should also read material and watch videos explaining how different methods (e.g. template matching) work!

But, first thing I should do is:
- find all dot-peens that fail to decode (likely all)
- use Yucheng code
- try to build a grid for decoding

## Outcome of Week
### YOLO Fixes
It turns out that the YOLO fixes are not required and that the orientation works differently than I expected.

The YOLO OBB model does not predict the orientation in the range of 0-360, but actually only in the range of 0-180. The angle is designed only as a way to reduce background noise in produced bounding boxes, instead of it being to actually know how an object is oriented. In a sense, this means the model cannot predict e.g. from an aerial image of a car, which way the car is facing, but will simply orient the bounding box nicely around the car. The image below shows how the angle is represented.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/angle_representation.png" width="500">

What's strange to me, is that even though the predicted angle is explained as being between 0-180 in that image, during training no angle is ever above 90 degrees. I believe that the way Ultralytics processes the label input from "x1 y1 x2 y2 x3 y3 x4 y4" to "x y w h r" is done in a way where max possible r is 90, which should mean the model is penalized for predicting above 90 degrees? Despite this, the model will (in rare cases) predict angle in the 90-180 range. Confusing.

Bottom line is that YOLO obb will not be able to predict the exact orientation of DMCs, so we will either have to decode on multiple orientations or have an extra step using the stacked hourglasses for orientation.

### Template Matching
Successfully decoded using template matching with manually retrieved template! Process with example below.

First, image is altered using retinex to make lighting more uniform. Then, given a template, matching algorithm tries to match contours of template in image to create bounding boxes on all similar blobs in image.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/template_matching/template_matching.png" width="750">

KMeans clustering is used on the x and y dimensions (separately) to estimate which blobs belong together to estimate a grid. Here, n_clusters was given as 16 as most if not all DMCs in the dataset are 16x16.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/template_matching/grid_estimation.png" width="250">

The intersections of the gridlines are retrieved.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/template_matching/grid_intersections.png" width="250">

Then, for each bounding box found earlier, the grid intersection closest to it are found.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/template_matching/closest_intersections.png" width="250">

Finally, since we now know which bounding box belongs to which grid intersection, we can rebuild the original DMC.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/template_matching/rebuilding_dmc.png" width="250">

Notice however in this example there is an error on 1 module. This error occured during the closest intersection calculation, as there is some slight warping present on the DMC near the bottom of it where it seems to shear slightly to the right. In this example this problem is luckily solved, as the module is part of the alignment pattern and can therefore be fixed before actual decoding. The chances of the problem occuring can be reduced by using more advanced methods for gridline estimation, such as linear regression after KMeans clustering.

Currently the actual decoding is still done with pylibdmtx, where the rebuilt DMC matrix is converted to an image and fed to the library's decoding algorithm. However, this can be avoided by finding and rewriting the original libdmtx code that converts DMC matrix to interesting data, I just haven't done it yet because even though it is a very specific process it is quite complex.

# Week 13 - 1 may 2025
## Goals
### UNet for Finding Templates :x:
For finding initial templates for template matching, methods used for detecting braille dots can be used.

I will bulk train a UNet model on braille datasets and finetune on MAN data. The UNet model will create heatmaps of the dot detections, which I can then use for getting templates for the template matching algorithm.

Braille datasets from snoop2head github. I will have to create the labels for the MAN images, but I can speed up the labelling process by manually cropping a couple templates for each image and using template matching to get the other labels. The label format is lines of x y coords.

Will combine all braille datasets together for bulk training, LR in finetuning will be reduced by factor 10 (may alter this if poor results).

### Better Grid Fitting ✔️:
Current grid fitting is a bit too simple and should be improved. Yucheng informed me there are different grid fitting algorithms out there so I will look into that later on.

### Template Matching Improvements :x:
Depending on performance on template matching, will maybe have to alter it to get more templates from template matches, so maybe 2-3 runs of template matching with newer templates.

### Decoding Pipeline :x:
If there's time, I should finish up the decoding pipeline:
- YOLO oriented crop (may rollback to non oriented YOLO if orientations are poor)
- UNet for creating dot heatmaps of likely good templates
- Method for getting templates from heatmap. Method will likely have to find some heatmap dots close to eachother (e.g. an area where there are 2x2 dots or L dots or sequential dots), then average around them to get 2-4 good templates.
- Template matching
- Estimate grid
- Decode

## Outcome of Week
### UNet for Finding Templates
Model training and finetuning set up. Manual MAN labels still in progress.

Problem with MAN patch preparation needs to be fixed.

### Better Grid Fitting
Better grid fitting achieved!

New grid estimates works first by creating a 16x16 grid template with decent starting parameters.
| Parameter  | Interpretation                                           | Starting Point Estimation                                                    |
| ---------- | -------------------------------------------------------- | ---------------------------------------------------------------------------- |
| xΘ         | origin x coord of grid (grid built from top left corner) | Estimated from most top-left matched templated                               |
| yΘ         | origin y coord of grid (grid built from top left corner) | Estimated from most top-left matched templated                               |
| a, d       | spacing between gridlines                                | Estimated from average point euclidean distances in neighborhoods            |
| b, c       | b, c: perspective                                        | (still fixing) Estimated from average slopes between points in neighborhoods |

Then a local minimization of the parameters is done based on the mean squared distance between observed points and grid template points. With a good enough starting estimate the local minimization is very effective, but with too poor starting parameters the local minima is not the global minima and the minimization fails.

Current implementation of origin estimation and hard coded abcd parameters means the current grid fitting works on -4 to 4 degree rotated images but hopefully much better with other parameter estimations.

# Week 14 - 8 may 2025
## Goals
### UNet fixes ✔️:
Changes to be made for UNet finetuning:
- use crops of MAN images based on labels
- for every crop make a template (from the crop itself!)
- in patch generation code, change heatmap generation to be based on size of template
- could add an extra line at the top of labels to denote size of heatmap (e.g. 5 for 5x5 templates)

### Better Grid Fitting Changes ✔️:
Changes to be made for grid fitting steps:
1. Change grid template to only include finder pattern
2. estimate for top left point (already done)
3. estimate for grid spacing (already done)
4. estimate for rotation
   - first try for 45 degrees (in steps of 5) to find good initial 90 deg rotation
   -  then try every 90 degree rotation to find best orientation
5. run minimization algorithm to find local minima
6. draw gridlines based on best fitted finder pattern
7. match bounding box centers with drawn grids, possibly with from outer-inner method that i used in the past

### Template Matching ✔️:
Depending on performance on template matching, will maybe have to alter it to get more templates from template matches, so maybe 2-3 runs of template matching with newer templates.

### Decoding Pipeline :x:
If there's time, I should finish up the decoding pipeline:
- YOLO oriented crop (may rollback to non oriented YOLO if orientations are poor)
- UNet for creating dot heatmaps of likely good templates
- Method for getting templates from heatmap. Method will likely have to find some heatmap dots close to eachother (e.g. an area where there are 2x2 dots or L dots or sequential dots), then average around them to get 2-4 good templates.
- Template matching
- Estimate grid
- Decode

## Outcome of Week
### Cascade Template Matching
Template matching has been improved.

The improvement works by running an initial template matching as normal, taking some of the new matches as new templates, and then rerunning the template matching with those.

The selecting process for new templates is to take the top K most diverse (different) templates from the matches of the initial template matching. The diversity is based on each candidate templates "Hu Moment", which are numbers that characterize the shape of an object.

Below images showcase cascade matching at different values of K for an image with an imperfect initial template.

| Initial Template Matching | Cascade K=1 | Cascade K=2 | Cascade K=3 |
|:-------------------------:|:-----------:|:-----------:|:-----------:|
|<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/cascade_template_matching/single_template_matching.png" width="250">|<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/cascade_template_matching/cascade_template_matching_k1.png" width="250">|<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/cascade_template_matching/cascade_template_matching_k2.png" width="250">|<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/cascade_template_matching/cascade_template_matching_k3.png" width="250">|

### Better Grid Fitting
A rewrite of the grid fitting achieves much better results that previous. With perfect template matching the grid fitting now works for all 360 deg rotated images.

The grid matching works by generating a standard DMC grid template (finder pattern on the outside but all inner points present), estimating some initial position for the grid and then optimizing the grids parameters.

Changes to parameters used for grid estimation:
| Parameter  | Interpretation                                           | Starting Point Estimation                                                                                            |
| ---------- | -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| x0, y0     | center (x,y) coords of grid (grid built from center)     | Estimated by calculating centroid of matched templates                                                               |
| sx, sy     | horizontal/vertical spacing between gridlines            | Estimated from points that form valid "L" shapes with distances <= mode distance of all closest template pairs       |
| theta      | rotation of grid (radians)                               | Estimated from points that form valid "L" shapes with distances <= mode distance * 2 of all closest template pairs   |

Valid "L" shapes are triplets of template matches that meet the following requirements:
- Form an angle close to 90 degrees (current leeway is +- 10 degrees)
- Distance between "middle" point and other two points does not exceed the modal distance of closest points

The below image shows a valid L shape (green) an L shape only valid for theta calculation (yellow) and an invalid L shape (red).

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/grid_fitting/L_shapes.png" width="250">

The red L is invalidated due to it's angle.
The yellow L is invalidated for (sx, sy) calculation due to it's distances, but a looser distance requirement is nice to have for theta calculation to have more L shapes overall. The incorrect theta from this particular L shape is accounted for later by taking the median thetas of all green + yellow L shapes.

Below is an example of how a typical starting estimation for the grid looks. The estimated starting parameters were (227.3, 240.0, 17.0, 17.0, 0.79).

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/grid_fitting/start_estimation.png" width="250">

After a good starting grid is found, the parameters are optimized based on the MSE of the closest grid-point to each template matching. More weight is put onto template matches further away from the template matches centroid, so that the optimization process prefers to keep the grid within the template matches area.

The optimization process works in steps that provide a better optimization than optimizing for all parameters at once:
0. Check other three 90 degree angles to see if a lower-cost starting angle exists
1. Optimize for center of grid
2. Optimize for spacing of grid
3. Optimize for angle of grid
   - (may remove this in future) Other three 90 degree angles checked again to seee if they have lower costs
4. Optimize for all parameters at once

Below table shows how the optimization looks at each step.
| Starting Estimate | Optimize for Center | Optimize for Spacing | Optimize for Angle | Optimize for All |
|:-----------------:|:-------------------:|:--------------------:|:------------------:|:----------------:|
|<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/grid_fitting/start_estimation.png" width="250">|<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/grid_fitting/opt_center.png" width="250">|<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/grid_fitting/opt_spacing.png" width="250">|<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/grid_fitting/opt_angle.png" width="250">|<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/grid_fitting/opt_all.png" width="250">|

After optimizing the grid, the grid points are mapped to the template matches and the decoding process is done. Mapped grids and resulting DMC shown below.

| Mapped Grid Points | Resulting DMC |
|:------------------:|:-------------:|
|<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/grid_fitting/mapped_grid.png" width="250">|<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/grid_fitting/resulting_dmc.png" width="250">|

Cool GIF showing the process.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/grid_fitting/ezgif.com-animated-gif-maker.gif" width="250">

Note that on some angles the methods fail! This happens if the initial start is not good enough. In this case the optimization process may have a row/col that sticks out of bounds of the matched templates. I have thought of a couple methods that could deal with this issue, but will only implement it if time allows.

The first method is to increase the dimensionality of the grid template. In theory, we don't actually need the dimensionality of the grid template to match the DMC. An oversized grid template _should_ still optimize well over the dots. We can then invert the dots the same way, and post-process to remove empty rows/columns from the DMC matrix, giving us back the 16x16 grid.

The second method would be to change the minimization objective. Instead of minimizing observed dots to their closest grid template dots, we could minimize observed dots to their 4 nearest grid dots. If we add an extra couple rows and colums to the grid template (or implement the first method), the optimization process may be smoother, and we would end up with the grid template making "squares" around each dot. We can still just as easily invert these dots back to DMC matrix and decode from there.

### UNet Fixes
The following changes were done to finetune the UNet model on MAN images:
- labelling of dot x,y coords completed
- patch augmentations done after cropping down to DMC area
- manual crops of dots saved for sizing the heatmap gaussian dists correctly during generation
  - patch labels include line at top for t_size. If False (for none MAN training image) a default size is used

Some work still needs to be done to get better finetuning results, but the finetuning does not completely kill the models predictions anymore.

Braille.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/UNet_finetuning/braille_1.png" width="500">

Finetuned.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/UNet_finetuning/finetuned_1.png" width="500">

Braille.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/UNet_finetuning/braille_2.png" width="500">

Finetuned.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/UNet_finetuning/finetuned_2.png" width="500">

Braille.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/UNet_finetuning/braille_3.png" width="500">

Finetuned.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/UNet_finetuning/finetuned_3.png" width="500">

### Decoding Pipeline
Decoding pipeline close to being completed.

Still working on method for extracting template fron UNet output, currently can extract template from low-res UNet input (384x384) but would like to keep original image quality for optimal template matching.

Desired decoding pipeline:
- Load original image directly to tensor
- YOLO inference to get two crop versions. xywhr converted back to original image space to keep high quality:
  - YOLO crop           (for use as input to UNet)
  - YOLO crop + padding (for use in later template matching)
- UNet inference on YOLO crop to get coords for template area. YOLO crop used to reduce false positives (crop is sometimes too small). coords converted back to original YOLO crop space to keep high quality (not implemented yet):
- Template extracted from "YOLO crop + padding" to ensure entire DMC present.
- Cascade template matching
- Grid fitting
- DMC decoding

# Week 15 - 15 may 2025
## Goals
### Finish Pipeline ✔️:
Finish the full decoding pipeline and test thoroughly on more images with full debugging enabled.

Double check that same retinex used during UNet training is used in full pipeline.

Full debugging will show the decoding process at every step, allowing me to see what step causes failure cases.

On identification of specific failures will improve methods. E.g. if grid fitting is the only point of failure I will implement my previous ideas for improvement of grid fitting.

### Watershed Segmentation :x:
For template extraction, watershed segmentation will be useful to extract the templates on overlapping dots.

This, along with similar L finding method to validate dots, the Hu Moments can be used to select top K dot templates for use in template matching.

### UNet Training Change ✔️:
Mix MAN dataset with Braille during training. This may be better than simply finetuning on MAN data.

Can use sampler param in pytorch dataloader to ensure balance. Will also add debug information in labels (source of data) so that I can print during training to ensure each batch is 50/50 braille and MAN (ig they should be shuffled too).

### (Optional) Cascade Template Matching Change :x:
During cascade template matching, can do it only in the areas near template dots, e.g. around a radius of the template, but ensure that entire image is used overall.

This would make the process computationally cheaper but not necessarily more accurate, so I will only do this if required / time allows.

## Outcome of Week
### UNet Training Change
Braille and MAN datasets mixed by creating two pytorch datasets, oversampling the MAN dataset to have equal size to braille, concatting the two datasets, and feeding that into the dataloader.

The results look better than previous!

Braille.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/UNet_finetuning/braille_1.png" width="500">

Finetuned.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/UNet_finetuning/finetuned_1.png" width="500">

Mixed.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/UNet_finetuning/mixed_1.png" width="500">

Braille.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/UNet_finetuning/braille_2.png" width="500">

Finetuned.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/UNet_finetuning/finetuned_2.png" width="500">

Mixed.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/UNet_finetuning/mixed_2.png" width="500">

Braille.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/UNet_finetuning/braille_3.png" width="500">

Finetuned.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/UNet_finetuning/finetuned_3.png" width="500">

Mixed.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/UNet_finetuning/mixed_3.png" width="500">

### Pipeline Updates
The full pipeline has been put together with some added tweaks:
- template extraction method slightly altered to get better templates by looking at circular region of bright pixel instead of square
- initial template matching done with K templates (with NMS after), and another round of template matching done with K of those matches
- outlier removal of matched templates done with DBSCAN

Below is an example of full pipeline from image input to output decoded string.

YOLO cropping.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/pipeline/rotated_image.png" width="250"><img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/pipeline/yolo_detection.png" width="250"><img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/pipeline/yolo_crop_post_padding.png" width="250">

UNet dot finding.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/pipeline/reflectance_image.png" width="250"><img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/pipeline/illumination_image.png" width="250"><img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/pipeline/unet_heatmap.png" width="250">

Template matching. (only 2 templates shown...)

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/pipeline/template_matching.png" width="500">

Grid fitting.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/pipeline/initial_grid_fitting.png" width="250"><img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/pipeline/optimized_grid_fitting.png" width="250"><img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/pipeline/mapped_grid.png" width="250">

Decoding.

<img src="https://github.com/woodstr/msc-thesis/blob/main/figures/github_readme/pipeline/resulting_grid.png" width="250">

Decoded string (try on your phone): 1D1I65212740006

# Week 16 - 22 may 2025
# Week 17 - 29 may 2025
## Goals
### Report Writing
Report is due 2 June 2025.
