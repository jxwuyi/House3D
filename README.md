# This is a C++ re-implementation of House3D
This version is **_much faster_** and consumes **_orders of magnitudes less memory_**.
It is developed for our ICCV2019 project, *[Bayesian Relational Memory for Semantic Visual Navigation](https://github.com/jxwuyi/HouseNavAgent/)*. Please cite our ICCV paper for using the code. 

See comments for APIs and arguments of the updated [House class](https://github.com/jxwuyi/House3D/blob/C%2B%2B/House3D/house.py#L134) and [RoomNavTask class](https://github.com/jxwuyi/House3D/blob/C%2B%2B/House3D/roomnav.py#L118). They are both back-compatible.

#


# House3D: A Rich and Realistic 3D Environment
#### [Yi Wu](https://jxwuyi.weebly.com/), [Yuxin Wu](https://github.com/ppwwyyxx), [Georgia Gkioxari](https://gkioxari.github.io/) and [Yuandong Tian](http://yuandong-tian.com/)
*For questions regarding House3D contact Yuxin Wu*
#

<p align="center"><img src="https://user-images.githubusercontent.com/1381301/33509559-87c4e470-d6b7-11e7-8266-27c940d5729a.jpg" align="middle" width="600" /></p>

House3D is a virtual 3D environment which consists of thousands of indoor scenes equipped with
a diverse set of scene types, layouts and objects sourced from the [SUNCG dataset](http://suncg.cs.princeton.edu/).
It consists of over 45k indoor 3D scenes, ranging from studios to two-storied houses
with swimming pools and fitness rooms. All 3D objects are fully annotated with category labels.
Agents in the environment have access to observations of multiple modalities, including RGB images,
depth, segmentation masks and top-down 2D map views.

Usage instructions can be found at [INSTRUCTION.md](INSTRUCTION.md)

## Existing Research Projects with House3D
### A. RoomNav ([paper](https://arxiv.org/abs/1801.02209))

*Yi Wu, Yuxin Wu, Georgia Gkioxari, Yuandong Tian*

In this work we introduce a concept learning task, RoomNav, where an agent is asked to navigate to a destination specified by a high-level concept, e.g. `dining room`.
We demonstrated two neural models: a gated-CNN and a gated-LSTM, which effectively improve the agent's sensitivity to different concepts.
For evaluation, we emphasize on generalization ability and show that our agent can __generalize across environments__
due to the diverse and large-scale dataset.
<p align="center">
<img src="https://user-images.githubusercontent.com/1381301/33511103-ff5a71b4-d6c9-11e7-8f6d-95cc42e5b4e0.gif" align="middle" width="800" />
</p>

### B. Embodied QA ([project page](http://embodiedqa.org/) | [paper](https://arxiv.org/abs/1711.11543))

*Abhishek Das, Samyak Datta, Georgia Gkioxari, Stefan Lee, Devi Parikh, Dhruv Batra*

Embodied Question Answering is a new AI task where an agent is spawned at a random location in a 3D environment and asked a natural language question ("What color is the car?").
In order to answer, the agent must first intelligently navigate to explore the environment, gather information through first-person (egocentric) vision, and then answer the question ("orange").

<p align="center">
<img src="https://user-images.githubusercontent.com/1381301/33509618-f77bf844-d6b7-11e7-850a-b10ba6ef4a68.gif" align="middle" width="800" />
</p>
