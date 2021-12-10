# EC601_RSDiverisificationProject
In this project we focus on building a tool that evaluates the performance of some existing Recommendation Algorithms. Instead of focusing on accuracy, we turn our focus on a less common but much needed metric, diversity. Using the summary and tools found the open source [Microsoft Recommenders](https://github.com/microsoft/recommenders) library. we inspect how diverse recommendations by different algorithms are.


## Setup and Dependencies
All code in this repository is heavily based on the jupyter notebooks, tools and functions of the [Microsoft Recommenders](https://github.com/microsoft/recommenders) library. For detailed setup of necessary dependencies and installations required to rerun the code, please see the [Microsoft Recommenders setup guide](https://github.com/microsoft/recommenders/blob/main/SETUP.md).

## Layout of git repo
All our jupyter notebooks with the algorithms and their diversity/accuracy evaluation are found in the <i>/experiments</i> folder. At the top of each notebook you will find the link to the corresponding Microsoft Recommenders notebook we based off on.

In <i>/recommenders_functions</i> you can find our mini-library, the functions we created to pipeline the evaluation process, along with a notebook testing those functions.

In other folders you will find results in various format and in the main repository our poster and an overview of the algorithms.

## The Project
1. [Background](#Background)<br>
2. [Project Goals](#Projectgoals) <br>
3. [Methods/Tools](#Methods) <br>
4. [The Metrics](#Metrics) <br>
5. [Challenges](#Challenges)<br>
6. [Results](#Results)<br>


<a name="Background"></a>
## Background
In the modern world we are surrounded by recommendation systems of various kinds. The most common drive behind the develeopement and research of these systems is to aim for more user satisfaction by improving accuracy of the recommendations [[2]](#2). That is, improving how well they fit together with the users pre-existing interests.

In recent times however, questions have started to emerge regarding the merit of this metric. Social and political events have sparked widespread discussions on the dark sides of recommendation systems. So called "echo chambers" where users get sucked into a loop of reinforcement promoting monotone content are now a well known concept and a topic of disucssion and concern [[3]](#3)[[4]](#4).

<p align="center">
<img src="https://user-images.githubusercontent.com/54936808/145511830-31ff08f4-ab24-43cf-a84a-b202f906b933.jpg" width="225" height="170">
</p>


Media giants like Instagram and Google have openly started acknowledging the problem [[5]](#5)[[6]](#6). For example, in their paper "Degenerate feedback loops in recommender systems", the researhcers at Google's DeepMind found that degenerate feedback loops are <i>intrinsic features</i> of recommendation systems [[6]](#6). The problem, it seems, is therefore rooted in the designs themselves and the challenge is raised to develop more diverse recommendation algorithms for the future.

As frequent users of these systems we, the project members, have noticed this ourselves and feel a great need for change. This experience of ours, along with the visible impacts these things are having in a societal context were our motivation to choose this topic. We set out on a mission to explore diversity in recommendation systems. Which algorithms have the potential to counteract the echo chambers?   

<a name="Projectgoals"></a>
## Project Goals
The topic of diversity in recommendation systems, just like the topic of recommendations sytems itself, is a very extensive one. Though it would be exciting to conquer it all, it was clear from the beginning we would need to narrow down our concentration. After doing some literature review, one of the things we found was that despite many researches agreeing on the importance of diverisification, few agree on the metrics. There seems to be little shared consensus or a trend, making the comparison of resluts from different research groups hard [[7]](#7). With that in mind we defined our mission: 
- Take part in shaping a better recommendation environment for the future by designing a tool that measures diversity in recommendation algorithms 
- Contribute to increased algorithmic control of recommendation systems and the emphasis on new metrics
- Create a library or set of models that evaluates the diversity of the recommendations from prominent algorithms. Give insight on how they compare to accuracy (tradeoff, correlation)

<a name="Methods"></a>
## Methods/Tools

- Various functions and tools from the Microsoft Recommenders library[[1]](#1)
- Jupyter Notebook
- Apache Spark
- Tensorflow
- BU SCC.
- And more...


<a name="Metrics"></a>
## The metrics
Diverisity is defined as the opposite of similarity. Where similarity is calculate with cosine similarity [[1]](#1).

<p align="center">
<img src="https://github.com/nannkat/EC601_RSDiverisificationProject/blob/main/images/the_metrics.png" height="400" >
</p>

Basing off of this formula, two types of diversity can be evaluated:<br>

<b>Co-Occurence Diversity</b>. The greater the proportion of preferring users two items have in common, the greater the similarity value and consequently the smaller diversity value they have.<br>

<b>Item-Feature Diversity</b>. How diverse features of items recommended to each user are on average.<br>

For each algorithm we calculate a score for both types of diversity.

<a name="Challenges"></a>
## Challenges
Though finding the Microsoft Recommenders gave us a good start, we soon discovered that their diversity metric tools were of a certain format. That certain format was not immediately generalizable to all the algorithms. In their example notebooks the Microsoft Recommenders use a wide variety of approaches, data types and toolkits so as to best represent the uniqueness of each algorithm.

<a name="Results"></a>
## Individual Algorithms and Results
As a result of some of the challenges stated above, our original dream of creating a unified library was not realistic at this time point. We therefore made a data-processing tool kit and summarize functions that greatly helped in automating the process.

The workflow:

<p align="center">
<img src="https://github.com/nannkat/EC601_RSDiverisificationProject/blob/main/images/the_workflow.png" >
</p>

We looked at the following algorithms:

<p align="center">
<img src="https://github.com/nannkat/EC601_RSDiverisificationProject/blob/main/images/the_algorithms.png" >
</p>

Below is a summary of our results in graph and table format. You can see both ranking metrics like precision/recall along with the diversity metrics. The RBM, Restricted Boltzman Machine, a probability distribution based algortihm was the one with the highest diversity score.

<p align="center">
<img src="https://github.com/nannkat/EC601_RSDiverisificationProject/blob/main/images/results_lineplot.jpg" >
</p>

<p align="center">
<img src="https://github.com/nannkat/EC601_RSDiverisificationProject/blob/main/images/the_results.png" >
</p>

For more detailed information on each algorithm, see our Jupyter Notebook experiments or the Microsoft Recommenders webpage.


## Working Sprints presentation links

Sprint 1: https://docs.google.com/presentation/d/1o7llqj_9ZRVHWVrhb6qLi9XfL2Tas5Li2nFZYkOjB5U/edit#slide=id.gcb474cc876_0_80 <br>
Sprint 2: https://docs.google.com/presentation/d/1tWMvDGobAUQmjLdnaEzuXU_a6DnrlS4aJ8MGa1LFvfE/edit#slide=id.gcb474cc876_0_90 <br>
Sprint 3: https://docs.google.com/presentation/d/1pG3jrlp_FT0KQwaV9ZG8MY3prjQyXYWFjUIOgfQ5-30/edit#slide=id.gfbb787297b_0_135 <br>
Sprint 4: https://docs.google.com/presentation/d/1fcyD5Jyr3t3QaRcmBgOYTkUlopxi72uap5pVuB21WiM/edit#slide=id.gfbb787297b_0_0


## References
<a name="1"></a>
[[1]](https://github.com/microsoft/recommenders) Microsoft (2021) Recommenders [Source code]. https://github.com/microsoft/recommenders.<br>
<a name="2"></a>
[[2]](https://link.springer.com/article/10.1007%2Fs00521-019-04128-6) Logesh, R., Subramaniyaswamy, V., Vijayakumar, V. et al., "Hybrid bio-inspired user clustering for the generation of diversified recommendations", <i>Neural Comput & Applic</i>, vol.32, pp. 2487–2506, April 2020. [Online]. Available: https://doi.org/10.1007/s00521-019-04128-6 [Accessed 14.09.2021].<br>
<a name="3"></a>
[[3]](https://www.theguardian.com/science/blog/2017/dec/04/echo-chambers-are-dangerous-we-must-try-to-break-free-of-our-online-bubbles) D. R. Grimes,"Echo chambers are dangerous –  we must try to break free of our online bubbles", <i>The Guardian</i>, December 04, 2017. Available: https://www.theguardian.com/science/blog/2017/dec/04/echo-chambers-are-dangerous-we-must-try-to-break-free-of-our-online-bubbles [Accessed 19.09.2021].<br>
<a name="4"></a>
[[4]](https://www.washingtonpost.com/technology/2021/07/22/facebook-youtube-vaccine-misinformation/) G. De Vynck, R. Lerman, "Facebook and YouTube spent a year fighting covid misinformation. It’s still spreading.", <i>The Washington Post</i>, July 22, 2021. Available: https://www.washingtonpost.com/technology/2021/07/22/facebook-youtube-vaccine-misinformation/ [Accessed 19.09.2021].<br>
<a name="5"></a>
[[5]](https://about.instagram.com/blog/engineering/on-the-value-of-diversified-recommendations) A. Mahapatra, "On the Value of Diversified Recommendations",
, <i>Instagram</i>, updated December 17, 2020, [Blog], Available: https://about.instagram.com/blog/engineering/on-the-value-of-diversified-recommendations [Accessed 09.09.2021]].<br>
<a name="6"></a>
[[6]](https://arxiv.org/abs/1902.10730) R. Jiang, S. Chiappa, T. Lattimore, A. György and P. Kohli, "Degenerate feedback loops in recommender systems", in <i>Proceedings of AAAI/ACM Conference on AI, Ethics, and Society, Honolulu, HI, USA, January 27-28, 2019</i> pp. 383-390.<br>
<a name="7"></a>
[[7]](https://www.sciencedirect.com/science/article/abs/pii/S0950705117300680) M. Kunaver and T. Pozrl, "Diversity in recommender systems – A survey", <i>Knowledge-Based Systems</i>, vol. 123, pp. 154-162, May 2017.[Online]. Available: https://www.sciencedirect.com/science/article/abs/pii/S0950705117300680.<br>

