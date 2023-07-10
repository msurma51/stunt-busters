# Stunt Hunting and Stunt Busting 

***An Exploration of Protection Against Pass Rush Games***

**Authors:** *Mark Surma* and *Luke Wiley*

Abstract:

Pass rush stunts have long been used as a tactic to generate pressure on the opposing quarterback. Over time the offensive line coaching community has developed a core set of philosophies on how to defeat these stunts. Here we put these three to the test: keeping rushers at the "same level", maintaining "square shoulders" and "stopping the penetrator". After isolating and classifying stunts pitting two rushers against two pass protectors, we determined that while each philosophy has merit they differ in utility. Ensuring that rushers have a low relative depth (are near the same level) is particularly useful when facing "ET" stunts but is much less relevant against "TE" stunts. For those two stunt types it benefits both protectors to stay square, but the offensive tackle has much less tolerance for outside rotation early in the rep. Conversely, on "TT" stunts the center's degree of "squareness" matters little before the stunt declares. Regardless of stunt type, impeding the penetrator is a worthy goal. In measuring feature importance using a variety of model algorithms we determined that these are the three most important factors in protecting against stunts, in reverse order of presentation above. One caveat, however, is that we found measuring a blocker's squareness with respect to the quarterback to be more informative than to the line of scrimmage. The speed of individual protectors emerged as a relevant aspect, with higher speed being associated with greater likelihood of rusher success. Whether or not protectors are able to exchange rushers also plays a role, with higher exchange rates being linked to successful protection in "ET" and "TT" stunts. Finally, we discuss how measuring these aspects and modeling pass rush win rate using all relevant features can be applied to player evaluation in opponent, self and pro scouting.

## Resources and Replication Info
**Python Version**: 3.11.2\
**Packages**: numpy, pandas, scipy, statsmodels, matplotlib, seaborn, sklearn, xgboost, shutil, torch, lassonet

To generate competition data and pre-processing, run "code/scripts/top_level_2v2.py".\
To generate all csv data used for project individual notebooks must be run to completion in "code/notebooks".\
To generate stunt dot gifs for all stunts, run "code/scripts/get_stunt_dots.py".\
Exploration, modeling and results can be viewed and tinkered with using Jupyter notebooks in "code/notebooks"\
Full project article can be downloaded [here](https://drive.google.com/file/d/1XKW4FOaKNCU4azP_XMdA5iec5BRkGnT9/view?usp=drive_link) and found in "code/notebooks/stunt-hunting-and-stunt-busting.html"
Abridged article can be found [here](https://www.kaggle.com/code/lwwiley17/stunt-hunting-and-stunt-busting)