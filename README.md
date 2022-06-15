# A Survey on Rain Removal from Video and Single Image
[Hong Wang](http://hongwang01.github.io), Yichen Wu, Minghan Li, Qian Zhao, and [Deyu Meng](http://gr.xjtu.edu.cn/web/dymeng) 

[[Arxiv]](https://arxiv.org/abs/1909.08326)

## Citation
```
@article{WangA,
  title={A Survey on Rain Removal from Video and Single Image}, 
  author={Wang, Hong and Wu, Yichen and Li, Minghan and Zhao, Qian and Meng, Deyu}, 
  journal={arXiv preprint arXiv:1909.08326},
  year={2019}
}
```

## Physical Properties of Raindrops
* Gemometric Property 
  * Terminal velocity of raindrops aloft (JAMC1969), Foote et al [[PDF]](https://journals.ametsoc.org/doi/pdf/10.1175/1520-0450%281969%29008%3C0249%3ATVORA%3E2.0.CO%3B2)
  * A new model for the equilibrium shape of raindrops (JAS1987), Beard et al. [[PDF]](https://journals.ametsoc.org/doi/pdf/10.1175/1520-0469\(1987\)044%3C1509:ANMFTE%3E2.0.CO%3B2)
* Brightness Property
  * Photometric model of a rain drop (Technical Report, Columbia University2004), Garg et al [[PDF]](http://www1.cs.columbia.edu/CAVE/publications/pdfs/Garg_TR04.pdf)
  * Vision and Rain (IJCV2007), Garg et al [[Project]](http://www.cs.columbia.edu/CAVE/projects/camera_rain/)[[PDF]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.442.9985&rep=rep1&type=pdf) 
* Chromatic Property 
  * Rain removal in video by combining temporal and chromatic properties (ICME2006), Zhang et al [[Project]](https://www.comp.nus.edu.sg/~leowwk/demo/rain-removal.mpg)[[PDF]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.64.1760&rep=rep1&type=pdf)
* Spatial and Temporal Propety
  * Simulation of rain in videos (TAS2003), Starik et al [[PDF]](http://lear.inrialpes.fr/people/triggs/events/iccv03/cdrom/texture03/texture03-ab018.pdf)
  * Pixel based temporal analysis using chromatic property for removing rain from videos (CIC2009), Liu et al [[PDF]](https://pdfs.semanticscholar.org/27fe/cae9d02fbfbea2b0fc9f2577179fb359c8b1.pdf) 
## Video Deraining Methods
* Time Domain
  * Detection and removal of rain from videos (CVPR2004), Garg et al [[Project]](http://www.cs.columbia.edu/CAVE/projects/rain_detection/)[[PDF]](http://vc.cs.nthu.edu.tw/home/paper/codfiles/cjhung/200708090922/01315077.pdf) 
  * When does camera see rain? (ICCV2005), Garg et al [[Project]](http://www.cs.columbia.edu/CAVE/projects/camera_rain/)[[PDF]](http://www1.cs.columbia.edu/CAVE/publications/pdfs/Garg_ICCV05.pdf)
  * Rain removal using kalman filter in video (ICSMA2008), Park et al [[PDF]](http://xanadu.cs.sjsu.edu/~drtylin/classes/cs157A/Project/PDF-files/CS157B_Team13/13_XuanYu_documents/04505573.pdf)
  * Using the shape characteristics of rain to identify and remove rain from video (S+SSPR2008), Brewer et al [[PDF]](https://link.springer.com/content/pdf/10.1007/978-3-540-89689-0_49.pdf) 
  * The application of histogram on rain detection in video (JCIS2008), Zhao et al [[PDF]](https://download.atlantis-press.com/article/1730.pdf)
  * Rain or snow detection in image sequences through use of a histogram of orientation of streaks (IJCV2011), Bossu et al [[PDF]](http://hautiere.nicolas.free.fr/pdf/2011/hautiere-ijcv11.pdf)
  * A probabilistic approach for detection and removal of rain from videos (IETE JR2011), Tripathi et al [[PDF]](https://www.tandfonline.com/doi/abs/10.4103/0377-2063.78382)
  * Video post processing: low latency spatiotemporal approach for detection and removal of rain (IET IP2012), Tripathi et al [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6168449)
   * Removal of rain from videos: a review (SIVP2014), Tripathi et al [[PDF]](https://www.deepdyve.com/lp/springer-journals/removal-of-rain-from-videos-a-review-ZLDEb0QAU0)
   * Stereo video deraining and desnowing based on spatiotemporal frame warping (ICIP2014), Kim et al [[PDF]](https://ieeexplore.ieee.org/document/7026099/?arnumber=7026099)
* Frequency Domain 
  * Spatio-temporal frequency analysis for removing rain and snow from videos (PACV2007), Barnum et al [[Project]](http://www.cs.cmu.edu/~pbarnum/rain/rainAndSnow.html) [[PDF]](http://www.cs.cmu.edu/~pbarnum/rain/papers/barnum07spatio.pdf)
  * Analysis of rain and snow in frequency space (IJCV2010), Barnum et al [[Project]](http://www.cs.cmu.edu/~pbarnum/rain/rainAndSnow.html) [[PDF]](http://www.cs.cmu.edu/~pbarnum/rain/papers/barnum08analysis.pdf)
  
* Low Rank and Sparsity
  * A generalized low-rank appearance model for spatio-temporally correlated rain streaks (ICCV2013), Chen et al [[PDF]](http://openaccess.thecvf.com/content_iccv_2013/papers/Chen_A_Generalized_Low-Rank_2013_ICCV_paper.pdf) 
  * A rain pixel recovery algorithm for videos with highly dynamic scenes (TIP2013), Chen et al [[PDF]](http://www3.ntu.edu.sg/home/elpchau/pdf/Dynamic%20Scene%20Rain%20Removal.pdf) 
  * Video deraining and desnowing using temporal correlation and low-rank matrix completion (TIP2015), Kim et al [[PDF]](https://ieeexplore.ieee.org/abstract/document/7101234/) [[Code]](http://mcl.korea.ac.kr/deraining/)
  * Adherent raindrop modeling, detection and removal in video (TPAMI2016), You et al. [[Project]](http://users.cecs.anu.edu.au/~shaodi.you/CVPR2013/Shaodi_CVPR2013.html) [[PDF]](https://ieeexplore.ieee.org/document/7299675/)
  * Video desnowing and deraining based on matrix decomposition (CVPR2017), Ren et al [[PDF]](http://openaccess.thecvf.com/content_cvpr_2017/html/Ren_Video_Desnowing_and_CVPR_2017_paper.html) [[Code]](http://vision.sia.cn/our%20team/RenWeihong-homepage/vision-renweihong%28English%29.html)
  * A novel tensor-based video rain streaks removal approach via utilizing discriminatively intrinsic priors (CVPR2017), Jiang et al [[PDF]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Jiang_A_Novel_Tensor-Based_CVPR_2017_paper.pdf) 
  * Should We encode rain streaks in video as deterministic or stochastic? (ICCV2017), Wei et al [[PDF]](http://openaccess.thecvf.com/content_iccv_2017/html/Wei_Should_We_Encode_ICCV_2017_paper.html) [[Code]](https://github.com/wwzjer/RainRemoval_ICCV2017)
  * A directional global sparse model for single image rain removal (AMM2018), Deng et al [[PDF]](https://www.sciencedirect.com/science/article/pii/S0307904X18301069) [[Code]](http://www.escience.cn/people/dengliangjian/codes.html)
  * Video rain streak removal by multiscale convolutional sparse coding (CVPR2018), Li et al [[Project]](https://sites.google.com/view/cvpr-anonymity) [[PDF]](https://pan.baidu.com/s/1iiRr7ns8rD7sFmvRFcxcvw) [[Code]](https://github.com/MinghanLi/MS-CSC-Rain-Streak-Removal)
  * Fastderain: A novel video rain streak removal method using directional gradient priors (TIP2019), Jiang et al [[PDF]](https://ieeexplore.ieee.org/document/8531762/) [[Code]](https://github.com/TaiXiangJiang/FastDeRain)
  
* Deep Learning 
  * Robust video content alignment and compensation for rain removal in a cnn framework (CVPR2018), Chen et al [[PDF]](https://arxiv.org/abs/1803.10433) [[Code]](https://bitbucket.org/st_ntu_corplab/mrp2a/src/bd2633dbc9912b833de156c799fdeb82747c1240?at=master)
  *  Erase or fill? deep joint recurrent rain removal and reconstruction in videos (CVPR2018), Liu et al. [[Project]](http://www.icst.pku.edu.cn/struct/Projects/J4RNet.html)[[PDF]](http://39.96.165.147/Pub%20Files/2018/ywh_cvpr18.pdf) [[Code]](https://github.com/flyywh/J4RNet-Deep-Video-Deraining-CVPR-2018)
  * D3R-Net: dynamic routing residue recurrent network for video rain removal (TIP2018), Liu et al. [[PDF]](http://39.96.165.147/Pub%20Files/2019/ywh_tip19.pdf)
  * Frame consistent recurrent video deraining with dual-level flow (CVPR2019), Yang et al. [[Code]](https://github.com/flyywh/Dual-FLow-Video-Deraining-CVPR-2019)
  * Self-Learning Video Rain Streak Removal: When Cyclic Consistency Meets Temporal Correspondence(CVPR2020), Yang et al.[[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Self-Learning_Video_Rain_Streak_Removal_When_Cyclic_Consistency_Meets_Temporal_CVPR_2020_paper.pdf)[[Supplementray Materials]](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Yang_Self-Learning_Video_Rain_CVPR_2020_supplemental.pdf) [[Code]](https://github.com/flyywh/CVPR-2020-Self-Rain-Removal)
* **Reivew paper**
  * Removal of rain from videos: a review (SIVP2014), Tripathi et al [[PDF]](https://www.deepdyve.com/lp/springer-journals/removal-of-rain-from-videos-a-review-ZLDEb0QAU0)
  * A Survey on Rain Removal from Video and Single Image (Arxiv2019), Wang et al. [[PDF]](https://arxiv.org/abs/1909.08326) [[Code]](https://github.com/hongwang01/Video-and-Single-Image-Deraining)
  
## Single Image Deraining Methods
* Filter based methods
  * Guided image filtering (ECCV2010), He et al. [[Project]](http://kaiminghe.com/eccv10/index.html) [[PDF]](http://kaiminghe.com/publications/eccv10guidedfilter.pdf) [[Code]](http://kaiminghe.com/eccv10/guided-filter-code-v1.rar)
  * Removing rain and snow in a single image using guided filter (CSAE2012), Xu et al. [[PDF]](https://ieeexplore_ieee.gg363.site/abstract/document/6272780)
  * An improved guidance image based method to remove rain and snow in a single image (CIS2012), Xu et al. [[PDF]](https://pdfs.semanticscholar.org/6eac/36e3334dd0c9188b5a61af73909dcbfff39c.pdf)
  * Single-image deraining using an adaptive nonlocal means filter (ICIP2013), Kim et al. [[PDF]](https://ieeexplore_ieee.gg363.site/abstract/document/6738189)
  * Single-image-based rain and snow removal using multi-guided filter (NIPS2013), Zheng et al. [[PDF]](https://pdfs.semanticscholar.org/f111/54e4e1adbde9f24b25fd2d98337a759d8b21.pdf)
  * Single image rain and snow removal via guided L0 smoothing filter (Multimedia Tools and Application2016), Ding et al. [[PDF]](https://link_springer.gg363.site/article/10.1007/s11042-015-2657-7)
* Prior based methods
  * Automatic single-image-based rain streaks removal via image decomposition (TIP2012), Kang et al [[PDF]](http://www.ee.nthu.edu.tw/cwlin/Rain_Removal/tip_rain_removal_2011.pdf) [[Code]](http://www.ee.nthu.edu.tw/cwlin/pub/rain_tip2012_code.rar)
  * Self-learning-based rain streak removal for image/video (ISCS2012), Kang et al. [[PDF]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.701.3957&rep=rep1&type=pdf)
  * Single-frame-based rain removal via image decomposition (ICA2013), Fu et al. [[PDF]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.707.1053&rep=rep1&type=pdf)
  * Exploiting image structural similarity for single image rain removal (ICIP2014), Sun et al.  [[PDF]](http://mml.citi.sinica.edu.tw/papers/ICIP_2014_Sun.pdf)
  * Visual depth guided color image rain streaks removal using sparse coding (TCSVT2014), Chen et al [[PDF]](https://ieeexplore.ieee.org/document/6748866/)
  * Removing rain from a single image via discriminative sparse coding (ICCV2015), Luo et al [[PDF]](http://ieeexplore.ieee.org/document/7410745/) [[Code]](https://pan.baidu.com/s/1AztZ5BSNKWmxr9PzZwpGDw) pwd: d229
  * Rain streak removal using layer priors (CVPR2016), Li et al [[PDF]](https://ieeexplore.ieee.org/document/7780668/) [[Code]](http://yu-li.github.io/)
  * Single image rain streak decomposition using layer priors (TIP2017), Li et al [[PDF]](https://ieeexplore.ieee.org/document/7934436/)
  * Error-optimized dparse representation for single image rain removal (IEEE TIE2017), Chen et al [[PDF]](https://ieeexplore.ieee.org/abstract/document/7878618/)
  * A hierarchical approach for rain or snow removing in a single color image (TIP2017), Wang et al. [[PDF]](http://ieeexplore.ieee.org/abstract/document/7934435/)
  * Joint bi-layer optimization for single-image rain streak removal (ICCV2017), Zhu et al. [[PDF]](http://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Joint_Bi-Layer_Optimization_ICCV_2017_paper.html)
  * Convolutional sparse and low-rank codingbased rain streak removal (WCACV2017), Zhang et al [[PDF]](https://ieeexplore_ieee.gg363.site/abstract/document/7926728/)
  * Joint convolutional analysis and synthesis sparse representation for single image layer separation (CVPR2017), Gu et al [[PDF]](http://openaccess.thecvf.com/content_iccv_2017/html/Gu_Joint_Convolutional_Analysis_ICCV_2017_paper.html) [[Code]](https://sites.google.com/site/shuhanggu/home)
  * Single image deraining via decorrelating the rain streaks and background scene in gradient domain (PR2018)， Du et al [[PDF]](https://www.sciencedirect.com/science/article/pii/S0031320318300700)
* Deep Learning
  * Restoring an image taken through a window covered with dirt or rain (ICCV2013), Eigen et al. [[Project]](https://cs.nyu.edu/~deigen/rain/) [[PDF]](http://openaccess.thecvf.com/content_iccv_2013/papers/Eigen_Restoring_an_Image_2013_ICCV_paper.pdf) [[Code]](https://cs.nyu.edu/~deigen/rain/restore-dirt-rain.tgz)
  * Attentive generative adversarial network for raindrop removal from a single image (CVPR2018), Qian et al [[Project]](https://rui1996.github.io/raindrop/raindrop_removal.html) [[PDF]](https://arxiv.org/abs/1711.10098)
  * Clearing the skies: A deep network architecture for single-image rain streaks removal (TIP2017), Fu et al. [[Project]](https://xueyangfu.github.io/projects/tip2017.html) [[PDF]](https://ieeexplore.ieee.org/abstract/document/7893758/) [[Code]](https://xueyangfu.github.io/projects/tip2017.html)
  * Removing rain from single images via a deep detail network (CVPR2017), Fu et al. [[Project]](https://xueyangfu.github.io/projects/cvpr2017.html) [[PDF]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Removing_Rain_From_CVPR_2017_paper.pdf) [[Code]](https://xueyangfu.github.io/projects/cvpr2017.html)
  * Image de-raining using a conditional generative adversarial network (Arxiv2017), Zhang et al [[PDF]](https://arxiv.org/abs/1701.05957) [[Code]](https://github.com/hezhangsprinter/ID-CGAN)
  * Deep joint rain detection and removal from a single image (CVPR2017), Yang et al.[[Project]](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html) [[PDF]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf) [[Code]](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)
  * Residual guide feature fusion network for single image deraining (ACMMM2018), Fan et al. [[Project]](https://zhiwenfan.github.io/) [[PDF]](http://export.arxiv.org/pdf/1804.07493)
  * Fast single image rain removal via a deep decomposition-composition network (Arxiv2018), Li et al [[Project]](https://sites.google.com/view/xjguo/rain)) [[PDF]](https://arxiv.org/abs/1804.02688) [[Code]](https://drive.google.com/open?id=1TPu9RX7Q9dAAn5M1ECNbqtRDa9c6_WOt)
  * Density-aware single image de-raining using a multi-stream dense network (CVPR2018), Zhang et al [[PDF]](https://arxiv.org/abs/1802.07412) [[Code]](https://github.com/hezhangsprinter/DID-MDN)
  * Recurrent squeeze-and-excitation context aggregation net for single image deraining (ECCV2018), Li et al. [[PDF]](https://export.arxiv.org/pdf/1807.05698) [[Code]](https://github.com/XiaLiPKU/RESCAN)
  * Rain streak removal for single image via kernel guided cnn (Arxiv2018), Wang et al [[PDF]](https://arxiv.org/pdf/1808.08545.pdf)
  * Physics-based generative adversarial models for image restoration and beyond (Arxiv2018), Pan et al [[PDF]](https://arxiv.org/pdf/1808.00605.pdf)
  * Learning dual convolutional neural networks for low-level vision (CVPR2018), Pan et al [[Project]](https://sites.google.com/site/jspanhomepage/dualcnn) [[PDF]](https://arxiv.org/pdf/1805.05020.pdf) [[Code]](https://sites.google.com/site/jspanhomepage/dualcnn)
  * Non-locally enhanced encoder-decoder network for single image de-raining (ACMMM2018), Li et al [[PDF]](https://arxiv.org/pdf/1808.01491.pdf) [[Code]](https://github.com/AlexHex7/NLEDN)
  * Single image rain removal via a deep decomposition-composition network (CVIU2019), Li et al.
  * Unsupervised single image deraining with self-supervised constraints (ICIP2019), Jin et al [[PDF]](https://arxiv.org/pdf/1811.08575)
  * Residual multiscale based single image deraining (BMVC2019), Zheng et al.
  * Erl-net: Entangled representation learning for single image de-raining (ICCV2019), Wang et al. [[code]](https://github.com/RobinCSIRO/ERL-Net-for-Single-Image-Deraining)
  * Uncertainty guided multi-scale residual learning-using a cycle spinning cnn for single image de-raining (CVPR2019),  Rajeev Yasarla et al.[[Code]](https://github.com/rajeevyasarla/UMRL--using-Cycle-Spinning)
  * Heavy rain image restoration: Integrating physics model and conditional adversarial learning (CVPR2019), Li et al.[[Code]](https://github.com/liruoteng/HeavyRainRemoval)
  * Progressive image deraining networks: A better and simpler baseline (CVPR2019), Ren et al [[PDF]](https://csdwren.github.io/papers/PReNet_cvpr_camera.pdf) [[Code]](https://github.com/csdwren/PReNet)
  * Spatial attentive single-image deraining with a high quality real rain dataset (CVPR2019), Wang et al [[Project]](https://stevewongv.github.io/derain-project.html) [[PDF]](https://arxiv.org/abs/1904.01538) [[Code]](https://github.com/stevewongv/SPANet)
  * Lightweight pyramid networks for image deraining (TNNLS2019), Fu et al [[PDF]](https://arxiv.org/pdf/1805.06173.pdf) [[Code]](https://xueyangfu.github.io/projects/LPNet.html)
  * Joint rain detection and removal from a single image with contextualized deep networks (TPAMI2019), Yang et al [[PDF]](https://ieeexplore.ieee.org/document/8627954) [[Code]](https://github.com/flyywh/JORDER-E-Deep-Image-Deraining-TPAMI-2019-Journal)
  * Scale-free single image deraining via visibility-enhanced recurrent wavelet learning (TIP2019), Yang et al.[[PDF]](https://xueshu.baidu.com/usercenter/paper/show?paperid=1s5y0tx0xn340as0916d02a0vr034675&site=xueshu_se)
  * Towards scale-free rain streak removal via selfsupervised fractal band learning (AAAI2020), Yang et al.[[Code]](https://github.com/flyywh/AAAI-2020-FBL-SS)
  * Structural Residual Learning for Single Image Rain Removal(Arxiv2020), Wang et al. [[PDF]](https://arxiv.org/abs/2005.09228)
  * All in One Bad Weather Removal Using Architectural Search (CVPR2020), Li et al.[[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_All_in_One_Bad_Weather_Removal_Using_Architectural_Search_CVPR_2020_paper.pdf)
  * Syn2Real Transfer Learning for Image Deraining Using Gaussian Processes(CVPR2020), Rajeev Yasarla et al. [[Code]](https://github.com/rajeevyasarla/)
  * Multi-Scale Progressive Fusion Network for Single Image Deraining(CVPR2020), Jiang et al. [[Code]](https://github.com/kuihua/MSPFN)
  * Detail-recovery Image Deraining via Context Aggregation Networks(CVPR2020), Deng et al.[[Code]](https://github.com/Dengsgithub/DRD-Net)
  * Variational image deraining(WACV2020), Du et al.[[PDF]](https://openaccess.thecvf.com/content_WACV_2020/papers/Du_Variational_Image_Deraining_WACV_2020_paper.pdf)
  * Learning Multiple Adverse Weather Removal via Two-stage Knowledge Learning and Multi-contrastive Regularization: Toward a Unified Model (CVPR2022), Chen et al.[[Project]](https://github.com/fingerk28/Two-stage-Knowledge-For-Multiple-Adverse-Weather-Removal) [[PDF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Learning_Multiple_Adverse_Weather_Removal_via_Two-Stage_Knowledge_Learning_and_CVPR_2022_paper.pdf) 
  * GIQE: Generic Image Quality Enhancement via Nth Order Iterative Degradation (CVPR2022), Pranjay Shyam et al. [[PDF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Shyam_GIQE_Generic_Image_Quality_Enhancement_via_Nth_Order_Iterative_Degradation_CVPR_2022_paper.pdf) [[Supplementary Materials]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Shyam_GIQE_Generic_Image_CVPR_2022_supplemental.pdf)
  * Dual Heterogeneous Complementary Networks for Single Image Deraining (CVPR2022), Yuuto Nanba et al. [[PDF]](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Nanba_Dual_Heterogeneous_Complementary_Networks_for_Single_Image_Deraining_CVPRW_2022_paper.pdf)
  * TransWeather: Transformer-based Restoration of Images Degraded by Adverse Weather Conditions (CVPR2022), Jeya Maria Jose Valanarasu et al. [[Project]](https://github.com/vfrantc/TransMod) [[PDF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Valanarasu_TransWeather_Transformer-Based_Restoration_of_Images_Degraded_by_Adverse_Weather_Conditions_CVPR_2022_paper.pdf)
  * Unpaired Deep Image Deraining Using Dual Contrastive Learning (CVPR2022), Chen et al. [[Project]](https://cxtalk.github.io/projects/DCD-GAN.html) [[PDF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Unpaired_Deep_Image_Deraining_Using_Dual_Contrastive_Learning_CVPR_2022_paper.pdf)
  * Unsupervised Deraining: Where Contrastive Learning Meets Self-similarity (CVPR2022), Ye et al. [[PDF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Ye_Unsupervised_Deraining_Where_Contrastive_Learning_Meets_Self-Similarity_CVPR_2022_paper.pdf)
  * Single Image Deraining Network with Rain Embedding Consistency and Layered LSTM (WACV2022), Li et al. [[PDF]](https://openaccess.thecvf.com/content/WACV2022/papers/Li_Single_Image_Deraining_Network_With_Rain_Embedding_Consistency_and_Layered_WACV_2022_paper.pdf)
  * FLUID: Few-Shot Self-Supervised Image Deraining (WACV2022), Shyam Nandan Rai et al. [[PDF]](https://openaccess.thecvf.com/content/WACV2022/papers/Nandan_FLUID_Few-Shot_Self-Supervised_Image_Deraining_WACV_2022_paper.pdf) [[Code]](https://github.com/biubiubiiu/derain-toolbox/tree/efc7b2f00e027f6b640317273a960c7388488129/configs/ecnet)
  * Single image rain removal using recurrent scale-guide networks (Neurocomputing2022), Wang et al. [[PDF]](https://www.sciencedirect.com/science/article/pii/S0925231221015071)
  * Non-local channel aggregation network for single image rain removal (Neurocomputing2022), Su et al. [[PDF]](https://pdf.sciencedirectassets.com/271597/1-s2.0-S0925231221X00392/1-s2.0-S0925231221015381/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJL%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIHpOkik5nfskcG65HE6FqbqgbwkZ8twfDW%2Fm%2BEs2febWAiEAurZ8S866NGW9PR%2Fc%2FFlUqfNxawXxdW6Z543MMkrcEBsq2wQIm%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAEGgwwNTkwMDM1NDY4NjUiDNbDUd46tEPRVAX6CyqvBN%2FbFhiqRJvZsLo4%2FBpmQdRhuh%2B7HtLjNr1iHNmhGbZTJIhe1b3JTn%2F9a0iluKh9LSLlTScczGJ202q8id6HrsIMq82Wecwbe%2F7PBA%2FgYGmWLutDble5wRxU%2BuiXyCE6L9B3BB3yJMy7zGz9DpDqSWPpoIoXStbQz0tb%2BAk04g%2FkSLOEOCj2qNq5cYCb1BKXyWNO4kkmIqLZDQFk%2FgwJZiWeQJi9Gg3Ebw4iOP5A1DYTfULMcmD9D8PcG%2FP6Q5Ju31j51tm9DfAqTvYx09dZEoJHlwnAlAObVd88hcQHzdA3jFciiTgRPQTPhr2Mw6RAhGHE7hdRsY8532outSJO87ohjMhrf7XRCnyuTEmbd2gfl%2FFCk9KfycqZ6fYo4yPNzdBa3mA0OkeRHU6vyjbTwI5XdX%2Bd%2FIXTSIX2i6eekCvQ7%2FuSPiXDQ8Vi%2F%2BxWUWfq9kz%2FqhbK5fOf262A1nQvEnqpw654D2wmFw8CzvCGvRCtyAF4GQi7p%2BpjGVAJJHQ0lZIsmc1DLNm1un3bu9JuBGpaSiuFugRqQa0cKjuqMMHLlAaZdTleQJxKktl6gL23y9gNcGNVIsAuLMEjMaajMQ%2B3Emp6LwAp44iD1n5uZK%2FofDlTUT%2FgHfdkqQlvRdAG9kDW3SIEW7zpy2jXv424%2FQG9Ws%2Foe9PYKSyx63GYSB99WTsQLbNPRnFpu3lTl2OaMiiHl2uM%2B75VPMmjKD0vjyX1W7Y7PhxN79ph8vNGdDwwmuuklQY6qQG8qNOsTboUz0gAVbA4OusBCV6IibOz0idkjBvudCdFJbFZr%2F4e7hMEdAm3LHGv%2BzNAjhK1IBRntIlt5x1qNR0JGmr5xSzzwF6DowmGEBHtVkgkiErpavRlftpOvVdi06cTkLExzgX6D2wFUfp8Nucp%2Fe4Ip3%2FKts8jmES6P7VkQ0JrLeo4ldlaul%2FYqOlgyTn5%2FURFy3zJLvBEjay5%2FFhq8tMGrgE%2F6SNe&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20220615T022439Z&X-Amz-SignedHeaders=host&X-Amz-Expires=299&X-Amz-Credential=ASIAQ3PHCVTYXQXMRR5U%2F20220615%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=763e61d3f99c3b21a24dcb8ad39e68f8f79ce53a6c4c26c4979a2da308915b3a&hash=c650a0ae313e1fee62e0884e4bc0d83d1cd8af0f8bd67adb655012b4c6fb7363&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0925231221015381&tid=spdf-22b22478-29f2-45f5-98d3-1ddc0ee40350&sid=7801520e69f8b34bba09bfb986cc1e7d45c7gxrqb&type=client&ua=4d530051015b0254020c&rr=71b7d9c6cc852095)
* Joint Model-driven and Data-driven
  * Deep Layer Prior Optimization for Single Image Rain Streaks Removal (ICASSP2018), Liu et al [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8461892)
  * Learning bilevel layer priors for single image rain streaks removal (SP Letters 2018), Mu et al [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8586910)
  * Semi-supervised transfer learning for image rain removal (CVPR2019), Wei et al [[PDF]](http://gr.xjtu.edu.cn/c/document_library/get_file?folderId=2618027&name=DLFE-118007.pdf) [[Code]](https://github.com/wwzjer/Semi-supervised-IRR)
  * Knowledge-driven deep unrolling for robust image layer separation (TNNLS2019), Liu et al.
  * A Model-driven Deep Neural Network for Single Image Rain Removal (CVPR2020), Wang et al [[PDF]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_A_Model-Driven_Deep_Neural_Network_for_Single_Image_Rain_Removal_CVPR_2020_paper.pdf)[[Supplementary Materials](http://openaccess.thecvf.com/content_CVPR_2020/supplemental/Wang_A_Model-Driven_Deep_CVPR_2020_supplemental.pdf) [[Code]](https://github.com/hongwang01/RCDNet)
  * Towards Robust Rain Removal Against Adversarial Attacks: A Comprehensive Benchmark Analysis and Beyond (CVPR2022), Yu et al [[Project]]([yuyi-sd/Robust_Rain_Removal (github.com)](https://github.com/yuyi-sd/Robust_Rain_Removal)) [[PDF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_Towards_Robust_Rain_Removal_Against_Adversarial_Attacks_A_Comprehensive_Benchmark_CVPR_2022_paper.pdf) 
* **Reivew paper**
  * Single image deraining: A comprehensive benchmark analysis(CVPR2019), Li et al.[[PDF]](https://arxiv.org/abs/1903.08558) [[Code]](https://github.com/lsy17096535/Single-Image-Deraining)
  * A Survey on Rain Removal from Video and Single Image (Arxiv2019), Wang et al. [[PDF]](https://arxiv.org/abs/1909.08326) [[Code]](https://github.com/hongwang01/Video-and-Single-Image-Deraining)
  * Single image deraining: From model-based to data-driven and beyond(TPAMI2020), Yang et al.[[Code]](https://flyywh.github.io/Single_rain_removal_survey)

## Datasets and Discriptions 
* Video
  * Synthetic Datasets: **highway** and **park**. 
  * Real Datasets: **compfinal** and **night**. 
    Please download from [[Baidu Netdisk]](https://pan.baidu.com/s/1N5I13BCqBs4iQT7Llw_KWg) provided by [Li Minghan](https://github.com/MinghanLi/MS-CSC-Rain-Streak-Removal).
  
* Single Image
  * Synthetic Datasets:  **RainTrainL/Rain100L**, **RainTrainH/Rain100H**, **Rain12600/Rain1400**, and **Rain12**.
     Please download from [[Baidu Netdisk]](https://pan.baidu.com/s/1J0q6Mrno9aMCsaWZUtmbkg) provided by [Ren Dongwei](https://github.com/csdwren/PReNet).
  * Real Datasets: Please download **SPA-Data** from [[Baidu Netdisk, key: 4fwo]](https://pan.baidu.com/s/1lPn3MWckHxh1uBYYucoWVQ) provided by [Wang Tianyu](https://github.com/stevewongv/SPANet) and  **Internet-Data** from the link provided by [Weiwei](https://github.com/wwzjer/Semi-supervised-IRR/tree/master/data/rainy_image_dataset/real_input) .
  
  * Other important datasets:
     Rain800 [[Link]](https://github.com/hezhangsprinter/ID-CGAN)，
     Rain12000[[Link]](https://github.com/hezhangsprinter/)，
     RainCityscapes[[Link]](https://github.com/xw-hu/DAF-Net)，
     NYU-Rain[[Link]](https://github.com/liruoteng/HeavyRainRemoval)，
     MPID [[Link]](https://github.com/lsy17096535/Single-Image-Deraining)
  

**We note that*:

*i. **RainTrainL/Rain100L** and **RainTrainH/Rain100H** are synthesized by [Yang Wenhan](https://github.com/flyywh). **Rain12600/Rain1400** is from [Fu Xueyang](https://xueyangfu.github.io/) and **Rain12** is from [Li Yu](http://yu-li.github.io/).*

*ii. In video experiment, the rain-removed results of [the deep learning method]((http://39.96.165.147/Pub%20Files/2018/ywh_cvpr18.pdf)) are provided by the author [Yang Wenhan](https://github.com/flyywh). Really thanks!*

*iii. In single image experiment, we seperately retrain all the recent state-of-the-art methods via the three training datasets: **RainTrainL**(200 input/clean image pairs), **RainTrainH**(1800 pairs), and **Rain12600**(12600 pairs), and then evaluate their rain removal performance based on the correponding test datasets: **Rain100L**(100 pairs), **Rain100H**(100 pairs), and **Rain1400**(1400 pairs). Besides, the trained model obtained by **RainTrainL** is adpoted to predict rain-removed results of **Rain12**(12 pairs). Moreover, we utilize the **Internet-Data**(147 input images) and **SPA-Data**(1000 pairs) to compare the generalization ability.* 

*iiii. In single image experiment,  when training the semi-supervised method--SIRR, we always utilize **Internet-Data** as unsupervised samples*.

##  Image Quality Metrics
* Reference 
   * PSNR (Peak Signal-to-Noise Ratio) [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4550695) [[Matlab Code]](https://www.mathworks.com/help/images/ref/psnr.html) [[Python Code]](https://github.com/aizvorski/video-quality)
   * SSIM (Structural Similarity) [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1284395) [[Matlab Code]](http://www.cns.nyu.edu/~lcv/ssim/ssim_index.m) [[Python Code]](https://github.com/aizvorski/video-quality/blob/master/ssim.py)
   * VIF (Visual Quality) [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1576816) [[Matlab Code]](http://sse.tongji.edu.cn/linzhang/IQA/Evalution_VIF/eva-VIF.htm)
   * FSIM (Feature Similarity) [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5705575) [[Matlab Code]](http://sse.tongji.edu.cn/linzhang/IQA/FSIM/FSIM.htm))
* Non-reference
  * NIQE (Naturalness Image Quality Evaluator)[[PDF]](https://live.ece.utexas.edu/research/Quality/niqe_spl.pdf) [[Matlab Code]](http://live.ece.utexas.edu/research/Quality/niqe_release.zip)
  * BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)[[PDF]](https://www.live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf) [[Matlab Code]](http://live.ece.utexas.edu/research/Quality/BRISQUE_release.zip)
  * SSEQ (Spatial-Spectral Entropy-based Quality)[[PDF]](https://www.sciencedirect.com/science/article/abs/pii/S0923596514000927) [[Matlab Code]](http://live.ece.utexas.edu/research/Quality/SSEQ_release.zip)
  

**Please note that all quantitative results in our survey paper are computed based on Y channel*.

## Contact
If you have any question, please feel free to concat Hong Wang (Email: hongwang01@stu.xjtu.edu.cn).
