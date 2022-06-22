# Introduction

**MDA GAN:**
The interpolation and reconstruction of missing traces is a crucial step in seismic data processing, moreover it is also a highly ill-posed problem, especially for complex cases such as high-ratio random discrete missing, continuous missing and missing in fault-rich or salt body surveys. These complex cases are rarely mentioned in current works. To cope with complex missing cases, we propose Multi-Dimensional Adversarial GAN (MDA GAN), a novel 3-D GAN framework. It employs three discriminators to ensure the consistency of the reconstructed data with the original data distribution in each dimension. The feature splicing module (FSM) is designed and embedded into the generator of this framework, which automatically splices the features of the unmissing part with those of the reconstructed part (missing part), thus fully preserving the information of the unmissing part. To prevent pixel distortion in the seismic data caused by the adversarial learning process, we propose a new reconstruction loss Tanh Cross Entropy (TCE) loss to provide smoother gradients. We experimentally verified the effectiveness of the individual components of the study and then tested the method on multiple publicly available data. The method achieves reasonable reconstructions for up to 95% of random discrete missing, 100 traces of continuous missing and a mixture of both types with a cumulative 99.4% missing. In fault and salt body enriched surveys, MDA GAN still yields promising results for complex cases. Experimentally it has been demonstrated that our method achieves better performance than other methods in both simple and complex cases.
# Quick start
pytorch>=1.10.0

Get test data: https://drive.google.com/drive/folders/11oZC1uwdpCui1tVbAgjwMlNQYSLpozmW?usp=sharing
    
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    cp ./download/Parihaka_part.npy  ./MDA_GAN/data/
    cp ./download/kerry_part.npy ./MDA_GAN/data/
    cp ./download/MDA_GAN.pt ./MDA_GAN/network


# Results
<div align=center><img src="https://github.com/douyimin/MDA_GAN/blob/main/results/output.jpg" width="850" height="550" alt="Results"/><br/></div>

# Cite us
   
     Dou, Yimin, et al. 
     "MDA GAN: Adversarial-Learning-based 3-D Seismic Data Interpolation and Reconstruction for Complex Missing."
     arXiv preprint arXiv:2204.03197 (2022).

# Contact us
emindou3015@gmail.com
