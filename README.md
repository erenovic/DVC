# DVC: An End-to-end Deep Video Compression Framework
An unofficial implementation of "DVC: An End-to-end Deep Video Compression Framework"

### The model is a reimplementation of architecture designed by Lu et al. (2019) 
For further details about the model and training, please refer to the the official project page:
> Guo Lu, Wanli Ouyang, Dong Xu, Xiaoyun Zhang, Chunlei Cai, and Zhiyong Gao, "DVC: An End-to-end Deep Video Compression Framework", Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019. [[Paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lu_DVC_An_End-To-End_Deep_Video_Compression_Framework_CVPR_2019_paper.pdf)
```
@inproceedings{lu2019dvc,
  title={Dvc: An end-to-end deep video compression framework},
  author={Lu, Guo and Ouyang, Wanli and Xu, Dong and Zhang, Xiaoyun and Cai, Chunlei and Gao, Zhiyong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11006--11015},
  year={2019}
}
```

The training and the re-implementation has to be followed according to the specifications in the paper.

For the key frame compression, the learned image compression model by Balle et al. (2018) is used. The implementation is taken from the ```compressai``` library:

```
@inproceedings{minnenbt18,
  author    = {David Minnen and
               Johannes Ball{\'{e}} and
               George Toderici},
  editor    = {Samy Bengio and
               Hanna M. Wallach and
               Hugo Larochelle and
               Kristen Grauman and
               Nicol{\`{o}} Cesa{-}Bianchi and
               Roman Garnett},
  title     = {Joint Autoregressive and Hierarchical Priors for Learned Image Compression},
  booktitle = {Advances in Neural Information Processing Systems 31: Annual Conference
               on Neural Information Processing Systems 2018, NeurIPS 2018, 3-8 December
               2018, Montr{\'{e}}al, Canada},
  pages     = {10794--10803},
  year      = {2018},
}
```
```
@article{begaint2020compressai,
	title={CompressAI: a PyTorch library and evaluation platform for end-to-end compression research},
	author={B{\'e}gaint, Jean and Racap{\'e}, Fabien and Feltman, Simon and Pushparaja, Akshay},
	year={2020},
	journal={arXiv preprint arXiv:2011.03029},
}
```
