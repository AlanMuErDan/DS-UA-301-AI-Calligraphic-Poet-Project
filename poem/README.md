# Poetry Generation
The Poem generator inputs image captioning and generates related poetry


The Implementation of the poem generator is heavily influenced and inspired by [SongNet](https://github.com/lipiji/SongNet).

## Code
Much Thanks to
### 
```bibtex
@inproceedings{li-etal-2020-rigid,
    title     = {Rigid Formats Controlled Text Generation},
    author    = {Li, Piji and Zhang, Haisong and Liu, Xiaojiang and Shi, Shuming},
    booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
    month     = jul,
    year      = {2020},
    address   = {Online},
    publisher = {Association for Computational Linguistics},
    url       = {https://www.aclweb.org/anthology/2020.acl-main.68},
    doi       = {10.18653/v1/2020.acl-main.68},
    pages     = {742--751}
}
```
See their github repo for more: https://github.com/lipiji/SongNet

## Data 

Datasets used for fine-tuning
1. [THUAIPoet Datasets](https://github.com/THUNLP-AIPoet/Datasets?tab=readme-ov-file)
2. [Ancient Chinese Poetry Collection Across Dynasties](https://github.com/Werneror/Poetry)

## Training and Testing
## Training
* ./train.sh
## Testing
* ./test.sh
