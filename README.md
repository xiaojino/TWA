# TWA
[From Token to Word: OCR Token Evolution via Contrastive Learning and Semantic Matching for Text-VQA](https://dl.acm.org/doi/abs/10.1145/3503161.3547977)

ACM International Conference on Multimedia (ACM MM), 2022

This repository is based on and inspired by @microsoft's [work](https://github.com/microsoft/TAP) . We sincerely thank for their sharing of the codes.

### Introduction
We propose a novel Text-VQA method with multi-modal OCR Token-Word Contrastive (TWC) learning.
For more details, please refer to our
[paper](https://dl.acm.org/doi/abs/10.1145/3503161.3547977).


### Citation

    @inproceedings{jin2022token,
      title={From Token to Word: OCR Token Evolution via Contrastive Learning and Semantic Matching for Text-VQA},
      author={Jin, Zan-Xia and Shou, Mike Zheng and Zhou, Fang and Tsutsui, Satoshi and Qin, Jingyan and Yin, Xu-Cheng},
      booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
      pages={4564--4572},
      year={2022}
    }

### Prerequisites
* Python 3.6
* Pytorch 1.4.0
* Please refer to ``requirements.txt``. Or using

  ```
  python setup.py develop
  ```

## Installation

1. Clone the repository

    ```
    git clone https://github.com/xiaojino/TWA.git
    cd TWA
    python setup.py develop
    ```

2. Data

* Please refer to the Readme in the ``data`` folder.

### Training
3. Train the model, run the code under main folder. 
Using flag ``--pretrain`` to access the pre-training mode, otherwise the main QA losses are used to optimize the model. Example yml files are in ``configs`` folder. Detailed configs are in [released models](https://drive.google.com/drive/folders/1huY8HtwuIgEv4wbzoiV92ZUeb4Gw6XCZ?usp=share_link).

    Pre-training:
    ```
    python -m torch.distributed.launch --nproc_per_node $num_gpu tools/run.py --pretrain --tasks vqa --datasets $dataset --model $model --seed $seed --config configs/vqa/$dataset/"$pretrain_yml".yml --save_dir save/$pretrain_savedir training_parameters.distributed True

    # for example
    python -m torch.distributed.launch --nproc_per_node 4 tools/run.py --pretrain --tasks vqa --datasets m4c_textvqa --model twa --seed 13 --config configs/vqa/m4c_textvqa/twa_pretrain.yml --save_dir save/twa_pretrain_test training_parameters.distributed True
    ```

    Fine-tuning:
    ```
    python -m torch.distributed.launch --nproc_per_node $num_gpu tools/run.py --tasks vqa --datasets $dataset --model $model --seed $seed --config configs/vqa/$dataset/"$refine_yml".yml --save_dir save/$refine_savedir --resume_file save/$pretrain_savedir/$savename/pretrain_best.ckpt training_parameters.distributed True

    # for example
    python -m torch.distributed.launch --nproc_per_node 4 tools/run.py --tasks vqa --datasets m4c_textvqa --model twa --seed 13 --config configs/vqa/m4c_textvqa/twa_refine.yml --save_dir save/twa_refine_test --resume_file save/pretrained/textvqa/twa_pretrain_best.ckpt training_parameters.distributed True
    ```

4. Evaluate the model, run the code under main folder. 
Set up val or test set by ``--run_type``.

    ```
    python -m torch.distributed.launch --nproc_per_node $num_gpu tools/run.py --tasks vqa --datasets $dataset --model $model --config configs/vqa/$dataset/"$refine_yml".yml --save_dir save/$refine_savedir --run_type val --resume_file save/$refine_savedir/$savename/best.ckpt training_parameters.distributed True

    # for val evaluation
    python -m torch.distributed.launch --nproc_per_node 4 tools/run.py --tasks vqa --datasets m4c_textvqa --model twa --config configs/vqa/m4c_textvqa/twa_refine.yml --save_dir save/twa_refine_test --run_type val --resume_file save/finetuned/textvqa/twa_best.ckpt training_parameters.distributed True
   
    # for test inference 
    python  tools/run.py --tasks vqa --datasets m4c_textvqa --model twa --config configs/vqa/m4c_textvqa/twa_refine.yml --save_dir save/twa_refine_test --run_type inference --evalai_inference 1 --resume_file save/finetuned/textvqa/twa_best.ckpt
    ```


## Pre-trained Models
Please check the detailed experiment settings in our [paper](https://dl.acm.org/doi/abs/10.1145/3503161.3547977). 

[Model checkpoints](https://drive.google.com/drive/folders/1huY8HtwuIgEv4wbzoiV92ZUeb4Gw6XCZ?usp=share_link). 

### Credits
The project is built based on the following repository:
* [TAP: Text-Aware Pre-training for Text-VQA and Text-Caption](https://github.com/microsoft/TAP/).
* [MMF: A multimodal framework for vision and language research](https://github.com/facebookresearch/mmf/).