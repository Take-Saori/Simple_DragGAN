import os
from pathlib import Path

sample_image_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sample_image')

def generate_sample_images(valid_checkpoints_dict):

    # Create folder to store sample image for each checkpoints
    Path(sample_image_dir).mkdir(parents=True, exist_ok=True)
    

    sample_image_list = [x for x in os.listdir(sample_image_dir) if '_sample_image.png' in x]

    available_ckpt = list(valid_checkpoints_dict.keys())

    for ckpt in available_ckpt:
        ckpt_sample_image_name = ckpt + '_sample_image.png'
        if not (ckpt_sample_image_name in sample_image_list):
            print("generate command:")
            print(f'python generate.py --outdir=\"{sample_image_dir}\" --seeds=1 --network=\"{valid_checkpoints_dict[ckpt]}\" --rename=\"{ckpt_sample_image_name}\" --trunc=0.7')
            os.system(f'python generate.py --outdir=\"{sample_image_dir}\" --seeds=1 --network=\"{valid_checkpoints_dict[ckpt]}\" --rename=\"{ckpt_sample_image_name}\" --trunc=0.7')
            
            # generate_images(network_pkl=valid_checkpoints_dict[ckpt],
            #                 seeds=[0],
            #                 outdir=sample_image_dir,
            #                 rename=ckpt_sample_image_name,
            #                 truncation_psi=1, 
            #                 class_idx=None,
            #                 noise_mode='const',
            #                 translate='0.0',
            #                 rotate=0
            #                 )
        
def get_sample_image_path(checkpoint):
    return os.path.join(sample_image_dir, checkpoint+"_sample_image.png")
    

