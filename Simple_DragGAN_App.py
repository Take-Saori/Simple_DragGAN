import os
import os.path as osp
from argparse import ArgumentParser
from functools import partial

import gradio as gr
import numpy as np
import torch
from PIL import Image

import dnnlib
from gradio_utils import (ImageMask, draw_mask_on_image, draw_points_on_image,
                          get_latest_points_pair, get_valid_mask,
                          on_change_single_global_state)
from viz.renderer import Renderer, add_watermark_np

import json
import random
import projector

from sample_image import generate_sample_images, get_sample_image_path

parser = ArgumentParser()
parser.add_argument('--share', action='store_true',default='True')
parser.add_argument('--cache-dir', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'checkpoints'))
parser.add_argument(
    "--listen",
    action="store_true",
    help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests",
)
args = parser.parse_args()

cache_dir = args.cache_dir

device = 'cuda'


def reverse_point_pairs(points):
    new_points = []
    for p in points:
        new_points.append([p[1], p[0]])
    return new_points


def clear_state(global_state, target=None):
    """Clear target history state from global_state
    If target is not defined, points and mask will be both removed.
    1. set global_state['points'] as empty dict
    2. set global_state['mask'] as full-one mask.
    """
    if target is None:
        target = ['point', 'mask']
    if not isinstance(target, list):
        target = [target]
    if 'point' in target:
        global_state['points'] = dict()
        print('Clear Points State!')
    if 'mask' in target:
        image_raw = global_state["images"]["image_raw"]
        global_state['mask'] = np.ones((image_raw.size[1], image_raw.size[0]),
                                       dtype=np.uint8)
        print('Clear mask State!')

    return global_state


def init_images(global_state):
    """This function is called only ones with Gradio App is started.
    0. pre-process global_state, unpack value from global_state of need
    1. Re-init renderer
    2. run `renderer._render_drag_impl` with `is_drag=False` to generate
       new image
    3. Assign images to global state and re-generate mask
    """

    if isinstance(global_state, gr.State):
        state = global_state.value
    else:
        state = global_state

    state['renderer'].init_network(
        state['generator_params'],  # res
        valid_checkpoints_dict[state['pretrained_weight']],  # pkl
        state['params']['seed'],  # w0_seed,
        state['w_load'],  # w_load
        state['params']['latent_space'] == 'w+',  # w_plus
        'const',
        state['params']['trunc_psi'],  # trunc_psi,
        state['params']['trunc_cutoff'],  # trunc_cutoff,
        None,  # input_transform
        state['params']['lr']  # lr,
    )

    state['renderer']._render_drag_impl(state['generator_params'],
                                        is_drag=False,
                                        to_pil=True)

    init_image = state['generator_params'].image
    state['images']['image_orig'] = init_image
    state['images']['image_raw'] = init_image
    state['images']['image_show'] = Image.fromarray(
        add_watermark_np(np.array(init_image)))
    state['mask'] = np.ones((init_image.size[1], init_image.size[0]),
                            dtype=np.uint8)
    return global_state


def update_image_draw(image, points, mask, show_mask, global_state=None):

    image_draw = draw_points_on_image(image, points)
    if show_mask and mask is not None and not (mask == 0).all() and not (
            mask == 1).all():
        image_draw = draw_mask_on_image(image_draw, mask)

    image_draw = Image.fromarray(add_watermark_np(np.array(image_draw)))
    if global_state is not None:
        global_state['images']['image_show'] = image_draw
    return image_draw


def preprocess_mask_info(global_state, image):
    """Function to handle mask information.
    1. last_mask is None: Do not need to change mask, return mask
    2. last_mask is not None:
        2.1 global_state is remove_mask:
        2.2 global_state is add_mask:
    """
    if isinstance(image, dict):
        last_mask = get_valid_mask(image['mask'])
    else:
        last_mask = None
    mask = global_state['mask']

    # mask in global state is a placeholder with all 1.
    if (mask == 1).all():
        mask = last_mask

    # last_mask = global_state['last_mask']
    editing_mode = global_state['editing_state']

    if last_mask is None:
        return global_state

    if editing_mode == 'remove_mask':
        updated_mask = np.clip(mask - last_mask, 0, 1)
        print(f'Last editing_state is {editing_mode}, do remove.')
    elif editing_mode == 'add_mask':
        updated_mask = np.clip(mask + last_mask, 0, 1)
        print(f'Last editing_state is {editing_mode}, do add.')
    else:
        updated_mask = mask
        print(f'Last editing_state is {editing_mode}, '
              'do nothing to mask.')

    global_state['mask'] = updated_mask
    # global_state['last_mask'] = None  # clear buffer
    return global_state


valid_checkpoints_dict = {
    f.split('/')[-1].split('.')[0]: osp.join(cache_dir, f)
    for f in os.listdir(cache_dir)
    if (f.endswith('pkl') and osp.exists(osp.join(cache_dir, f)))
}
print(f'File under cache_dir ({cache_dir}):')
print(os.listdir(cache_dir))
print('Valid checkpoint file:')
print(valid_checkpoints_dict)

# Generate sample images for each checkpoints if not already available
generate_sample_images(valid_checkpoints_dict)

# Get object name and type available to generate from the downloaded models
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_pickle_info.json')) as json_file:
# with open("model_pickle_info.json") as json_file:
    model_info_dict = dict(json.load(json_file))

# Get available object name to generate, with type (Body only, Face only, etc) in brackets
object_model_dict = {}
for key, value in model_info_dict.items():
    for pickle in value:
        object_model_dict[pickle["display_name"]] = pickle["pickle_name"]

print(object_model_dict)

object_name_list = list(object_model_dict.keys())
init_object_name = object_name_list[0]
init_pkl = object_model_dict[init_object_name]
init_sample_image_path = get_sample_image_path(init_pkl)

with gr.Blocks(title="Simple DragGAN") as app:

    # renderer = Renderer()
    global_state = gr.State({
        "images": {
            # image_orig: the original image, change with seed/model is changed
            # image_raw: image with mask and points, change durning optimization
            # image_show: image showed on screen
        },
        "temporal_params": {
            # stop
        },
        'mask':
        None,  # mask for visualization, 1 for editing and 0 for unchange
        'last_mask': None,  # last edited mask
        'show_mask': True,  # add button
        "generator_params": dnnlib.EasyDict(),
        "params": {
            "seed": 0,
            "motion_lambda": 20,
            "r1_in_pixels": 3,
            "r2_in_pixels": 12,
            "magnitude_direction_in_pixels": 1.0,
            "latent_space": "w+",
            "trunc_psi": 0.7,
            "trunc_cutoff": None,
            "lr": 0.001,
        },
        "device": device,
        "draw_interval": 1,
        "renderer": Renderer(disable_timing=True),
        "points": {},
        "curr_point": None,
        "curr_type_point": "start",
        'editing_state': 'add_points',
        'pretrained_weight': init_pkl,
        'w_load': None,
        'sample_image_path': init_sample_image_path,
    })

    # init image
    global_state = init_images(global_state)

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("""
                    # Simple DragGAN

                        
                    ## Quick Start

                    1. Select desired `Image Object` to generate an image of in `1. Select object to generate`.
                    2. Click on image to add points to drag the features in the image.
                    3. In `3. Drag Image`, click `Start` and enjoy it!
                    
                        
                    For more information, go to the \"Guide\" tab below!""")
        with gr.Column(scale=3):
            gr.Image(os.path.join(os.path.dirname(os.path.realpath(__file__)), "app_image/quick_start_guide_image_1.png"), interactive=False, label="Usage example (image taken from DragGAN paper)")
        with gr.Column(scale=3):
            print()

    with gr.Tab("DragGAN"):

        with gr.Row():

            # Left --> tools
            with gr.Column(scale=3):

                with gr.Accordion("1. Select image object to generate"):
                    cur_object = gr.Dropdown(object_name_list,
                                            value = init_object_name, label="Image Object")
            

                with gr.Accordion("2. Select Image Generation mode", open=False):
                    generation_method = gr.Radio(["Generate Random Image", "Generate Custom Image"],
                                                value = "Generate Random Image", interactive = True,
                                                label="Choose how you want the model to generate image")
                    gr.Markdown(value="Upload your image (Applicable only for \"Generate Custom Image\" mode)")
                    with gr.Row():
                        with gr.Column(scale=1, min_width=10):
                            sample_image = gr.Image(init_sample_image_path, interactive=False, label="Example of uploaded image:")
                        with gr.Column(scale=1, min_width=10):
                            uploaded_image = gr.Image(type="filepath", label="Upload your image here:")
                    gr.Markdown(value="*Upload a square image, cropped similar to the example image, for better custom image generation.")
                    accuracy_steps = gr.Slider(label="Image generation accuracy (Applicable only for \"Generate Custom Image\" mode)", value=1000, minimum=10, maximum=3000)
                    # gr.Markdown(value="*Adjust the accuracy of the image generation similar to custom image.\
                    #                     The higher the number, the closer the image will be to the custom image.\
                    #                     But it will take a longer time to generate the image.")
                    generate_button = gr.Button("Generate Image")

                with gr.Accordion("2.5. (Optional) Mask Image", open=False):
                    gr.Markdown(value='Mask areas to edit, and avoid unspecified areas to remain unchanged', show_label=False)
                    with gr.Row():
                        with gr.Column(scale=1, min_width=10):
                            enable_add_mask = gr.Button('Edit Flexible Area')
                        gr.Markdown(value='*Mask will be translucent in "3. Drag Image", the next step.', show_label=False)
                    # form_reset_mask_btn = gr.Button("Reset mask")

                with gr.Accordion("3. Drag Image", open=True):
                    gr.Markdown(value='Place points to drag', show_label=False)
                    with gr.Row():
                        with gr.Column(scale=1, min_width=10):
                            enable_add_points = gr.Button('Add Points')
                        with gr.Column(scale=1, min_width=10):
                            show_mask = gr.Checkbox(    
                                    label='Show Mask',
                                    value=global_state.value['show_mask'],
                                    show_label=False)
                    undo_points = gr.Button('Clear Points')

                    gr.Markdown(value='Start Dragging', show_label=False)
                    with gr.Row():
                        with gr.Column(scale=1, min_width=10):
                            form_start_btn = gr.Button("Start")
                        with gr.Column(scale=1, min_width=10):
                            form_stop_btn = gr.Button("Stop")

                    form_reset_image = gr.Button("Revert to Original Image")


                # # Latent
                # with gr.Accordion("Addtional options"):
                #     form_lr_number = gr.Number(
                #         value=global_state.value["params"]["lr"],
                #         interactive=True,
                #         label="Step Size")
                    
                #     form_latent_space = gr.Radio(
                #         ['w', 'w+'],
                #         value=global_state.value['params']
                #         ['latent_space'],
                #         interactive=True,
                #         label='Latent space to optimize',
                #         show_label=False,
                #     )

            # Right --> Image
            with gr.Column(scale=8):
                form_image = ImageMask(
                    value=global_state.value['images']['image_show'],
                    brush_radius=20).style(
                        width=768,
                        height=768)  # NOTE: hard image size code here.
    
    with gr.Tab("Guide"):
        gr.Markdown("""
            ## Advance Usage
            
            ### 1. Change how you want to change generate image in `2. Select Image Generation mode`.
            * `Generate Random Image` mode: The model will generate random image of the current `Image Object`.
            * `Generate Custom Image` mode: The model will try to generate an image similar to the image you have uploaded below.
            * To generate a good image, please upload an image that satisfy the following:
                1. Your uploaded image is an image of the current `Image Object` 
                2. Square image
                3. Image cropped similar to the example image (better if you could crop such that only the object is shown)
                4. Increase the accuracy of the image generation using the slider below.
            * Please take note that high accuracy does not gurantee a good image.
            * Example of failed (left) and successful (right) generation of custom image with different cropping:
                    """)
        with gr.Row():
            with gr.Column(scale=3):
                gr.Image(os.path.join(os.path.dirname(os.path.realpath(__file__)), "app_image/custom_image_failed.png"), interactive=False)
            with gr.Column(scale=3):
                gr.Image(os.path.join(os.path.dirname(os.path.realpath(__file__)), "app_image/custom_image_success.png"), interactive=False)
            with gr.Column(scale=3):
                print()

        gr.Markdown("""   
            (The original image of the man is taken from https://www.analyticsvidhya.com/blog/2022/03/facial-landmarks-detection-using-mediapipe-library/)  
                         
            ### 2. In `2.5 (Optional) Mask Image`, click `Edit Flexible Area` to create a mask and avoid the unmasked region to remain change while dragging.
            * The brush size of the mask can be changed with the slider on the top right of the image.
            * To clear the mask, click the button with the eraser icon, located on the top right corner of the image.
            * The mask will appear translucent when you click `Add Points` in `3. Drag Image`.
                    """)
        with gr.Row():
            with gr.Column(scale=6):
                gr.Image(os.path.join(os.path.dirname(os.path.realpath(__file__)), "app_image/mask_guide.png"), interactive=False)
            with gr.Column(scale=3):
                print()
        gr.Markdown("""
            ### 3. To download the edited image, click on the button on the top right corner,
            * Before downloading, be sure to `Reset mask` and `Reset Points` to get the image without the points and the masking.
                    """)
                    
    
    with gr.Tab("FAQ"):
        gr.Markdown("""
            ### 1. What is DragGAN?
            * Ans: DragGAN is an AI image editor that can drag features of an image to a specified destination.
                    (e.g. Close an eye of a person by placing a point on the eyelid and another point to tell where the eyelid should go to.)
                    
            ### 2. I want to edit my own image itself. How do I do that?
            * Ans: Currently DragGAN does not support editing of real image. However, you can edit a similar image, by generating it using the
            `Generate Custom Image` mode and the correct `Image Object` selected, and then edit it.
            * For example, if you want to edit an image of Golden Retriever dog, select "Dog (Body)" in the `Image Object` dropdown,
                upload an image of a golder retriever, and finally click generate. 
            
            ### 3. Where can I find the original DragGAN software?
            * Ans: The following GitHub Repository contains the original software: https://github.com/XingangPan/DragGAN
                    
                     
""")

    # gr.Markdown("""
    #     ## Quick Start

    #     1. Select desired `Pretrained Model` and adjust `Seed` to generate an
    #        initial image.
    #     2. Click on image to add control points.
    #     3. Click `Start` and enjoy it!

    #     ## Advance Usage

    #     1. Change `Step Size` to adjust learning rate in drag optimization.
    #     2. Select `w` or `w+` to change latent space to optimize:
    #     * Optimize on `w` space may cause greater influence to the image.
    #     * Optimize on `w+` space may work slower than `w`, but usually achieve
    #       better results.
    #     * Note that changing the latent space will reset the image, points and
    #       mask (this has the same effect as `Reset Image` button).
    #     3. Click `Edit Flexible Area` to create a mask and constrain the
    #        unmasked region to remain unchanged.
    #     """)
    gr.HTML("""
        <style>
            .container {
                position: absolute;
                height: 50px;
                text-align: center;
                line-height: 50px;
                width: 100%;
            }
        </style>
        <div class="container">
        Gradio demo supported by
        <img src="https://avatars.githubusercontent.com/u/10245193?s=200&v=4" height="20" width="20" style="display:inline;">
        <a href="https://github.com/open-mmlab/mmagic">OpenMMLab MMagic</a>
        </div>
        """)

    def on_click_generate_image(cur_object, generation_method, uploaded_image, accuracy_steps, global_state):
        if generation_method == "Generate Random Image":
            global_state["params"]["seed"] = random.randint(0, 2**32 - 1)
            w_load = None

        else:
            if uploaded_image:
                w_load = projector.run_projection(cur_object, uploaded_image, accuracy_steps)
            else:
                gr.Warning("Please upload an image for \"Generate Custom Image\" mode.")
                w_load = None

        global_state['pretrained_weight'] = object_model_dict[cur_object]
        global_state['w_load'] = w_load
        init_images(global_state)
        clear_state(global_state)
        return global_state, global_state["images"]['image_show']
            
    generate_button.click(
        on_click_generate_image,
        inputs = [cur_object, generation_method, uploaded_image, accuracy_steps, global_state],
        outputs = [global_state, form_image]
    )

    def on_change_object_dropdown(cur_object, global_state):
        # Change to the first pretrained model trained with the specified object
        global_state["pretrained_weight"] = object_model_dict[cur_object]
        init_images(global_state)
        clear_state(global_state)
        return global_state, global_state["images"]['image_show'], get_sample_image_path(object_model_dict[cur_object])

    cur_object.change(
        on_change_object_dropdown,
        inputs=[cur_object, global_state],
        outputs=[global_state, form_image, sample_image],
    )

    def on_click_reset_image(global_state):
        """Reset image to the original one and clear all states
        1. Re-init images
        2. Clear all states
        """

        init_images(global_state)
        clear_state(global_state)

        return global_state, global_state['images']['image_show']

    form_reset_image.click(
        on_click_reset_image,
        inputs=[global_state],
        outputs=[global_state, form_image],
    )

    def on_click_start(global_state, image):
        p_in_pixels = []
        t_in_pixels = []
        valid_points = []

        # handle of start drag in mask editing mode
        global_state = preprocess_mask_info(global_state, image)

        # Prepare the points for the inference
        if len(global_state["points"]) == 0:
            # yield on_click_start_wo_points(global_state, image)
            image_raw = global_state['images']['image_raw']
            update_image_draw(
                image_raw,
                global_state['points'],
                global_state['mask'],
                global_state['show_mask'],
                global_state,
            )

            yield (
                global_state,
                global_state['images']['image_show'],

                gr.Dropdown.update(interactive=True),

                gr.Radio.update(interactive=True),
                gr.Image.update(interactive=True),
                gr.Slider.update(interactive=True),
                gr.Button.update(interactive=True),

                gr.Button.update(interactive=True),

                gr.Button.update(interactive=True),
                gr.Checkbox.update(interactive=True),
                gr.Button.update(interactive=True),
                
                gr.Button.update(interactive=True),
                # NOTE: disable stop button
                gr.Button.update(interactive=False),
                gr.Button.update(interactive=True),
            )
        else:

            # Transform the points into torch tensors
            for key_point, point in global_state["points"].items():
                try:
                    p_start = point.get("start_temp", point["start"])
                    p_end = point["target"]

                    if p_start is None or p_end is None:
                        continue

                except KeyError:
                    continue

                p_in_pixels.append(p_start)
                t_in_pixels.append(p_end)
                valid_points.append(key_point)

            mask = torch.tensor(global_state['mask']).float()
            drag_mask = 1 - mask

            renderer: Renderer = global_state["renderer"]
            global_state['temporal_params']['stop'] = False
            global_state['editing_state'] = 'running'

            # reverse points order
            p_to_opt = reverse_point_pairs(p_in_pixels)
            t_to_opt = reverse_point_pairs(t_in_pixels)
            print('Running with:')
            print(f'    Source: {p_in_pixels}')
            print(f'    Target: {t_in_pixels}')
            step_idx = 0
            while True:
                if global_state["temporal_params"]["stop"]:
                    break

                # do drage here!
                renderer._render_drag_impl(
                    global_state['generator_params'],
                    p_to_opt,  # point
                    t_to_opt,  # target
                    drag_mask,  # mask,
                    global_state['params']['motion_lambda'],  # lambda_mask
                    reg=0,
                    feature_idx=5,  # NOTE: do not support change for now
                    r1=global_state['params']['r1_in_pixels'],  # r1
                    r2=global_state['params']['r2_in_pixels'],  # r2
                    # random_seed     = 0,
                    # noise_mode      = 'const',
                    trunc_psi=global_state['params']['trunc_psi'],
                    # force_fp32      = False,
                    # layer_name      = None,
                    # sel_channels    = 3,
                    # base_channel    = 0,
                    # img_scale_db    = 0,
                    # img_normalize   = False,
                    # untransform     = False,
                    is_drag=True,
                    to_pil=True)

                if step_idx % global_state['draw_interval'] == 0:
                    print('Current Source:')
                    for key_point, p_i, t_i in zip(valid_points, p_to_opt,
                                                   t_to_opt):
                        global_state["points"][key_point]["start_temp"] = [
                            p_i[1],
                            p_i[0],
                        ]
                        global_state["points"][key_point]["target"] = [
                            t_i[1],
                            t_i[0],
                        ]
                        start_temp = global_state["points"][key_point][
                            "start_temp"]
                        print(f'    {start_temp}')

                    image_result = global_state['generator_params']['image']
                    image_draw = update_image_draw(
                        image_result,
                        global_state['points'],
                        global_state['mask'],
                        global_state['show_mask'],
                        global_state,
                    )
                    global_state['images']['image_raw'] = image_result

                yield (
                    global_state,
                    global_state['images']['image_show'],

                    gr.Dropdown.update(interactive=False),

                    gr.Radio.update(interactive=False),
                    gr.Image.update(interactive=False),
                    gr.Slider.update(interactive=False),
                    gr.Button.update(interactive=False),

                    gr.Button.update(interactive=False),

                    gr.Button.update(interactive=False),
                    gr.Checkbox.update(interactive=False),
                    gr.Button.update(interactive=False),
                    
                    gr.Button.update(interactive=False),
                    # enable stop button in loop
                    gr.Button.update(interactive=True),
                    gr.Button.update(interactive=False),
                )

                # increate step
                step_idx += 1

            image_result = global_state['generator_params']['image']
            global_state['images']['image_raw'] = image_result
            image_draw = update_image_draw(image_result,
                                           global_state['points'],
                                           global_state['mask'],
                                           global_state['show_mask'],
                                           global_state)

            # fp = NamedTemporaryFile(suffix=".png", delete=False)
            # image_result.save(fp, "PNG")

            global_state['editing_state'] = 'add_points'

            yield (
                global_state,
                global_state['images']['image_show'],

                gr.Dropdown.update(interactive=True),

                gr.Radio.update(interactive=True),
                gr.Image.update(interactive=True),
                gr.Slider.update(interactive=True),
                gr.Button.update(interactive=True),

                gr.Button.update(interactive=True),

                gr.Button.update(interactive=True),
                gr.Checkbox.update(interactive=True),
                gr.Button.update(interactive=True),
                
                gr.Button.update(interactive=True),
                # NOTE: disable stop button with loop finish
                gr.Button.update(interactive=False),
                gr.Button.update(interactive=True),
            )

    form_start_btn.click(
        on_click_start,
        inputs=[global_state, form_image],
        outputs=[
            global_state,
            form_image,

            cur_object,

            generation_method,
            uploaded_image,
            accuracy_steps,
            generate_button,

            enable_add_mask,

            enable_add_points,
            show_mask,
            undo_points,
            
            form_start_btn,
            form_stop_btn,
            form_reset_image,
        ],
    )

    def on_click_stop(global_state):
        """Function to handle stop button is clicked.
        1. send a stop signal by set global_state["temporal_params"]["stop"] as True
        2. Disable Stop button
        """
        global_state["temporal_params"]["stop"] = True

        return global_state, gr.Button.update(interactive=False)

    form_stop_btn.click(on_click_stop,
                        inputs=[global_state],
                        outputs=[global_state, form_stop_btn])

    # form_draw_interval_number.change(
    #     partial(
    #         on_change_single_global_state,
    #         "draw_interval",
    #         map_transform=lambda x: int(x),
    #     ),
    #     inputs=[form_draw_interval_number, global_state],
    #     outputs=[global_state],
    # )

    def on_click_remove_point(global_state):
        choice = global_state["curr_point"]
        del global_state["points"][choice]

        choices = list(global_state["points"].keys())

        if len(choices) > 0:
            global_state["curr_point"] = choices[0]

        return (
            gr.Dropdown.update(choices=choices, value=choices[0]),
            global_state,
        )

    # Mask
    # def on_click_reset_mask(global_state):
    #     global_state['mask'] = np.ones(
    #         (
    #             global_state["images"]["image_raw"].size[1],
    #             global_state["images"]["image_raw"].size[0],
    #         ),
    #         dtype=np.uint8,
    #     )
    #     image_draw = update_image_draw(global_state['images']['image_raw'],
    #                                    global_state['points'],
    #                                    global_state['mask'],
    #                                    global_state['show_mask'], global_state)
    #     return global_state, image_draw

    # form_reset_mask_btn.click(
    #     on_click_reset_mask,
    #     inputs=[global_state],
    #     outputs=[global_state, form_image],
    # )

    # Image
    def on_click_enable_draw(global_state, image):
        """Function to start add mask mode.
        1. Preprocess mask info from last state
        2. Change editing state to add_mask
        3. Set curr image with points and mask
        """
        global_state = preprocess_mask_info(global_state, image)
        global_state['editing_state'] = 'add_mask'
        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(image_raw, global_state['points'],
                                       global_state['mask'], True,
                                       global_state)
        return (global_state,
                gr.Image.update(value=image_draw, interactive=True))

    def on_click_remove_draw(global_state, image):
        """Function to start remove mask mode.
        1. Preprocess mask info from last state
        2. Change editing state to remove_mask
        3. Set curr image with points and mask
        """
        global_state = preprocess_mask_info(global_state, image)
        global_state['edinting_state'] = 'remove_mask'
        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(image_raw, global_state['points'],
                                       global_state['mask'], True,
                                       global_state)
        return (global_state,
                gr.Image.update(value=image_draw, interactive=True))

    enable_add_mask.click(on_click_enable_draw,
                          inputs=[global_state, form_image],
                          outputs=[
                              global_state,
                              form_image,
                          ])

    def on_click_add_point(global_state, image: dict):
        """Function switch from add mask mode to add points mode.
        1. Updaste mask buffer if need
        2. Change global_state['editing_state'] to 'add_points'
        3. Set current image with mask
        """

        global_state = preprocess_mask_info(global_state, image)
        global_state['editing_state'] = 'add_points'
        mask = global_state['mask']
        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(image_raw, global_state['points'], mask,
                                       global_state['show_mask'], global_state)

        return (global_state,
                gr.Image.update(value=image_draw, interactive=False))

    enable_add_points.click(on_click_add_point,
                            inputs=[global_state, form_image],
                            outputs=[global_state, form_image])

    def on_click_image(global_state, evt: gr.SelectData):
        """This function only support click for point selection
        """
        xy = evt.index
        if global_state['editing_state'] != 'add_points':
            print(f'In {global_state["editing_state"]} state. '
                  'Do not add points.')

            return global_state, global_state['images']['image_show']

        points = global_state["points"]

        point_idx = get_latest_points_pair(points)
        if point_idx is None:
            points[0] = {'start': xy, 'target': None}
            print(f'Click Image - Start - {xy}')
        elif points[point_idx].get('target', None) is None:
            points[point_idx]['target'] = xy
            print(f'Click Image - Target - {xy}')
        else:
            points[point_idx + 1] = {'start': xy, 'target': None}
            print(f'Click Image - Start - {xy}')

        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(
            image_raw,
            global_state['points'],
            global_state['mask'],
            global_state['show_mask'],
            global_state,
        )

        return global_state, image_draw

    form_image.select(
        on_click_image,
        inputs=[global_state],
        outputs=[global_state, form_image],
    )

    def on_click_clear_points(global_state):
        """Function to handle clear all control points
        1. clear global_state['points'] (clear_state)
        2. re-init network
        2. re-draw image
        """
        clear_state(global_state, target='point')

        renderer: Renderer = global_state["renderer"]
        renderer.feat_refs = None

        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(image_raw, {}, global_state['mask'],
                                       global_state['show_mask'], global_state)
        return global_state, image_draw

    undo_points.click(on_click_clear_points,
                      inputs=[global_state],
                      outputs=[global_state, form_image])

    def on_click_show_mask(global_state, show_mask):
        """Function to control whether show mask on image."""
        global_state['show_mask'] = show_mask

        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(
            image_raw,
            global_state['points'],
            global_state['mask'],
            global_state['show_mask'],
            global_state,
        )
        return global_state, image_draw

    show_mask.change(
        on_click_show_mask,
        inputs=[global_state, show_mask],
        outputs=[global_state, form_image],
    )

    

gr.close_all()
app.queue(concurrency_count=3, max_size=20)
app.launch(share=args.share, server_name="0.0.0.0" if args.listen else "127.0.0.1", inbrowser=True)
