import os
import shutil
import gradio as gr
import sys
from PIL import Image
import importlib.util
import argparse
import socket
import webbrowser
import threading
import toml


# ログでエラーが出るので、念のため環境変数を設定
os.environ['TERM'] = 'dumb'

# ビルドしているかしていないかでパスを変更
if getattr(sys, 'frozen', False):
    path = os.path.dirname(sys.executable)
    sd_scripts_dir = os.path.join(path, "_internal", 'sd-scripts')
    networks_path = os.path.join(sd_scripts_dir, 'networks')
    library_path = os.path.join(sd_scripts_dir, 'library')
    tools_path = os.path.join(sd_scripts_dir, 'tools')
    utils_path = os.path.join(path, "_internal", 'utils')
else:
    path = os.path.dirname(os.path.abspath(__file__))
    sd_scripts_dir = os.path.join(path, 'sd-scripts')
    networks_path = os.path.join(sd_scripts_dir, 'networks')
    library_path = os.path.join(sd_scripts_dir, 'library')
    tools_path = os.path.join(sd_scripts_dir, 'tools')
    utils_path = os.path.join(path, 'utils')


# パスをシステムパスに追加
sys.path.append(sd_scripts_dir)
sys.path.append(networks_path)
sys.path.append(library_path)
sys.path.append(tools_path)
sys.path.append(utils_path)

import tagger
import sdxl_train_network

# モジュールのパスを直接指定してインポート
spec_merge = importlib.util.spec_from_file_location("merge", os.path.join(networks_path, 'sdxl_merge_lora.py'))
merge = importlib.util.module_from_spec(spec_merge)
spec_merge.loader.exec_module(merge)

spec_resize = importlib.util.spec_from_file_location("resize", os.path.join(networks_path, 'resize_lora.py'))
resize = importlib.util.module_from_spec(spec_resize)
spec_resize.loader.exec_module(resize)

models_dir = os.path.join(path, "models")
sdxl_dir = os.path.join(models_dir, "SDXL")
tagger_dir = os.path.join(models_dir, "tagger")
lora_dir = os.path.join(models_dir, "LoRA")
train_data_dir = os.path.join(path, "train_data")
image_dir = os.path.join(train_data_dir, "4000")
config_path = os.path.join(path, "config.toml")
repo_id = 'SmilingWolf/wd-swinv2-tagger-v3'
tagger_model = tagger.modelLoad(tagger_dir, repo_id)

# ベースモデル候補を取得する関数
def get_base_model_options():
    """sdxl_dir の中身を走査してファイル名のリストを返す"""
    return [f for f in os.listdir(sdxl_dir) if f.endswith(".safetensors")]

# base_model を選択肢として更新するための関数
def update_base_model_options():
    return gr.Dropdown.update(choices=get_base_model_options())

def find_free_port(start_port=7860):
    """指定したポートから開始して空いているポートを見つけて返す関数"""
    for port in range(start_port, 65535):  # 65535はポート番号の最大値
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('localhost', port))
            sock.close()
            return port  # バインドに成功したらそのポート番号を返す
        except OSError:
            continue
    raise RuntimeError("No free ports available.")  # 空いているポートが見つからなかった場合

def analyze_tags(image_path):
    """画像に対してタグ分析を行い、タグテキストを返す関数"""
    if not os.path.exists(image_path):
        return "画像が見つかりません"
    tagger_model_location = os.path.join(tagger_dir, repo_id.replace("/", "_"))
    tag_text = tagger.analysis(image_path, tagger_model_location, tagger_model)  # taggerモジュールのanalysis関数を使用
    return tag_text

def detail_train(
        base_model,
        detail_lora_name,
        image_num,
        *args,
    ):
    detail_base_img_path = args[:50][:image_num]
    detail_base_img_caption = args[50:100][:image_num]
    detail_input_image_path = args[100:150][:image_num]
    detail_input_image_caption = args[150:][:image_num]

    # 学習データのセットアップ
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
    os.makedirs(image_dir)

    output_dir = os.path.join(path, "output")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for i in range(image_num):
        input_image = Image.open(detail_base_img_path[i])

        for size in [1024, 768, 512]:
            resize_image = input_image.resize((size, size))
            resize_image.save(os.path.join(image_dir, f"{i}_{size}.png"))

            caption_size_txt = os.path.join(image_dir, f"{i}_{size}.txt")
            with open(caption_size_txt, "w") as f2:
                f2.write(detail_base_img_caption[i])

    # 訓練のセットアップ
    with open(config_path, 'r') as f:
        config = toml.load(f)

    base_model_path = os.path.join(sdxl_dir, base_model)
    # base_model_path = 'Linaqruf/animagine-xl'
    base_lora_name = "copi-ki-base"

    args_dict = {
        "pretrained_model_name_or_path": base_model_path,
        "train_data_dir": config["train_data_dir"],
        "output_dir": config["output_dir"],
        "output_name": base_lora_name,
        "max_train_steps": config["max_train_steps"],
        "network_module": config["network_module"],
        "xformers": config["xformers"],
        "gradient_checkpointing": config["gradient_checkpointing"],
        "persistent_data_loader_workers": config["persistent_data_loader_workers"],
        "max_data_loader_n_workers": config["max_data_loader_n_workers"],
        "enable_bucket": config["enable_bucket"],
        "save_model_as": config["save_model_as"],
        "lr_scheduler_num_cycles": config["lr_scheduler_num_cycles"],
        "learning_rate": config["learning_rate"],
        "resolution": config["resolution"],
        "train_batch_size": config["train_batch_size"],
        "network_dim": config["network_dim"],
        "network_alpha": config["network_alpha"],
        "optimizer_type": config["optimizer_type"],
        "mixed_precision": config["mixed_precision"],
        "save_precision": config["save_precision"],
        "lr_scheduler": config["lr_scheduler"],
        "bucket_no_upscale": config["bucket_no_upscale"],
        "min_bucket_reso": config["min_bucket_reso"],
        "max_bucket_reso": config["max_bucket_reso"],
        "caption_extension": config["caption_extension"],
        "seed": config["seed"],
        "network_train_unet_only": config["network_train_unet_only"],
        "no_half_vae": config["no_half_vae"],
        "cache_latents": config["cache_latents"],
        "cache_latents_to_disk": config["cache_latents_to_disk"],
        "cache_text_encoder_outputs": config["cache_text_encoder_outputs"],
        "cache_text_encoder_outputs_to_disk": config["cache_text_encoder_outputs_to_disk"],
        "fp8_base": config["fp8_base"],
        "console_log_simple": config["console_log_simple"],
        "lowram": config["lowram"],
    }

    parser = sdxl_train_network.setup_parser()
    args = parser.parse_args()
    sdxl_train_network.train_util.verify_command_line_training_args(args)
    args = sdxl_train_network.train_util.read_config_from_file(args, parser)
    args2 = argparse.Namespace(**args_dict)
    for key, value in vars(args2).items():
        setattr(args, key, value)
    trainer = sdxl_train_network.SdxlNetworkTrainer()
    trainer.train(args)
    
    # 学習データのセットアップ
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
    os.makedirs(image_dir)

    for i in range(image_num):    
        input_image = Image.open(detail_input_image_path[i])

        for size in [1024, 768, 512]:
            resize_image = input_image.resize((size, size))
            resize_image.save(os.path.join(image_dir, f"{size}.png"))

            caption_size_txt = os.path.join(image_dir, f"{i}_{size}.txt")
            with open(caption_size_txt, "w") as f2:
                f2.write(detail_input_image_caption[i])

    kari_lora_name = "copi-ki-kari"
    args_dict = {
        "pretrained_model_name_or_path": base_model_path,
        "train_data_dir": config["train_data_dir"],
        "output_dir": config["output_dir"],
        "output_name": kari_lora_name,
        "max_train_steps": config["max_train_steps"],
        "network_module": config["network_module"],
        "xformers": config["xformers"],
        "gradient_checkpointing": config["gradient_checkpointing"],
        "persistent_data_loader_workers": config["persistent_data_loader_workers"],
        "max_data_loader_n_workers": config["max_data_loader_n_workers"],
        "enable_bucket": config["enable_bucket"],
        "save_model_as": config["save_model_as"],
        "lr_scheduler_num_cycles": config["lr_scheduler_num_cycles"],
        "learning_rate": config["learning_rate"],
        "resolution": config["resolution"],
        "train_batch_size": config["train_batch_size"],
        "network_dim": config["network_dim"],
        "network_alpha": config["network_alpha"],
        "optimizer_type": config["optimizer_type"],
        "mixed_precision": config["mixed_precision"],
        "save_precision": config["save_precision"],
        "lr_scheduler": config["lr_scheduler"],
        "bucket_no_upscale": config["bucket_no_upscale"],
        "min_bucket_reso": config["min_bucket_reso"],
        "max_bucket_reso": config["max_bucket_reso"],
        "caption_extension": config["caption_extension"],
        "seed": config["seed"],
        "network_train_unet_only": config["network_train_unet_only"],
        "no_half_vae": config["no_half_vae"],
        "cache_latents": config["cache_latents"],
        "cache_latents_to_disk": config["cache_latents_to_disk"],
        "cache_text_encoder_outputs": config["cache_text_encoder_outputs"],
        "cache_text_encoder_outputs_to_disk": config["cache_text_encoder_outputs_to_disk"],
        "fp8_base": config["fp8_base"],
        "console_log_simple": config["console_log_simple"],
        "lowram": config["lowram"],
    }

    parser = sdxl_train_network.setup_parser()
    args = parser.parse_args()
    sdxl_train_network.train_util.verify_command_line_training_args(args)
    args = sdxl_train_network.train_util.read_config_from_file(args, parser)
    args2 = argparse.Namespace(**args_dict)
    for key, value in vars(args2).items():
        setattr(args, key, value)
    trainer = sdxl_train_network.SdxlNetworkTrainer()
    trainer.train(args)

    base_lora  = os.path.join(lora_dir , f"{base_lora_name}.safetensors")
    kari_lora = os.path.join(lora_dir , f"{kari_lora_name}.safetensors")
    merge_lora = os.path.join(lora_dir , "merge_lora.safetensors")
    train_lora = os.path.join(output_dir, f"{detail_lora_name}.safetensors")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    args_dict = {
        "save_precision": "bf16",
        "precision": "float",
        "sd_model": None,
        "save_to": merge_lora,
        "models": [kari_lora, base_lora],
        "no_metadata": False,
        "ratios": [1.41, -1.41],
        "concat": True,
        "shuffle": True,
         "lbws": [],       
    }
    args = argparse.Namespace(**args_dict)
    merge.merge(args)

    args_dict = {
        "save_precision": "bf16",
        "new_rank": 16,
        "new_conv_rank": 16,
        "save_to": train_lora,
        "model": merge_lora,
        "device": "cuda",
        "verbose": "store_true",
        "dynamic_param": None,
        "dynamic_method": None,
        "lbws": [],        
    }
    args = argparse.Namespace(**args_dict)
    resize.resize(args)
    return train_lora

def main():
    with gr.Blocks() as demo:
        base_model_options = get_base_model_options()
        with gr.Column():
            with gr.Row():
                base_model = gr.Dropdown(label="Base Model", choices=base_model_options, value="animagine-xl-3.1.safetensors")
                update_button = gr.Button("List Update")

            image_num = gr.Slider(label="Number of Images", value=1, minimum=0, maximum=50, step=1)

            with gr.Column():
                img_rows = []
                detail_base_img_paths = []
                detail_base_img_captions = []
                analyze_base_img_buttons = []
                detail_input_image_paths = []
                detail_input_image_captions = []
                analyze_input_img_buttons = []
                for i in range(50):
                    gr.Markdown(f"### Image {i + 1}")
                    with gr.Row(visible=False) as row:
                        with gr.Row():
                            detail_base_img_path = gr.Image(label="Detail Base Input Image", type='filepath')
                            detail_base_img_caption = gr.Textbox(label="Caption Text")
                            analyze_base_img_button = gr.Button("Analyze Tags for Base Image")
                        with gr.Row():
                            detail_input_image_path = gr.Image(label="Detail Input Image", type='filepath')
                            detail_input_image_caption = gr.Textbox(label="Caption Text")
                            analyze_input_img_button = gr.Button("Analyze Tags for Input Image")
                    analyze_base_img_button.click(
                        fn=analyze_tags,
                        inputs=[detail_base_img_path],
                        outputs=detail_base_img_caption
                    )
                    analyze_input_img_button.click(
                        fn=analyze_tags,
                        inputs=[detail_input_image_path],
                        outputs=detail_input_image_caption
                    )
                    detail_base_img_paths.append(detail_base_img_path)
                    detail_base_img_captions.append(detail_base_img_caption)
                    analyze_base_img_buttons.append(analyze_base_img_button)
                    detail_input_image_paths.append(detail_input_image_path)
                    detail_input_image_captions.append(detail_input_image_caption)
                    analyze_input_img_buttons.append(analyze_input_img_button)
                    
                    img_rows.append(row)

            detail_lora_name = gr.Textbox(label="LoRa Name", value="")
            detail_train_button = gr.Button("Train")

        with gr.Column():
            detail_output_file = gr.File(label="Download Output File")

        update_button.click(
            fn=update_base_model_options,
            inputs=[],
            outputs=base_model
        )

        detail_train_button.click(
            fn=detail_train,
            inputs=[
                base_model,
                detail_lora_name,
                image_num,
                *detail_base_img_paths,
                *detail_base_img_captions,
                *detail_input_image_paths,
                *detail_input_image_captions
            ],
            outputs=detail_output_file
        )
        
        def update_imgs_visibility(num):
            update_img_rows = [gr.Row.update(visible=i < num) for i in range(50)]
            return update_img_rows

        image_num.change(
            fn=update_imgs_visibility,
            inputs=[image_num],
            outputs = img_rows
        )
        img_rows = update_imgs_visibility(1)

    demo.queue()
    port = find_free_port()
    url = f"http://127.0.0.1:{port}"

    threading.Thread(target=lambda: webbrowser.open_new(url)).start()
    is_colab = 'COLAB_GPU' in os.environ
    share_setting = True if is_colab else False
    demo.launch(share=share_setting, server_name="0.0.0.0", server_port=port)

if __name__ == "__main__":
    main()
