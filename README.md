### gguf node for comfyui [![Static Badge](https://img.shields.io/badge/ver-0.1.1-black?logo=github)](https://github.com/calcuis/gguf/releases)

[<img src="https://raw.githubusercontent.com/calcuis/comfy/master/gguf.gif" width="128" height="128">](https://github.com/calcuis/gguf)

#### install it via pip/pip3
```
pip install gguf-node
```
#### enter the user menu by (if no py command; use python/python3 instead)
```
py -m gguf_node
```
>Please select:
>1. download the full pack
>2. clone the node only
>
>Enter your choice (1 to 2): _
#### for new/all user(s)
opt `1` to download the compressed comfy pack (7z), decompress it, and run the .bat file striaght (idiot option)

#### for existing user/developer(s)
opt `2` to clone the gguf repo to the current directory (navigate to `./ComfyUI/custom_nodes` first)

alternatively, you could execute the git clone command to perform that task (see below):
- navigate to `./ComfyUI/custom_nodes`
- clone the gguf repo to that folder by
```
git clone https://github.com/calcuis/gguf
```
*same operation for the standalone pack; then you should be able to see it under `Add Node >`*
![screenshot](https://raw.githubusercontent.com/calcuis/comfy/master/gguf-node.png)
check the dropdown menu for `gguf`

🐷🐷📄 for the latest update, gguf-connector deployment copy is now attached to the node itself; don't need to clone it to site-packages; and, as the default setting in comfyui is sufficient; no dependencies needed right away 🙌 no extra step anymore

#### other(s)
you are also welcome to get the node through other channels, i.e., comfy-cli, comfyui-manager (available now; search `gguf` from the bar; see picture below)
![screenshot](https://raw.githubusercontent.com/calcuis/comfy/master/comfyui-manager.png)

#### setup (in general)
- drag gguf file(s) to diffusion_models folder (./ComfyUI/models/diffusion_models)
- drag clip or encoder(s) to text_encoders folder (./ComfyUI/models/text_encoders)
- drag controlnet adapter(s), if any, to controlnet folder (./ComfyUI/models/controlnet)
- drag lora adapter(s), if any, to loras folder (./ComfyUI/models/loras)
- drag vae decoder(s) to vae folder (./ComfyUI/models/vae)

#### workflow
- drag the workflow json file to the activated browser; or
- drag any generated output file (i.e., picture, video, etc.; which contains the workflow metadata) to the activated browser

#### simulator
- design your own prompt; or
- generate a random prompt/descriptor by the [simulator](https://prompt.calcuis.us) (though it might not be applicable for all)

#### convertor (new feature: convert safetensors to gguf)
- drag safetensors file(s) to diffusion_models folder (./ComfyUI/models/diffusion_models)
- choose the last option from the gguf menu: `GGUF Convertor (Alpha)`
- select your safetensors model inside the box; don't need to connect anything; it works independently
- click `Queue` (run); then you can simply check the processing progress from console
- when it was done; the converted gguf file will be saved in the output folder (./ComfyUI/output)

![screenshot](https://raw.githubusercontent.com/calcuis/comfy/master/convertor.png)

#### reference
[comfyui](https://github.com/comfyanonymous/ComfyUI)
[confyui_vlm_nodes](https://github.com/gokayfem/ComfyUI_VLM_nodes)
[comfyui-gguf](https://github.com/city96/ComfyUI-GGUF)
[gguf-comfy](https://github.com/calcuis/gguf-comfy)
[gguf-connector](https://github.com/calcuis/gguf-connector)
[testkit](https://huggingface.co/calcuis/gguf-node)
