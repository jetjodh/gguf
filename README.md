### gguf node for comfyui [![Static Badge](https://img.shields.io/badge/ver-0.2.0-black?logo=github)](https://github.com/calcuis/gguf/releases)

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
opt `2` to clone the gguf repo to the current directory (either navigate to `./ComfyUI/custom_nodes` first or drag and drop there after the clone)

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

#### other(s): get it somewhere else trustworthy/reliable
you are also welcome to get the node through other available channels, i.e., comfy-cli, comfyui-manager (search `gguf` from the bar; and opt to install it there should be fine; see picture below), etc.
![screenshot](https://raw.githubusercontent.com/calcuis/comfy/master/comfyui-manager.png)
`gguf` node is no conflict with the popular `comfyui-gguf` node (can coexist; and this project actually inspired by it; built upon its code base; we are here honor their developers' contribution; we all appreciate their great work truly; then you could test our version and their version both; or mixing up use, switch in between freely, all for your own purpose and need); and is more lightweight (no dependencies needed), more functions (i.e., built-in `tensor cutter`, `gguf convertor`, etc.), compatible with the latest version numpy and other updated libraries come with comfyui

![screenshot](https://raw.githubusercontent.com/calcuis/comfy/master/demo4.png)
for the demo workflow (picture) above, you could get the test model gguf [here](https://huggingface.co/calcuis/illustrious), test it whether you can generate the similar outcome or not

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

#### cutter (cut safetensors in half - bf16 to fp8)
- drag safetensors file(s) to diffusion_models folder (./ComfyUI/models/diffusion_models)
- choose the last option from the gguf menu: `TENSOR Cutter (Beta)`
- select your safetensors model inside the box; don't need to connect anything; it works independently
- click `Queue` (run); then you can simply check the processing progress from console
- when it was done; the quantized/half-cut safetensors file will be saved to the output folder (./ComfyUI/output)

![screenshot](https://raw.githubusercontent.com/calcuis/comfy/master/cutter.png)

#### convertor (convert safetensors to gguf)
- drag safetensors file(s) to diffusion_models folder (./ComfyUI/models/diffusion_models)
- choose the third last option from the gguf menu: `GGUF Convertor (Alpha)`
- select your safetensors model inside the box; don't need to connect anything; it works independently also
- click `Queue` (run); then you can simply check the processing progress from console
- when it was done; the converted gguf file will be saved to the output folder (./ComfyUI/output)

![screenshot](https://raw.githubusercontent.com/calcuis/comfy/master/convertor.png)
**little tips**: to make a so-called `fast` model; could try to cut the selected model (bf16) half (use cutter) first; and convert the trimmed model (fp8) to gguf (pretty much the same file size with the bf16 quantized output but less tensors inside; load faster theoretically, but no guarantee, you should test it probably, and might also be prepared for the significant quality loss)

#### reverser (reverse convert gguf to safetensors)
- drag gguf file(s) to diffusion_models folder (./ComfyUI/models/diffusion_models)
- choose the seventh option from the gguf menu: `GGUF Convertor (Reverse)`
- select your gguf file inside the box; don't need to connect anything; it works independently as well
- click `Queue` (run); then you can simply check the processing progress from console
- when it was done; the converted safetensors file will be saved to the output folder (./ComfyUI/output)

![screenshot](https://raw.githubusercontent.com/calcuis/comfy/master/reverser.png)
**little little tips**: the reverse converted safetensors file doesn't contain any clip and vae (cannot be used as checkpoint); should drag it to diffusion_models folder (./ComfyUI/models/diffusion_models) and select Add Node > advanced > loaders > Load Diffusion Model; then use it like gguf model (very similar to gguf loader) along with separate clip(s) and vae

### convertor ZERO
#### new flagship feature: convert safetensors to gguf without any restriction; no unsupported models anymore; never
- drag safetensors file(s) to diffusion_models folder (./ComfyUI/models/diffusion_models)
- choose the second last option from the gguf menu: `GGUF Convertor (Zero)`
- select your safetensors model inside the box; don't need to connect anything; it works independently!
- click `Queue` (run); then you can simply check the processing progress from console
- when it was done; the converted gguf file will be saved to the output folder (./ComfyUI/output)
- 🐷**pig architecture**: not merely model conversion; it works for converting text encoder and vae as well; true! the amazing thing is - any form of safetensors can be converted! (but the file works or not; is another story)🐷

### GGUF VAE Loader🐷
- you can't IMAGINE it works; but it really works!
- convert you safetensors vae to gguf vae using convertor zero; then try it out! 

#### reference
[comfyui](https://github.com/comfyanonymous/ComfyUI)
[confyui_vlm_nodes](https://github.com/gokayfem/ComfyUI_VLM_nodes)
[comfyui-gguf](https://github.com/city96/ComfyUI-GGUF) (special thanks city96)
[gguf-comfy](https://github.com/calcuis/gguf-comfy)
[gguf-connector](https://github.com/calcuis/gguf-connector)
[testkit](https://huggingface.co/calcuis/gguf-node)

#### parent
node is a member of family [gguf](https://gguf.org)
