import comfy.sd
import comfy.ops
import comfy.utils
import comfy.model_patcher
import comfy.model_management
import torch, os, logging, collections, folder_paths
from safetensors.torch import load_file, save_file
from tqdm import tqdm as loading
from .gguf_connector import reader as gr
from .gguf_connector.writer import GGUFWriter, GGMLQuantizationType
from .gguf_connector.const import GGML_QUANT_VERSION, LlamaFileType
from .gguf_connector.quant import quantize, QuantError
from .gguf_connector.quant2 import dequantize_tensor, is_quantized, is_torch_compatible
class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    patch_on_device = False
    def patch_weight_to_device(self, key, device_to=None, inplace_update=False
        ):
        if key not in self.patches:
            return
        weight = comfy.utils.get_attr(self.model, key)
        try:
            from comfy.lora import calculate_weight
        except Exception:
            calculate_weight = self.calculate_weight
        patches = self.patches[key]
        if is_quantized(weight):
            out_weight = weight.to(device_to)
            patches = load_patch_to_device(patches, self.load_device if
                self.patch_on_device else self.offload_device)
            out_weight.patches = [(calculate_weight, patches, key)]
        else:
            inplace_update = self.weight_inplace_update or inplace_update
            if key not in self.backup:
                self.backup[key] = collections.namedtuple('Dimension', [
                    'weight', 'inplace_update'])(weight.to(device=self.
                    offload_device, copy=inplace_update), inplace_update)
            if device_to is not None:
                temp_weight = comfy.model_management.cast_to_device(weight,
                    device_to, torch.float32, copy=True)
            else:
                temp_weight = weight.to(torch.float32, copy=True)
            out_weight = calculate_weight(patches, temp_weight, key)
            out_weight = comfy.float.stochastic_rounding(out_weight, weight
                .dtype)
        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)
    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for p in self.model.parameters():
                if is_torch_compatible(p):
                    continue
                patches = getattr(p, 'patches', [])
                if len(patches) > 0:
                    p.patches = []
        return super().unpatch_model(device_to=device_to, unpatch_weights=
            unpatch_weights)
    mmap_released = False
    def load(self, *args, force_patch_weights=False, **kwargs):
        super().load(*args, force_patch_weights=True, **kwargs)
        if not self.mmap_released:
            linked = []
            if kwargs.get('lowvram_model_memory', 0) > 0:
                for n, m in self.model.named_modules():
                    if hasattr(m, 'weight'):
                        device = getattr(m.weight, 'device', None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
                    if hasattr(m, 'bias'):
                        device = getattr(m.bias, 'device', None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
            if linked:
                print(f'Attempting to release mmap ({len(linked)})')
                for n, m in linked:
                    m.to(self.load_device).to(self.offload_device)
            self.mmap_released = True
    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        self.__class__ = GGUFModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = GGUFModelPatcher
        self.__class__ = src_cls
        n.patch_on_device = getattr(self, 'patch_on_device', False)
        return n
class GGMLTensor(torch.Tensor):
    def __init__(self, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        self.patches = patches
    def __new__(cls, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        return super().__new__(cls, *args, **kwargs)
    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, 'tensor_type', None)
        new.tensor_shape = getattr(self, 'tensor_shape', new.data.shape)
        new.patches = getattr(self, 'patches', []).copy()
        return new
    def clone(self, *args, **kwargs):
        return self
    def detach(self, *args, **kwargs):
        return self
    def copy_(self, *args, **kwargs):
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:
            print(f"ignoring 'copy_' on tensor: {e}")
    def empty_(self, size, *args, **kwargs):
        new_tensor = super().empty_(size, *args, **kwargs)
        return GGMLTensor(new_tensor, tensor_type=getattr(self,
            'tensor_type', None), tensor_shape=size, patches=getattr(self,
            'patches', []).copy())
    @property
    def shape(self):
        if not hasattr(self, 'tensor_shape'):
            self.tensor_shape = self.size()
        return self.tensor_shape
class GGMLLayer(torch.nn.Module):
    comfy_cast_weights = True
    dequant_dtype = None
    patch_dtype = None
    largest_layer = False
    torch_compatible_tensor_types = {None, gr.GGMLQuantizationType.F32, gr.
        GGMLQuantizationType.F16}
    def is_ggml_quantized(self, *, weight=None, bias=None):
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        return is_quantized(weight) or is_quantized(bias)
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight, bias = state_dict.get(f'{prefix}weight'), state_dict.get(
            f'{prefix}bias')
        if self.is_ggml_quantized(weight=weight, bias=bias) or isinstance(self,
            torch.nn.Linear):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args,
                **kwargs)
        return super()._load_from_state_dict(state_dict, prefix, *args, **
            kwargs)
    def ggml_load_from_state_dict(self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs):
        prefix_len = len(prefix)
        for k, v in state_dict.items():
            if k[prefix_len:] == 'weight':
                self.weight = torch.nn.Parameter(v, requires_grad=False)
            elif k[prefix_len:] == 'bias' and v is not None:
                self.bias = torch.nn.Parameter(v, requires_grad=False)
            else:
                missing_keys.append(k)
        if self.weight is None and isinstance(self, torch.nn.Linear):
            v = torch.zeros(self.in_features, self.out_features)
            self.weight = torch.nn.Parameter(v, requires_grad=False)
            missing_keys.append(prefix + 'weight')
        if getattr(self.weight, 'is_largest_weight', False):
            self.largest_layer = True
    def _save_to_state_dict(self, *args, **kwargs):
        if self.is_ggml_quantized():
            return self.ggml_save_to_state_dict(*args, **kwargs)
        return super()._save_to_state_dict(*args, **kwargs)
    def ggml_save_to_state_dict(self, destination, prefix, keep_vars):
        weight = torch.zeros_like(self.weight, device=torch.device('meta'))
        destination[prefix + 'weight'] = weight
        if self.bias is not None:
            bias = torch.zeros_like(self.bias, device=torch.device('meta'))
            destination[prefix + 'bias'] = bias
        if self.largest_layer:
            shape = getattr(self.weight, 'tensor_shape', self.weight.shape)
            dtype = self.dequant_dtype or torch.float16
            temp = torch.empty(*shape, device=torch.device('meta'), dtype=dtype
                )
            destination[prefix + 'temp.weight'] = temp
        return
        destination[prefix + 'weight'] = self.get_weight(self.weight)
        if bias is not None:
            destination[prefix + 'bias'] = self.get_weight(self.bias)
    def get_weight(self, tensor, dtype):
        if tensor is None:
            return
        patch_list = []
        device = tensor.device
        for function, patches, key in getattr(tensor, 'patches', []):
            patch_list += load_patch_to_device(patches, device)
        weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)
        if isinstance(weight, GGMLTensor):
            weight.__class__ = torch.Tensor
        if patch_list:
            if self.patch_dtype is None:
                weight = function(patch_list, weight, key)
            else:
                patch_dtype = (dtype if self.patch_dtype == 'target' else
                    self.patch_dtype)
                weight = function(patch_list, weight, key, patch_dtype)
        return weight
    def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype
        =None):
        if input is not None:
            if dtype is None:
                dtype = getattr(input, 'dtype', torch.float32)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device
        bias = None
        non_blocking = comfy.model_management.device_supports_non_blocking(
            device)
        if s.bias is not None:
            bias = s.get_weight(s.bias.to(device), dtype)
            bias = comfy.ops.cast_to(bias, bias_dtype, device, non_blocking
                =non_blocking, copy=False)
        weight = s.get_weight(s.weight.to(device), dtype)
        weight = comfy.ops.cast_to(weight, dtype, device, non_blocking=
            non_blocking, copy=False)
        return weight, bias
    def forward_comfy_cast_weights(self, input, *args, **kwargs):
        if self.is_ggml_quantized():
            out = self.forward_ggml_cast_weights(input, *args, **kwargs)
        else:
            out = super().forward_comfy_cast_weights(input, *args, **kwargs)
        if isinstance(out, GGMLTensor):
            out.__class__ = torch.Tensor
        return out
    def forward_ggml_cast_weights(self, input):
        raise NotImplementedError
class GGMLOps(comfy.ops.manual_cast):
    class Linear(GGMLLayer, comfy.ops.manual_cast.Linear):
        def __init__(self, in_features, out_features, bias=True, device=
            None, dtype=None):
            torch.nn.Module.__init__(self)
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.linear(input, weight, bias)
    class Conv2d(GGMLLayer, comfy.ops.manual_cast.Conv2d):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return self._conv_forward(input, weight, bias)
    class Embedding(GGMLLayer, comfy.ops.manual_cast.Embedding):
        def forward_ggml_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype
            if (self.weight.dtype == torch.float16 or self.weight.dtype ==
                torch.bfloat16):
                out_dtype = None
            weight, _bias = self.cast_bias_weight(self, device=input.device,
                dtype=out_dtype)
            return torch.nn.functional.embedding(input, weight, self.
                padding_idx, self.max_norm, self.norm_type, self.
                scale_grad_by_freq, self.sparse).to(dtype=output_dtype)
    class LayerNorm(GGMLLayer, comfy.ops.manual_cast.LayerNorm):
        def forward_ggml_cast_weights(self, input):
            if self.weight is None:
                return super().forward_comfy_cast_weights(input)
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.layer_norm(input, self.
                normalized_shape, weight, bias, self.eps)
    class GroupNorm(GGMLLayer, comfy.ops.manual_cast.GroupNorm):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.group_norm(input, self.num_groups,
                weight, bias, self.eps)
def load_patch_to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(load_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [load_patch_to_device(x, device) for x in item]
    else:
        return item
def get_folder_names_and_paths(key, targets=[]):
    base = folder_paths.folder_names_and_paths.get(key, ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    target = next((x for x in targets if x in folder_paths.
        folder_names_and_paths), targets[0])
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = orig or base, {'.gguf'}
    if base and base != orig:
        logging.warning(
            f'Unknown file list already present on key {key}: {base}')
get_folder_names_and_paths('model_gguf', ['diffusion_models', 'unet'])
get_folder_names_and_paths('clip_gguf', ['text_encoders', 'clip'])
CUS_ARCH_LIST = {'flux', 'sd1', 'sdxl', 'sd3', 'aura', 'mochi', 'ltxv', 'hyvid', 'cosmos'}
TXT_ARCH_LIST = {'t5', 't5encoder', 'llama'}
def get_orig_shape(reader, tensor_name):
    field_key = f'comfy.gguf.orig_shape.{tensor_name}'
    field = reader.get_field(field_key)
    if field is None:
        return None
    if len(field.types) != 2 or field.types[0
        ] != gr.GGUFValueType.ARRAY or field.types[1
        ] != gr.GGUFValueType.INT32:
        raise TypeError(
            f'Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}'
            )
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in
        field.data))
def load_gguf_sd(path, handle_prefix='model.diffusion_model.',
    return_arch=False):
    reader = gr.GGUFReader(path)
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)
    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))
    compat = None
    arch_str = None
    arch_field = reader.get_field('general.architecture')
    if arch_field is not None:
        if len(arch_field.types) != 1 or arch_field.types[0
            ] != gr.GGUFValueType.STRING:
            raise TypeError(
                f'Bad type for GGUF general.architecture key: expected string, got {arch_field.types!r}'
                )
        arch_str = str(arch_field.parts[arch_field.data[-1]], encoding='utf-8')
        if arch_str not in CUS_ARCH_LIST and arch_str not in TXT_ARCH_LIST:
            raise ValueError(
                f'Unexpected architecture type in GGUF file, expected one of flux, sd1-3/sdxl, ltxv, hyvid, t5encoder, etc. but got {arch_str!r}'
                )
    else:
        compat = 'sd.cpp'
    state_dict = {}
    qtype_dict = {}
    for sd_key, tensor in tensors:
        tensor_name = tensor.name
        tensor_type_str = str(tensor.tensor_type)
        torch_tensor = torch.from_numpy(tensor.data)
        shape = get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            if compat == 'sd.cpp' and arch_str == 'sdxl':
                if any([tensor_name.endswith(x) for x in ('.proj_in.weight',
                    '.proj_out.weight')]):
                    while len(shape) > 2 and shape[-1] == 1:
                        shape = shape[:-1]
        if tensor.tensor_type in {gr.GGMLQuantizationType.F32, gr.
            GGMLQuantizationType.F16}:
            torch_tensor = torch_tensor.view(*shape)
        state_dict[sd_key] = GGMLTensor(torch_tensor, tensor_type=tensor.
            tensor_type, tensor_shape=shape)
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1
    qsd = {k: v for k, v in state_dict.items() if is_quantized(v)}
    if len(qsd) > 0:
        max_key = max(qsd.keys(), key=lambda k: qsd[k].numel())
        state_dict[max_key].is_largest_weight = True
    print('\nggml_sd_loader:')
    for k, v in qtype_dict.items():
        print(f' {k:30}{v:3}')
    if return_arch:
        return state_dict, arch_str
    return state_dict
T5_SD_MAP = {'enc.': 'encoder.', '.blk.': '.block.', 'token_embd': 'shared',
    'output_norm': 'final_layer_norm', 'attn_q': 'layer.0.SelfAttention.q',
    'attn_k': 'layer.0.SelfAttention.k', 'attn_v':
    'layer.0.SelfAttention.v', 'attn_o': 'layer.0.SelfAttention.o',
    'attn_norm': 'layer.0.layer_norm', 'attn_rel_b':
    'layer.0.SelfAttention.relative_attention_bias', 'ffn_up':
    'layer.1.DenseReluDense.wi_1', 'ffn_down': 'layer.1.DenseReluDense.wo',
    'ffn_gate': 'layer.1.DenseReluDense.wi_0', 'ffn_norm': 'layer.1.layer_norm'
    }
LLAMA_SD_MAP = {'blk.': 'model.layers.', 'attn_norm': 'input_layernorm',
    'attn_q': 'self_attn.q_proj', 'attn_k': 'self_attn.k_proj', 'attn_v':
    'self_attn.v_proj', 'attn_output': 'self_attn.o_proj', 'ffn_up':
    'mlp.up_proj', 'ffn_down': 'mlp.down_proj', 'ffn_gate': 'mlp.gate_proj',
    'ffn_norm': 'post_attention_layernorm', 'token_embd':
    'model.embed_tokens', 'output_norm': 'model.norm', 'output.weight':
    'lm_head.weight'}
def sd_map_replace(raw_sd, key_map):
    sd = {}
    for k, v in raw_sd.items():
        for s, d in key_map.items():
            k = k.replace(s, d)
        sd[k] = v
    return sd
def llama_permute(raw_sd, n_head, n_head_kv):
    sd = {}
    permute = lambda x, h: x.reshape(h, x.shape[0] // h // 2, 2, *x.shape[1:]
        ).swapaxes(1, 2).reshape(x.shape)
    for k, v in raw_sd.items():
        if k.endswith(('q_proj.weight', 'q_proj.bias')):
            v.data = permute(v.data, n_head)
        if k.endswith(('k_proj.weight', 'k_proj.bias')):
            v.data = permute(v.data, n_head_kv)
        sd[k] = v
    return sd
def load_gguf_clip(path):
    sd, arch = load_gguf_sd(path, return_arch=True)
    if arch in {'t5', 't5encoder'}:
        sd = sd_map_replace(sd, T5_SD_MAP)
    elif arch in {'llama'}:
        temb_key = 'token_embd.weight'
        if temb_key in sd and sd[temb_key].shape != (128320, 4096):
            print(
                'Warning! token_embd shape may be incorrect for llama 3 model!'
                )
        sd = sd_map_replace(sd, LLAMA_SD_MAP)
        sd = llama_permute(sd, 32, 8)
    else:
        pass
    return sd
class LoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        gguf_names = [x for x in folder_paths.get_filename_list('model_gguf')]
        return {'required': {'gguf_name': (gguf_names,)}}
    RETURN_TYPES = 'MODEL',
    FUNCTION = 'load_model'
    CATEGORY = 'gguf'
    TITLE = 'GGUF Loader'
    def load_model(self, gguf_name, dequant_dtype=None, patch_dtype=None,
        patch_on_device=None):
        ops = GGMLOps()
        if dequant_dtype in ('default', None):
            ops.Linear.dequant_dtype = None
        elif dequant_dtype in ['target']:
            ops.Linear.dequant_dtype = dequant_dtype
        else:
            ops.Linear.dequant_dtype = getattr(torch, dequant_dtype)
        if patch_dtype in ('default', None):
            ops.Linear.patch_dtype = None
        elif patch_dtype in ['target']:
            ops.Linear.patch_dtype = patch_dtype
        else:
            ops.Linear.patch_dtype = getattr(torch, patch_dtype)
        model_path = folder_paths.get_full_path('unet', gguf_name)
        sd = load_gguf_sd(model_path)
        model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=
            {'custom_operations': ops})
        if model is None:
            logging.error('ERROR UNSUPPORTED MODEL {}'.format(model_path))
            raise RuntimeError('ERROR: Could not detect model type of: {}'.
                format(model_path))
        model = GGUFModelPatcher.clone(model)
        model.patch_on_device = patch_on_device
        return model,
class LoaderGGUFAdvanced(LoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        model_names = [x for x in folder_paths.get_filename_list('model_gguf')]
        return {'required': {'gguf_name': (model_names,), 'dequant_dtype':
            (['default', 'target', 'float32', 'float16', 'bfloat16'], {
            'default': 'default'}), 'patch_dtype': (['default', 'target',
            'float32', 'float16', 'bfloat16'], {'default': 'default'}),
            'patch_on_device': ('BOOLEAN', {'default': False})}}
    TITLE = 'GGUF Loader (Advanced)'
CLIP_ENUM_MAP = {'stable_diffusion': 'STABLE_DIFFUSION', 'stable_cascade':
    'STABLE_CASCADE', 'stable_audio': 'STABLE_AUDIO', 'sdxl':
    'STABLE_DIFFUSION', 'sd3': 'SD3', 'flux': 'FLUX', 'mochi': 'MOCHI',
    'ltxv': 'LTXV', 'hunyuan_video': 'HUNYUAN_VIDEO', 'pixart': 'PIXART', 'cosmos': 'COSMOS'}
def get_clip_type(name):
    enum_name = CLIP_ENUM_MAP.get(name, None)
    if enum_name is None:
        raise ValueError(f'Unknown CLIP model type {name}')
    clip_type = getattr(comfy.sd.CLIPType, CLIP_ENUM_MAP[name], None)
    if clip_type is None:
        raise ValueError(f'Unsupported CLIP model type {name} (Update ComfyUI)'
            )
    return clip_type
class ClipLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'clip_name': (s.get_filename_list(),), 'type':
            (['stable_diffusion', 'stable_cascade', 'sd3', 'stable_audio',
            'mochi', 'ltxv', 'pixart', 'cosmos'],)}}
    RETURN_TYPES = 'CLIP',
    FUNCTION = 'load_clip'
    CATEGORY = 'gguf'
    TITLE = 'GGUF CLIPLoader'
    @classmethod
    def get_filename_list(s):
        files = []
        files += folder_paths.get_filename_list('clip')
        files += folder_paths.get_filename_list('clip_gguf')
        return sorted(files)
    def load_data(self, ckpt_paths):
        clip_data = []
        for p in ckpt_paths:
            if p.endswith('.gguf'):
                sd = load_gguf_clip(p)
            else:
                sd = comfy.utils.load_torch_file(p, safe_load=True)
            clip_data.append(sd)
        return clip_data
    def load_patcher(self, clip_paths, clip_type, clip_data):
        clip = comfy.sd.load_text_encoder_state_dicts(clip_type=clip_type,
            state_dicts=clip_data, model_options={'custom_operations':
            GGMLOps, 'initial_device': comfy.model_management.
            text_encoder_offload_device()}, embedding_directory=
            folder_paths.get_folder_paths('embeddings'))
        clip.patcher = GGUFModelPatcher.clone(clip.patcher)
        return clip
    def load_clip(self, clip_name, type='stable_diffusion'):
        clip_path = folder_paths.get_full_path('clip', clip_name)
        return self.load_patcher([clip_path], get_clip_type(type), self.
            load_data([clip_path])),
class DualClipLoaderGGUF(ClipLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        file_options = s.get_filename_list(),
        return {'required': {'clip_name1': file_options, 'clip_name2':
            file_options, 'type': (('sdxl', 'sd3', 'flux'),)}}
    TITLE = 'GGUF DualCLIPLoader'
    def load_clip(self, clip_name1, clip_name2, type):
        clip_path1 = folder_paths.get_full_path('clip', clip_name1)
        clip_path2 = folder_paths.get_full_path('clip', clip_name2)
        clip_paths = clip_path1, clip_path2
        return self.load_patcher(clip_paths, get_clip_type(type), self.
            load_data(clip_paths)),
class TripleClipLoaderGGUF(ClipLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        file_options = s.get_filename_list(),
        return {'required': {'clip_name1': file_options, 'clip_name2':
            file_options, 'clip_name3': file_options}}
    TITLE = 'GGUF TripleCLIPLoader'
    def load_clip(self, clip_name1, clip_name2, clip_name3, type='sd3'):
        clip_path1 = folder_paths.get_full_path('clip', clip_name1)
        clip_path2 = folder_paths.get_full_path('clip', clip_name2)
        clip_path3 = folder_paths.get_full_path('clip', clip_name3)
        clip_paths = clip_path1, clip_path2, clip_path3
        return self.load_patcher(clip_paths, get_clip_type(type), self.
            load_data(clip_paths)),
QUANTIZATION_THRESHOLD = 1024
REARRANGE_THRESHOLD = 512
MAX_TENSOR_NAME_LENGTH = 127
class ModelTemplate:
    arch = 'invalid'
    shape_fix = False
    keys_detect = []
    keys_banned = []
class ModelHYVID(ModelTemplate):
    arch = 'hyvid'
    keys_detect = [('transformer_blocks.0.attn.norm_added_k.weight',), (
        'double_blocks.0.img_attn.proj.weight',)]
    keys_banned = ['transformer_blocks.0.attn.norm_added_k.weight']
class ModelSD3(ModelTemplate):
    arch = 'sd3'
    keys_detect = [('transformer_blocks.0.attn.add_q_proj.weight',), (
        'joint_blocks.0.x_block.attn.qkv.weight',)]
    keys_banned = ['transformer_blocks.0.attn.add_q_proj.weight']
class ModelAura(ModelTemplate):
    arch = 'aura'
    keys_detect = [('double_layers.3.modX.1.weight',), (
        'joint_transformer_blocks.3.ff_context.out_projection.weight',)]
    keys_banned = [
        'joint_transformer_blocks.3.ff_context.out_projection.weight']
class ModelLTXV(ModelTemplate):
    arch = 'ltxv'
    keys_detect = [('adaln_single.emb.timestep_embedder.linear_2.weight',
        'transformer_blocks.27.scale_shift_table',
        'caption_projection.linear_2.weight')]
class ModelSDXL(ModelTemplate):
    arch = 'sdxl'
    shape_fix = True
    keys_detect = [('down_blocks.0.downsamplers.0.conv.weight',
        'add_embedding.linear_1.weight'), ('input_blocks.3.0.op.weight',
        'input_blocks.6.0.op.weight', 'output_blocks.2.2.conv.weight',
        'output_blocks.5.2.conv.weight'), ('label_emb.0.0.weight',)]
class ModelSD1(ModelTemplate):
    arch = 'sd1'
    shape_fix = True
    keys_detect = [('down_blocks.0.downsamplers.0.conv.weight',), (
        'input_blocks.3.0.op.weight', 'input_blocks.6.0.op.weight',
        'input_blocks.9.0.op.weight', 'output_blocks.2.1.conv.weight',
        'output_blocks.5.2.conv.weight', 'output_blocks.8.2.conv.weight')]
arch_list = [ModelSD3, ModelAura, ModelLTXV, ModelHYVID, ModelSDXL, ModelSD1]
def is_model_arch(model, state_dict):
    matched = False
    invalid = False
    for match_list in model.keys_detect:
        if all(key in state_dict for key in match_list):
            matched = True
            invalid = any(key in state_dict for key in model.keys_banned)
            break
    assert not invalid, 'Model architecture not allowed for conversion! (i.e. reference VS diffusers format)'
    return matched
def detect_arch(state_dict):
    model_arch = None
    for arch in arch_list:
        if is_model_arch(arch, state_dict):
            model_arch = arch
            break
    assert model_arch is not None, 'Unknown model architecture!'
    return model_arch
def load_state_dict(path):
    state_dict = load_file(path)
    prefix = None
    for pfx in ['model.diffusion_model.', 'model.']:
        if any([x.startswith(pfx) for x in state_dict.keys()]):
            prefix = pfx
            break
    sd = {}
    for k, v in state_dict.items():
        if prefix and prefix not in k:
            continue
        if prefix:
            k = k.replace(prefix, '')
        sd[k] = v
    return sd
def load_model(path):
    state_dict = load_state_dict(path)
    model_arch = detect_arch(state_dict)
    print(f'* Architecture detected from input: {model_arch.arch}')
    writer = GGUFWriter(path=None, arch=model_arch.arch)
    return writer, state_dict, model_arch
def handle_tensors(args, writer, state_dict, model_arch):
    name_lengths = tuple(sorted(((key, len(key)) for key in state_dict.keys
        ()), key=lambda item: item[1], reverse=True))
    if not name_lengths:
        return
    max_name_len = name_lengths[0][1]
    if max_name_len > MAX_TENSOR_NAME_LENGTH:
        bad_list = ', '.join(f'{key!r} ({namelen})' for key, namelen in
            name_lengths if namelen > MAX_TENSOR_NAME_LENGTH)
        raise ValueError(
            f'Can only handle tensor names up to {MAX_TENSOR_NAME_LENGTH} characters. Tensors exceeding the limit: {bad_list}'
            )
    for key, data in loading(state_dict.items()):
        old_dtype = data.dtype
        if data.dtype == torch.bfloat16:
            data = data.to(torch.float32).numpy()
        elif data.dtype in [getattr(torch, 'float8_e4m3fn', '_invalid'),
            getattr(torch, 'float8_e5m2', '_invalid')]:
            data = data.to(torch.float16).numpy()
        else:
            data = data.numpy()
        n_dims = len(data.shape)
        data_shape = data.shape
        data_qtype = getattr(GGMLQuantizationType, 'BF16' if old_dtype ==
            torch.bfloat16 else 'F16')
        n_params = 1
        for dim_size in data_shape:
            n_params *= dim_size
        blacklist = {'time_embedding.', 'add_embedding.', 'time_in.',
            'txt_in.', 'vector_in.', 'img_in.', 'guidance_in.', 'final_layer.'}
        if old_dtype in (torch.float32, torch.bfloat16):
            if n_dims == 1:
                data_qtype = GGMLQuantizationType.F32
            elif n_params <= QUANTIZATION_THRESHOLD:
                data_qtype = GGMLQuantizationType.F32
            elif '.weight' in key and any(x in key for x in blacklist):
                data_qtype = GGMLQuantizationType.F32
        if (model_arch.shape_fix and n_dims > 1 and n_params >=
            REARRANGE_THRESHOLD and (n_params / 256).is_integer() and not (
            data.shape[-1] / 256).is_integer()):
            orig_shape = data.shape
            data = data.reshape(n_params // 256, 256)
            writer.add_array(f'comfy.gguf.orig_shape.{key}', tuple(int(dim) for
                dim in orig_shape))
        try:
            data = quantize(data, data_qtype)
        except (AttributeError, QuantError) as e:
            loading.write(f'falling back to F16: {e}')
            data_qtype = GGMLQuantizationType.F16
            data = quantize(data, data_qtype)
        new_name = key
        shape_str = f"{{{', '.join(str(n) for n in reversed(data.shape))}}}"
        loading.write(
            f"{f'%-{max_name_len + 4}s' % f'{new_name}'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}"
            )
        writer.add_tensor(new_name, data, raw_dtype=data_qtype)
if 'select_safetensors' not in folder_paths.folder_names_and_paths:
    orig = folder_paths.folder_names_and_paths.get('diffusion_models',
        folder_paths.folder_names_and_paths.get('checkpoints', [[], set()]))
    folder_paths.folder_names_and_paths['select_safetensors'] = orig[0], {
        '.safetensors'}
class GGUFSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'select_safetensors': (s.get_filename_list(),)}}
    RETURN_TYPES = ()
    FUNCTION = 'save'
    OUTPUT_NODE = True
    CATEGORY = 'gguf'
    TITLE = 'GGUF Convertor (Alpha)'
    @classmethod
    def get_filename_list(s):
        files = []
        files += folder_paths.get_filename_list('select_safetensors')
        return sorted(files)
    def save(self, select_safetensors):
        path = folder_paths.get_full_path('select_safetensors',
            select_safetensors)
        writer, state_dict, model_arch = load_model(path)
        writer.add_quantization_version(GGML_QUANT_VERSION)
        if next(iter(state_dict.values())).dtype == torch.bfloat16:
            output_path = (
                f'{self.output_dir}/{os.path.splitext(select_safetensors)[0]}-bf16.gguf'
                )
            writer.add_file_type(LlamaFileType.MOSTLY_BF16)
        else:
            output_path = (
                f'{self.output_dir}/{os.path.splitext(select_safetensors)[0]}-f16.gguf'
                )
            writer.add_file_type(LlamaFileType.MOSTLY_F16)
        if os.path.isfile(output_path):
            input('Output exists enter to continue or ctrl+c to abort!')
        handle_tensors(output_path, writer, state_dict, model_arch)
        writer.write_header_to_file(path=output_path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
        return {}
def quantize_to_fp8(tensor):
    if tensor.dtype != torch.bfloat16:
        raise ValueError('Input tensor must be in BF16 format.')
    tensor = tensor.to(torch.float16)
    fp8_max = 240.0
    fp8_min = -fp8_max
    clamped_tensor = tensor.clamp(min=fp8_min, max=fp8_max)
    scale = fp8_max / torch.max(torch.abs(clamped_tensor))
    quantized_tensor = torch.round(clamped_tensor * scale) / scale
    return quantized_tensor.to(torch.float8_e4m3fn)
class TENSORCut:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'select_safetensors': (s.get_filename_list(),)}}
    RETURN_TYPES = ()
    FUNCTION = 'cut'
    OUTPUT_NODE = True
    CATEGORY = 'gguf'
    TITLE = 'TENSOR Cutter (Beta)'
    @classmethod
    def get_filename_list(s):
        files = []
        files += folder_paths.get_filename_list('select_safetensors')
        return sorted(files)
    def cut(self, select_safetensors):
        input_file = folder_paths.get_full_path('select_safetensors',
            select_safetensors)
        output_file = (
            f'{self.output_dir}/{os.path.splitext(select_safetensors)[0]}_fp8_e4m3fn.safetensors'
            )
        data = load_file(input_file)
        quantized_data = {}
        print('Starting quantization process...')
        for key, tensor in loading(data.items(), desc='Quantizing tensors',
            unit='tensor'):
            tensor = tensor.to(dtype=torch.bfloat16, device='cuda')
            quantized_tensor = quantize_to_fp8(tensor)
            quantized_data[key] = quantized_tensor.cpu()
        save_file(quantized_data, output_file)
        print(f'Quantized safetensors saved to {output_file}.')
        return {}
NODE_CLASS_MAPPINGS = {
    "LoaderGGUF": LoaderGGUF,
    "ClipLoaderGGUF": ClipLoaderGGUF,
    "DualClipLoaderGGUF": DualClipLoaderGGUF,
    "TripleClipLoaderGGUF": TripleClipLoaderGGUF,
    "LoaderGGUFAdvanced": LoaderGGUFAdvanced,
    "TENSORCut": TENSORCut,
    "GGUFSave": GGUFSave,
}
