import struct
class ReaderError(Exception):
    """Exception raised for errors in GGUF reader"""
class GGUFReader:
    GGUF_FORMAT = b'GGUF'
    VALUE_FORMATS = {(0): 'B', (1): 'b', (2): 'H', (3): 'h', (4): 'I', (5):
        'i', (6): 'f', (7): '?', (10): 'Q', (11): 'q', (12): 'd'}
    TENSOR_TYPES = {(0): 'GGML_TYPE_F32', (1): 'GGML_TYPE_F16', (2):
        'GGML_TYPE_Q4_0', (3): 'GGML_TYPE_Q4_1', (4):
        'GGML_TYPE_Q4_1_SOME_F16', (5): 'GGML_TYPE_TYPE_Q4_2', (6):
        'GGML_TYPE_Q4_3', (7): 'GGML_TYPE_Q8_0', (8): 'GGML_TYPE_Q5_0', (9):
        'GGML_TYPE_Q5_1', (10): 'GGML_TYPE_Q2_K', (11): 'GGML_TYPE_Q3_K_S',
        (12): 'GGML_TYPE_Q3_K_M', (13): 'GGML_TYPE_Q3_K_L', (14):
        'GGML_TYPE_Q4_K_S', (15): 'GGML_TYPE_Q4_K_M', (16):
        'GGML_TYPE_Q5_K_S', (17): 'GGML_TYPE_Q5_K_M', (18):
        'GGML_TYPE_Q6_K', (19): 'GGML_TYPE_IQ2_XXS', (20):
        'GGML_TYPE_IQ2_XS', (21): 'GGML_TYPE_Q2_K_S', (22):
        'GGML_TYPE_IQ3_XS', (23): 'GGML_TYPE_IQ3_XXS', (24):
        'GGML_TYPE_IQ1_S', (25): 'GGML_TYPE_IQ4_NL', (26):
        'GGML_TYPE_IQ3_S', (27): 'GGML_TYPE_IQ3_M', (28): 'GGML_TYPE_IQ2_S',
        (29): 'GGML_TYPE_IQ2_M', (30): 'GGML_TYPE_IQ4_XS', (31):
        'GGML_TYPE_IQ1_M', (32): 'GGML_TYPE_BF16', (33):
        'GGML_TYPE_Q4_0_4_4', (34): 'GGML_TYPE_Q4_0_4_8', (35):
        'GGML_TYPE__Q4_0_8_8', (36): 'GGML_TYPE__TQ1_0', (37):
        'GGML_TYPE__TQ2_0'}
    def __init__(self, file_path):
        """Initialize the GGUF reader"""
        self.file_path = file_path
        self.version = None
        self.format = None
        self.tensors_info = None
        self.metadata = None
        self.alignment = None
    def read(self):
        """Read the GGUF file."""
        with open(self.file_path, 'rb') as f:
            self.format = f.read(4)
            if self.format != self.GGUF_FORMAT:
                raise ReaderError('Invalid format')
            self.version = struct.unpack('I', f.read(4))[0]
            tensor_count = struct.unpack('Q', f.read(8))[0]
            metadata_kv_count = struct.unpack('Q', f.read(8))[0]
            self.metadata = {}
            for _ in range(metadata_kv_count):
                key, value = self._read_metadata_kv(f)
                self.metadata[key] = value
            self.alignment = self.metadata.get('general.alignment', 1)
            self.tensors_info = []
            for _ in range(tensor_count):
                tensor_info = self._read_tensor_info(f)
                self.tensors_info.append(tensor_info)
    def _read_string(self, f):
        """Read a string from the file"""
        length = struct.unpack('Q', f.read(8))[0]
        return f.read(length).decode('utf-8')
    def _read_metadata_kv(self, f):
        """Read a metadata key-value pair from the file"""
        key = self._read_string(f)
        value_type = struct.unpack('I', f.read(4))[0]
        value = self._read_value(f, value_type)
        return key, value
    def _read_value(self, f, value_type):
        """Read a value of the given type from the file"""
        if value_type in self.VALUE_FORMATS:
            return struct.unpack(self.VALUE_FORMATS[value_type], f.read(
                struct.calcsize(self.VALUE_FORMATS[value_type])))[0]
        if value_type == 8:
            return self._read_string(f)
        if value_type == 9:
            array_type = struct.unpack('I', f.read(4))[0]
            array_len = struct.unpack('Q', f.read(8))[0]
            return [self._read_value(f, array_type) for _ in range(array_len)]
        raise ReaderError('Unsupported value type')
    def _read_tensor_info(self, f):
        """Read tensor information from the file"""
        name = self._read_string(f)
        n_dimensions = struct.unpack('I', f.read(4))[0]
        dimensions = struct.unpack(f'{n_dimensions}Q', f.read(8 * n_dimensions)
            )
        tensor_type = struct.unpack('I', f.read(4))[0]
        offset = struct.unpack('Q', f.read(8))[0]
        return {'name': name, 'n_dimensions': n_dimensions, 'dimensions':
            dimensions, 'type': tensor_type, 'offset': offset}
    def load_tensors(self):
        """Load the tensors from the file"""
        tensors = []
        with open(self.file_path, 'rb') as f:
            for tensor_info in self.tensors_info:
                f.seek(tensor_info['offset'])
                tensor_data = f.read(tensor_info['n_dimensions'])
                if f.tell() % self.alignment != 0:
                    f.read(self.alignment - f.tell() % self.alignment)
                tensors.append(tensor_data)
        return tensors
    def print(self):
        """Print the file details"""
        print(f'Version: {self.version}')
        print(f'Format: {self.format}')
        print('Tensors Info:')
        for tensor_info in self.tensors_info:
            print(
                f"  Name: {tensor_info['name']},\tShape: {tensor_info['dimensions']},\tType: {self.TENSOR_TYPES[tensor_info['type']]},\tOffset: {tensor_info['offset']}"
                )
        print('Metadata:')
        for key, value in self.metadata.items():
            if isinstance(value, list) and len(value) > 50:
                print(
                    f'  {key}: {value[:50]}... ({len(value) - 50} more elements)'
                    )
            else:
                print(f'  {key}: {value}')