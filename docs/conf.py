from pathlib import Path
import sys
import types


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


class _TorchTensorMock:
    def __init__(self, *args, **kwargs):
        pass

    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self

    def __sub__(self, other):
        return self

    def __rsub__(self, other):
        return self

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __truediv__(self, other):
        return self

    def __rtruediv__(self, other):
        return self

    def __pow__(self, other):
        return self

    def __rpow__(self, other):
        return self

    def __getitem__(self, key):
        return self

    def to(self, *args, **kwargs):
        return self

    def sum(self, *args, **kwargs):
        return self


class _Module:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        if hasattr(self, "forward"):
            return self.forward(*args, **kwargs)
        return None

    def apply(self, *args, **kwargs):
        return self


_Module.__module__ = "torch.nn"


class _ModuleList(list):
    pass


_ModuleList.__module__ = "torch.nn"


class _Layer:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return args[0] if args else _TorchTensorMock()


_Layer.__module__ = "torch.nn"


def _torch_factory(*args, **kwargs):
    return _TorchTensorMock()


torch_mock = types.ModuleType("torch")
torch_mock.Tensor = _TorchTensorMock
torch_mock.arange = _torch_factory
torch_mock.ones = _torch_factory
torch_mock.exp = _torch_factory
torch_mock.erf = _torch_factory
torch_mock.cdist = _torch_factory
torch_mock.vstack = _torch_factory
torch_mock.matmul = _torch_factory
torch_mock.split = lambda *args, **kwargs: (_TorchTensorMock(), _TorchTensorMock())
torch_mock.softmax = _torch_factory

optim_mock = types.ModuleType("torch.optim")
dist_mock = types.ModuleType("torch.distributed")
multiprocessing_mock = types.ModuleType("torch.multiprocessing")
utils_mock = types.ModuleType("torch.utils")
data_mock = types.ModuleType("torch.utils.data")
cuda_mock = types.SimpleNamespace(
    manual_seed=_torch_factory,
    manual_seed_all=_torch_factory,
)
backends_mock = types.SimpleNamespace(
    cudnn=types.SimpleNamespace(deterministic=False, benchmark=False)
)

nn_mock = types.ModuleType("torch.nn")
nn_mock.Module = _Module
nn_mock.ModuleList = _ModuleList
nn_mock.Sequential = _Layer
nn_mock.Linear = _Layer
nn_mock.LeakyReLU = _Layer
nn_mock.GELU = _Layer
nn_mock.LayerNorm = _Layer
nn_mock.Dropout = _Layer
nn_mock.Embedding = _Layer
nn_mock.Parameter = _torch_factory

functional_mock = types.ModuleType("torch.nn.functional")
functional_mock.softmax = _torch_factory

torch_mock.nn = nn_mock
torch_mock.optim = optim_mock
torch_mock.distributed = dist_mock
torch_mock.multiprocessing = multiprocessing_mock
torch_mock.utils = utils_mock
torch_mock.cuda = cuda_mock
torch_mock.backends = backends_mock
sys.modules.setdefault("torch", torch_mock)
sys.modules.setdefault("torch.nn", nn_mock)
sys.modules.setdefault("torch.nn.functional", functional_mock)
sys.modules.setdefault("torch.optim", optim_mock)
sys.modules.setdefault("torch.distributed", dist_mock)
sys.modules.setdefault("torch.multiprocessing", multiprocessing_mock)
sys.modules.setdefault("torch.utils", utils_mock)
sys.modules.setdefault("torch.utils.data", data_mock)

project = "SpaWeaver"
author = "SpaWeaver contributors"
copyright = "2026, SpaWeaver contributors"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
nbsphinx_execute = "never"

autodoc_mock_imports = [
    "anndata",
    "cv2",
    "matplotlib",
    "numpy",
    "pandas",
    "PIL",
    "timm",
    "scanpy",
    "scipy",
    "skimage",
    "sklearn",
    "torchvision",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = []
